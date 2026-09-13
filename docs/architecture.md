# Architecture

## Model execution graph

`proj_mnist.tflite` is an int8-quantized CNN with uint8 input and output tensors. The operator set is exactly four builtins: `QUANTIZE`, `CONV_2D`, `MEAN`, `FULLY_CONNECTED`. The sketch registers only those four via `MicroMutableOpResolver<4>`.

| # | Op | Input | Filter | Output | Notes |
|---|---|---|---|---|---|
| 0 | QUANTIZE | uint8 [1,28,28,1] | - | int8 [1,28,28,1] | zero-point shift (-128) |
| 1 | CONV_2D (Conv1) | int8 [1,28,28,1] | [8,3,3,1] | int8 [1,14,14,8] | stride 2, SAME, ~14.1K MAC |
| 2 | CONV_2D (Conv2) | int8 [1,14,14,8] | [16,3,3,8] | int8 [1,14,14,16] | stride 1, SAME, ~225.8K MAC |
| 3 | CONV_2D (Conv3) | int8 [1,14,14,16] | [32,3,3,16] | int8 [1,7,7,32] | stride 2, SAME, ~225.8K MAC |
| 4 | CONV_2D (Conv4) | int8 [1,7,7,32] | [64,3,3,32] | int8 [1,7,7,64] | stride 1, SAME, ~903.2K MAC |
| 5 | CONV_2D (Conv5) | int8 [1,7,7,64] | [128,3,3,64] | int8 [1,4,4,128] | stride 2, SAME, ~1179.6K MAC |
| 6 | MEAN (axes 1,2) | int8 [1,4,4,128] | - | int8 [1,128] | global average pool |
| 7 | FULLY_CONNECTED | int8 [1,128] | [10,128] | int8 [1,10] | |
| 8 | QUANTIZE | int8 [1,10] | - | uint8 [1,10] | zero-point shift (+128) |

Shapes are the ones asserted by the fast-path guards in `tflm_patches/cmsis_nn/conv.cpp` (`IsProject2Conv1..5`) and `tflm_patches/kernels/reduce.cpp`. The MAC figures are from the project report.

Conv4 and Conv5 account for roughly 80% of the MACs and were the first bottleneck candidates. Profiling showed that latency did not track MAC count alone; the remaining factors were:

- memory access pattern of filter reads (strided per output channel in the generic kernel)
- generic kernel overhead (runtime shape resolution, int4-unpack wrapper, type switches)
- branches inside inner loops
- requantization cost per output element
- filter layout and code locality
- whether the Cortex-M4 DSP extension (`SMLAD`) is actually used

## Runtime layering (final build)

```
proj2_final_best.ino
  ├─ tflite::GetModel(g_proj_mnist_tflite)
  ├─ MicroMutableOpResolver<4>  { Quantize, Conv2D, Mean, FullyConnected }
  ├─ MicroInterpreter(model, resolver, tensor_arena[160 KiB])
  └─ RunBatch30Release()
        └─ per image: memcpy -> Invoke() -> Argmax10UInt8
                          │
                          ▼
           patched TFLM kernels (selected by tensor shape/type, else stock path)
             quantize.cpp    EvalProject2FastQuantize     (784 uint8->int8, 10 int8->uint8)
             conv.cpp        EvalInt8DirectFilter
                               ├─ Conv1  EvalProject2Conv1DirectSparse
                               ├─ Conv2  EvalProject2Conv2PairMajorSparseV1
                               ├─ Conv3  EvalProject2Conv3PairMajorSparseV1
                               ├─ Conv4  EvalProject2Conv4PairMajorSparseV1
                               ├─ Conv5  EvalProject2Conv5PairMajorSparseV4
                               └─ (fallback) EvalProject2Conv45FixedCmsis -> CMSIS-NN
             reduce.cpp      EvalProject2Mean4x4x128Fast
             fully_connected EvalInt8 direct -> arm_fully_connected_s8 (CMSIS-NN)
```

## Pair-major prepacked filter layout

Stock CMSIS-NN / TFLM Conv2D reads the filter tensor in `[oc][kh][kw][ic]` order, which means for each output channel the kernel walks the whole 3x3xIC window. The specialized kernels instead use a build-time copy of each filter tensor laid out as:

```
[(kh, kw, ic_pair)] [oc]     one int32 per element = two int8 weights as int16 lanes
```

For a given input position and input-channel pair, the two activations (offset-corrected, so a zero-point value becomes 0) are packed into one int32 `(x1 << 16) | x0`. The inner loop then walks output channels contiguously:

```
acc[oc] = SMLAD(xpack, w_pairmajor[pair][oc], acc[oc])
```

One `SMLAD` performs two 16x16 multiplies and an add. Before the loop, if `x0 == 0 && x1 == 0` the whole pair is skipped (zero-contribution MAC elimination). The tables live in flash (`project2_conv*_filter_pairmajor_packed.h`, generated from the unchanged `.tflite`), so RAM use does not grow.

Requantization after accumulation uses TFLM's own `MultiplyByQuantizedMultiplier` with the per-channel multiplier/shift from the model, then output offset and activation clamp, so the math matches the reference kernel.

## Zero activations

With ReLU-style quantization the activation zero point is -128, so an int8 input value of -128 means a real value of exactly 0. Profiling the activation distribution at the Conv4 and Conv5 inputs showed about 54% of values at the zero point. Skipping those pairs removes real work without changing any output value. The hybrid variants in `conv.cpp` (`*_HYBRID_V1`, `CONV5_SPARSE_HYBRID_V2`) count zeros at runtime and fall back to dense CMSIS-NN below a threshold; the final build uses the always-sparse variants because the measured input distribution made the count overhead unnecessary.
