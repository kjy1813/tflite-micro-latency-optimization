# TFLite Micro Latency Optimization on Arduino Nano 33 BLE

> Optimizing the runtime execution path of a fixed MNIST TFLite model without changing its graph, weights, or quantization parameters.

| Metric | Baseline | Final |
|---|---:|---:|
| Average latency (30-image batch) | 239.60 ms | **95.17 ms** |
| Accuracy (30 fixed test images) | 100% | **100% (30/30)** |
| Speedup | 1.00x | **2.52x** |
| Latency reduction | - | **60.28% (144.43 ms)** |
| Model file (`proj_mnist.tflite`) | unchanged | unchanged |

![Baseline vs final latency](assets/latency_comparison.png)

![C++](https://img.shields.io/badge/C%2B%2B-embedded-blue)
![TensorFlow Lite Micro](https://img.shields.io/badge/TensorFlow%20Lite-Micro-orange)
![Arduino](https://img.shields.io/badge/Arduino-Nano%2033%20BLE-teal)
![ARM Cortex-M4](https://img.shields.io/badge/ARM-Cortex--M4F-lightgrey)

## Highlights

- **239.60 ms to 95.17 ms** end-to-end inference latency on an ARM Cortex-M4F MCU, with the `.tflite` file, layer graph, weights, and quantization parameters left untouched.
- **Fixed-shape kernel specialization**: pair-major prepacked filters, DSP `SMLAD` dual-MAC accumulation, and zero-contribution activation skipping replaced the generic TFLM/CMSIS-NN convolution path for every conv layer.
- **The fastest version was rejected.** A 92.51 ms variant dropped accuracy to 22/30 (73.3%). Each change was re-isolated and only the accuracy-preserving set was kept, giving 95.17 ms at 30/30.

## Overview

Course project (Embedded System Design, 2026-05-28 to 2026-06-18). The task: take a provided quantized MNIST classifier (`proj_mnist.tflite`) running under TensorFlow Lite Micro on an Arduino Nano 33 BLE and minimize inference latency **without touching the model**.

Constraints:

- `.tflite` model file, layer graph, filter shapes, strides, padding, and tensor shapes must stay the same
- weight, bias, scale, and zero-point values must stay the same
- accuracy must not regress
- the only optimization target is the TFLM runtime, its kernels, and the memory/execution path

So this is not a retraining, pruning, or model-swap project. It is a runtime profiling and kernel-engineering project on a fixed workload.

## System / Architecture

Model execution path (all tensors int8 after the first Quantize; shapes verified against the kernel dispatch code):

```
uint8 [1,28,28,1]  input image
        │
        ▼  QUANTIZE   uint8 -> int8 (zero-point shift)
Conv1   3x3, stride 2, SAME      -> [1,14,14,8]      ~14.1K MAC
Conv2   3x3, stride 1, SAME      -> [1,14,14,16]     ~225.8K MAC
Conv3   3x3, stride 2, SAME      -> [1,7,7,32]       ~225.8K MAC
Conv4   3x3, stride 1, SAME      -> [1,7,7,64]       ~903.2K MAC
Conv5   3x3, stride 2, SAME      -> [1,4,4,128]      ~1179.6K MAC
MEAN    over H,W (global avg pool) -> [1,128]
FULLY_CONNECTED                  -> [1,10]
        │
        ▼  QUANTIZE   int8 -> uint8
uint8 [10] class scores  ->  argmax
```

Where the time goes was not fully explained by MAC count. On this MCU, memory access pattern, generic-kernel branch overhead, requantization, filter layout, code locality, and use of the Cortex-M4 DSP extension mattered as much as arithmetic count. See [docs/architecture.md](docs/architecture.md).

Runtime layering after optimization:

```
Arduino sketch (firmware/proj2_final_best)
  └─ MicroMutableOpResolver<4>: QUANTIZE, CONV_2D, MEAN, FULLY_CONNECTED
       └─ patched TFLM kernels (tflm_patches/)
            ├─ conv.cpp            fixed-shape Conv1..Conv5 fast paths
            ├─ fully_connected.cpp direct int8 FC path
            ├─ quantize.cpp        fixed-size quantize fast paths
            └─ reduce.cpp          fixed-shape MEAN fast path
                 └─ CMSIS-NN / ARM DSP (SMLAD) on nRF52840 Cortex-M4F
```

## Environment

| Category | Stack |
|---|---|
| Hardware | Arduino Nano 33 BLE (nRF52840, ARM Cortex-M4F, 64 MHz, 256 KB RAM, 1 MB flash) |
| Language | C / C++ (firmware and kernels), Python (result analysis) |
| Framework | TensorFlow Lite Micro (Arduino `TensorFlowLite` library), CMSIS-NN |
| Toolchain | Arduino IDE, `arduino:mbed_nano:nano33ble` core, GCC `-O3` via `platform.local.txt` |
| Model | `proj_mnist.tflite` (int8 CNN, uint8 I/O), provided by the course |
| Measurement | `micros()` around input copy + `Invoke()` + argmax, 10 warm-up runs, 30-image batch |

## Implementation

Compile-time configuration of the final build (`platform/platform.local.txt`):

| Flag | Effect |
|---|---|
| `-O3 -fomit-frame-pointer -fno-unwind-tables -fno-asynchronous-unwind-tables -fno-common` | Compiler optimization level for the whole core and library |
| `-DARM_MATH_DSP=1 -DCMSIS_NN=1` | Keep CMSIS-NN kernels and enable DSP intrinsics (`SMLAD`) |
| `-DPROJECT2_CONV1_DIRECT_SPARSE=1` | Conv1: direct fixed-shape kernel on the original filter tensor, skipping zero input contributions |
| `-DPROJECT2_CONV2_PAIRMAJOR_SPARSE_ALWAYS=1` (also CONV3, CONV4) | Conv2..Conv4: pair-major prepacked filter kernels, always on |
| `-DPROJECT2_CONV5_SPARSE_V1=1` | Conv5: pair-major sparse kernel, always on (the hybrid threshold variant is compiled out) |
| `-DPROJECT2_DIRECTION_C_CONV45_FIXED=1`, `_DIRECT_PRIMITIVE=0` | Fallback fixed-dimension CMSIS-NN call for Conv4/Conv5 if a shape check fails |
| `-DPROJECT2_DISABLE_KERNEL_TIMER=1 -DPROJECT2_ENABLE_KERNEL_PROFILING=0` | Per-layer profiling counters removed from the final measurement build |

What each fast path does (all of them check tensor shape and type first and fall back to the stock TFLM implementation otherwise):

1. **Conv2D dispatch** (`tflm_patches/cmsis_nn/conv.cpp`): `Register_CONV_2D` routes straight to an int8 path, removing the int4-unpack wrapper object that stock TFLM builds on every call. Inside, five shape-matched fast paths cover Conv1..Conv5.
2. **Conv1 direct path**: 28x28x1 input, 8 filters, stride 2. Reads the original filter tensor and skips input pixels whose dequantized value is zero.
3. **Conv2..Conv5 pair-major kernels**: each layer's int8 filter is prepacked at build time into a flash table laid out `[kh_kw_input_channel_pair][output_channel]` (`project2_conv*_filter_pairmajor_packed.h`, generated from the unchanged `.tflite`). Two int8 activations are packed as int16 lanes and multiplied against two packed weights with one `SMLAD`, accumulating contiguously across output channels instead of strided weight reads. When both activations in a pair are exactly the zero point (q == -128 with input offset 128), the pair contributes nothing and is skipped.
4. **Quantize** (`kernels/quantize.cpp`): the 784-element uint8 to int8 input quantize and the 10-element int8 to uint8 output quantize become plain offset shifts (unrolled by 8 for the input).
5. **Mean** (`kernels/reduce.cpp`): the `[1,4,4,128] -> [1,128]` global average pool uses a precomputed fixed-point multiplier for this tensor's scale pair (`input_scale / (16 * output_scale)`), summing 16 spatial positions per channel with NHWC stride 128.
6. **Fully connected** (`cmsis_nn/fully_connected.cpp`): skips the generic type switch and int4-unpack wrapper and calls `arm_fully_connected_s8` directly.
7. **Sketch level** (`firmware/`): only the four ops used by the model are registered, debug prints and per-prediction output are compiled out, and the sketch itself applies `#pragma GCC optimize("O3")` and `("unroll-loops")`.

## Challenge

After the "obvious" work (CMSIS-NN path, trimming op registration, removing measurement overhead, compiler flags) latency settled around 136 ms. Further changes in the same style produced only small gains. Conv4 and Conv5 dominated the MAC count, so the natural question was how to compute the same thing faster. That question stalled.

## Approach

The question was changed to **"does every MAC actually have to be executed?"**

- Profiled `Invoke()` at layer and kernel level: generic kernel branches, memory access, requantization, per-layer hot paths, and the activation distribution feeding each layer.
- Observed that roughly **54% of the Conv4 / Conv5 input activations sit at the zero point**, contributing exactly zero to the accumulation. Those MACs can be skipped without changing the result.
- Since the model graph and every tensor shape are fixed, the generic kernels' runtime shape handling could be replaced with shape-specialized paths and a weight layout chosen for the access pattern (pair-major, contiguous across output channels).
- Each change was compiled with its own flag (`PROJECT2_*`), measured in isolation with the same 30-image batch, and kept only if end-to-end latency dropped and accuracy stayed at 30/30.

Full history in [docs/optimization_history.md](docs/optimization_history.md).

## Validation

Every candidate build ran the same protocol (`RunBatch30Release()` in the sketch):

- 10 warm-up inferences on image 0
- 30 fixed MNIST test images with known labels (`test_images_30.h`, `TEST_IMAGES_REAL_DATA=1`)
- timed region per image: `memcpy` into the input tensor, `interpreter->Invoke()`, argmax over the 10 uint8 outputs
- output lines `BATCH30_STATS,...,avg_us=...,min_us=...,max_us=...,runs=30` and `ACCURACY,...,correct=N,total=30,accuracy=...`
- tensor arena 160 * 1024 bytes, `MicroMutableOpResolver<4>`

Accuracy was compared against the baseline (30/30). A build that lowered accuracy was rejected regardless of latency. Details in [docs/validation.md](docs/validation.md).

## Results

| Stage | Change | Avg latency | Accuracy | Kept |
|---|---|---:|---:|:---:|
| 0 | Baseline TFLM path | 239.60 ms | 30/30 | - |
| 1 | CMSIS-NN path, minimal op set, debug/measurement overhead removed, `-O3` | ~136 ms | 30/30 | yes |
| 2-4 | Fixed-shape kernels, pair-major prepacked filters, zero-activation skip, fast quantize/mean/FC | | | yes |
| 5 | Branch-free / hard-coded requantization variant ("aggressive") | 92.51 ms | 22/30 (73.3%) | **no** |
| 6 | Final valid build (this repository) | **95.17 ms** | **30/30** | yes |

![Optimization history](assets/optimization_history.png)

Reduction 239.60 -> 95.17 ms = 144.43 ms (60.28%), speedup 2.52x. The numbers are taken from the final project report; the raw Serial logs of the runs were not kept, so no log files are included here. The charts in `assets/` are generated from these reported values by `scripts/plot_results.py`.

## Engineering Takeaways

1. **MAC count alone does not explain MCU latency.** Memory access, branch structure, requantization, and generic runtime overhead can dominate even when the arithmetic count is fixed.
2. **Optimization must remain measurable.** Each change was isolated behind its own compile flag and benchmarked with the same protocol to see whether it actually reduced end-to-end latency.
3. **The fastest implementation is not always the valid implementation.** The 92.51 ms build was rejected because accuracy fell to 73.3%. The final build prioritizes latency and functional correctness together.
4. **Fixed workloads enable specialization.** Because the graph and tensor shapes never change, generic runtime paths could be replaced by specialized execution paths with a weight layout matched to the access pattern.

## Repository Structure

```
tflite-micro-latency-optimization/
├── README.md
├── LICENSE
├── .gitignore
├── firmware/
│   └── proj2_final_best/
│       ├── proj2_final_best.ino        final benchmark sketch (BATCH30 protocol)
│       ├── proj_mnist_model_data.h     model as C array (generated from proj_mnist.tflite, unchanged)
│       ├── test_images_30.h            30 fixed MNIST test images + labels
│       └── profile_image_default.h     fixed all-zero profiling input
├── tflm_patches/
│   ├── README.md                       where each file goes in the Arduino TFLM library
│   ├── cmsis_nn/
│   │   ├── conv.cpp                    Conv1..Conv5 fixed-shape fast paths
│   │   ├── fully_connected.cpp         direct int8 FC path
│   │   └── project2_conv{2,3,4,5}_filter_pairmajor_packed.h   prepacked filters
│   └── kernels/
│       ├── quantize.cpp                fixed-size quantize fast paths
│       └── reduce.cpp                  fixed-shape MEAN fast path
├── platform/
│   ├── platform.local.txt              final compile flags
│   └── README.md
├── model/
│   ├── proj_mnist.tflite               course-provided model, unchanged
│   └── README.md
├── docs/
│   ├── architecture.md
│   ├── optimization_history.md
│   └── validation.md
├── scripts/
│   └── plot_results.py                 regenerates assets/*.png from reported numbers
└── assets/
    ├── latency_comparison.png
    └── optimization_history.png
```

## Reproduction

Hardware required: an Arduino Nano 33 BLE. This cleanup did not re-flash the board; the steps below are the procedure used during the project.

1. Arduino IDE with the `Arduino Mbed OS Nano Boards` core (`arduino:mbed_nano`) and the `Arduino_TensorFlowLite` library installed.
2. Copy the files in `tflm_patches/` over the corresponding files in the installed TFLM library (paths listed in [tflm_patches/README.md](tflm_patches/README.md)). Keep backups of the originals.
3. Copy `platform/platform.local.txt` next to the core's `platform.txt` (see [platform/README.md](platform/README.md)). This is what enables the `PROJECT2_*` fast paths and `-O3`.
4. Open `firmware/proj2_final_best/proj2_final_best.ino`, select board `Arduino Nano 33 BLE`, upload.
5. Open Serial Monitor at 115200 baud. Expected output shape:

```
BOOT,project=EmbeddedSystemDesign_Project2
BOOT,mode=PROJECT2_AGGRESSIVE_SAFE
RUN_META,mode=PROJECT2_AGGRESSIVE_SAFE,board=Arduino Nano 33 BLE,warmup=10,tensor_arena=163840,...
SIMD_META,arm_feature_dsp=1,arm_math_dsp=1,cmsis_nn_macro=1
BATCH30_STATS,mode=PROJECT2_AGGRESSIVE_SAFE,avg_us=<avg>,min_us=<min>,max_us=<max>,runs=30
ACCURACY,mode=PROJECT2_AGGRESSIVE_SAFE,correct=30,total=30,accuracy=1.0000
RUN_DONE
```

`python scripts/plot_results.py` regenerates the two charts (needs `matplotlib`).

## Notes / Limitations

- **Original coursework vs. portfolio cleanup.** Everything under `firmware/`, `tflm_patches/`, `platform/`, and `model/` is the original submission, byte-for-byte except for one rename (the sketch's student-ID suffix was dropped to give `proj2_final_best.ino`, with the containing folder renamed to match, as Arduino requires). `README.md`, `docs/`, `scripts/`, `assets/`, `LICENSE`, and `.gitignore` were added afterwards for publication.
- The MNIST model and its C-array header were provided by the course and are included only so the sketch builds. The pair-major packed filter headers are derived from that model's weights (no values changed, only reordered).
- The baseline sketch, the intermediate ~136 ms build, and the rejected 92.51 ms build are not part of the submission archive and are therefore not included. Their behavior is described in `docs/optimization_history.md`.
- Raw Serial benchmark logs were not retained; results are quoted from the final report.
- The patches target the TFLM version bundled with the Arduino `Arduino_TensorFlowLite` library at the time of the project (mid-2026). Kernel file layout in other TFLM versions may differ.
- The fast paths are model-specific by design. They fall back to stock TFLM code for any other tensor shape, but they are not a general-purpose optimization.

## License

MIT for the files written in this project (sketch, patches, docs). The patched kernel files are derived from TensorFlow Lite Micro sources, which remain under the Apache License 2.0 (headers preserved in each file). The MNIST model belongs to the course.
