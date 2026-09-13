# TFLM kernel patches

These files replace kernels inside the Arduino `Arduino_TensorFlowLite` library. Copy each one over the file at the corresponding path (keep a backup of the originals). The library root is the `src/` folder of the installed library.

| File in this folder | Destination inside the TFLM library |
|---|---|
| `cmsis_nn/conv.cpp` | `tensorflow/lite/micro/kernels/cmsis_nn/conv.cpp` |
| `cmsis_nn/fully_connected.cpp` | `tensorflow/lite/micro/kernels/cmsis_nn/fully_connected.cpp` |
| `cmsis_nn/project2_conv2_filter_pairmajor_packed.h` | `tensorflow/lite/micro/kernels/cmsis_nn/` (new file) |
| `cmsis_nn/project2_conv3_filter_pairmajor_packed.h` | `tensorflow/lite/micro/kernels/cmsis_nn/` (new file) |
| `cmsis_nn/project2_conv4_filter_pairmajor_packed.h` | `tensorflow/lite/micro/kernels/cmsis_nn/` (new file) |
| `cmsis_nn/project2_conv5_filter_pairmajor_packed.h` | `tensorflow/lite/micro/kernels/cmsis_nn/` (new file) |
| `kernels/quantize.cpp` | `tensorflow/lite/micro/kernels/quantize.cpp` |
| `kernels/reduce.cpp` | `tensorflow/lite/micro/kernels/reduce.cpp` |

The `conv.cpp` includes the packed headers via `tensorflow/lite/micro/kernels/cmsis_nn/project2_conv*_filter_pairmajor_packed.h`, so they must sit next to `conv.cpp`.

## Compile-time flags

The fast paths are selected with `PROJECT2_*` macros. The final configuration is defined in `../platform/platform.local.txt`:

| Macro | Default in source | Final build | Meaning |
|---|---|---|---|
| `PROJECT2_CONV1_DIRECT_SPARSE` | off | 1 | Conv1 direct fixed-shape kernel with zero-input skip |
| `PROJECT2_CONV2_PAIRMAJOR_SPARSE_ALWAYS` | off | 1 | Conv2 pair-major sparse kernel, unconditional |
| `PROJECT2_CONV3_PAIRMAJOR_SPARSE_ALWAYS` | off | 1 | Conv3 pair-major sparse kernel, unconditional |
| `PROJECT2_CONV4_PAIRMAJOR_SPARSE_ALWAYS` | off | 1 | Conv4 pair-major sparse kernel, unconditional |
| `PROJECT2_CONV5_SPARSE_V1` | off | 1 | Conv5 pair-major sparse kernel (V4), unconditional |
| `PROJECT2_CONV{2,3,4}_PAIRMAJOR_SPARSE_HYBRID_V1`, `PROJECT2_CONV5_SPARSE_HYBRID_V2` | off | off | Runtime zero-count with threshold, dense CMSIS-NN fallback below it (experimental, not used in the final build) |
| `PROJECT2_CONV123_FIXED_CMSIS` | off | off | Route Conv1-3 to the fixed-dimension CMSIS-NN call instead of the sparse kernels |
| `PROJECT2_DIRECTION_C_CONV45_FIXED` | off | 1 | Fixed-dimension CMSIS-NN call for Conv4/Conv5 (reached only if a sparse-path shape guard fails) |
| `PROJECT2_DIRECTION_C_DIRECT_PRIMITIVE` | off | 0 | Call the CMSIS-NN conv primitive directly instead of `arm_convolve_wrapper_s8` |
| `PROJECT2_FAST_QUANTIZE` | 1 | 1 | Fixed-size Quantize fast paths (784 in, 10 out) |
| `PROJECT2_FAST_MEAN` | 1 | 1 | Fixed-shape `[1,4,4,128] -> [1,128]` Mean fast path |
| `PROJECT2_ENABLE_KERNEL_PROFILING` / `PROJECT2_DISABLE_KERNEL_TIMER` | - | 0 / 1 | Per-kernel profiling instrumentation off in the measurement build |

Every fast path checks tensor shape and dtype before running and falls back to the stock TFLM / CMSIS-NN implementation otherwise, so the patched library still works for other models (without any speedup).

## Provenance

`conv.cpp`, `fully_connected.cpp`, `quantize.cpp`, and `reduce.cpp` are modified copies of TensorFlow Lite Micro sources (Apache License 2.0, original headers kept). The project additions are delimited by `PROJECT2_*_BEGIN` / `_END` comment markers inside `conv.cpp`. The four `project2_conv*_filter_pairmajor_packed.h` files were generated from the unchanged `proj_mnist.tflite` filter tensors; they contain the same weight values in a different order.
