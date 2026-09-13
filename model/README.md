# proj_mnist.tflite

Provided by the course as the fixed workload for the project. **It was not modified.** The optimization rules forbade any change to the graph, filter shapes, strides, padding, tensor shapes, weights, biases, or quantization parameters, and all work in this repository respects that.

| Property | Value |
|---|---|
| Size | 106,040 bytes |
| Input | uint8 [1,28,28,1] |
| Output | uint8 [1,10] |
| Operators | QUANTIZE, CONV_2D (x5), MEAN, FULLY_CONNECTED, QUANTIZE |
| Internal dtype | int8, per-channel quantized conv filters |

Graph:

```
uint8 [1,28,28,1]
  -> QUANTIZE -> int8
  -> Conv1 3x3 s2  [1,14,14,8]
  -> Conv2 3x3 s1  [1,14,14,16]
  -> Conv3 3x3 s2  [1,7,7,32]
  -> Conv4 3x3 s1  [1,7,7,64]
  -> Conv5 3x3 s2  [1,4,4,128]
  -> MEAN(1,2)     [1,128]
  -> FULLY_CONNECTED [1,10]
  -> QUANTIZE -> uint8 [1,10]
```

Derived artifacts elsewhere in the repository:

- `firmware/proj2_final_best/proj_mnist_model_data.h` - the same bytes as a C array for the sketch
- `tflm_patches/cmsis_nn/project2_conv{2,3,4,5}_filter_pairmajor_packed.h` - the Conv2..Conv5 filter tensors reordered into pair-major layout for the specialized kernels (values unchanged)

The model is included so the firmware can be built and the result reproduced; it is not claimed as original work.
