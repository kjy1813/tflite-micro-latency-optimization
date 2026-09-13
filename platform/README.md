# platform.local.txt

`platform.local.txt` is the Arduino mechanism for adding compiler flags to a board core without editing its `platform.txt`. It is picked up automatically when it sits in the same folder as the core's `platform.txt`.

For the Nano 33 BLE that folder is:

```
Windows : %LOCALAPPDATA%\Arduino15\packages\arduino\hardware\mbed_nano\<version>\
macOS   : ~/Library/Arduino15/packages/arduino/hardware/mbed_nano/<version>/
Linux   : ~/.arduino15/packages/arduino/hardware/mbed_nano/<version>/
```

Copy `platform.local.txt` there and restart the IDE. The file sets `compiler.cpp.extra_flags` and `compiler.c.extra_flags` to:

```
-DARM_MATH_DSP=1 -DCMSIS_NN=1
-DPROJECT2_DIRECTION_C_CONV45_FIXED=1 -DPROJECT2_DIRECTION_C_DIRECT_PRIMITIVE=0
-DPROJECT2_DISABLE_KERNEL_TIMER=1 -DPROJECT2_ENABLE_KERNEL_PROFILING=0
-O3 -fomit-frame-pointer -fno-unwind-tables -fno-asynchronous-unwind-tables -fno-common
-DPROJECT2_CONV1_DIRECT_SPARSE=1
-DPROJECT2_CONV2_PAIRMAJOR_SPARSE_ALWAYS=1
-DPROJECT2_CONV3_PAIRMAJOR_SPARSE_ALWAYS=1
-DPROJECT2_CONV4_PAIRMAJOR_SPARSE_ALWAYS=1
-DPROJECT2_CONV5_SPARSE_V1=1
```

These flags are what turn the patched kernels' fast paths on. Without the file, the patched library compiles but runs the stock paths (only `PROJECT2_FAST_QUANTIZE` and `PROJECT2_FAST_MEAN` default to on in source).
