# Validation and Benchmark Methodology

## Measurement setup

| Item | Value |
|---|---|
| Board | Arduino Nano 33 BLE (nRF52840, Cortex-M4F) |
| Core | `arduino:mbed_nano:nano33ble` |
| Runtime | TensorFlow Lite Micro (Arduino library) with the patches in `tflm_patches/` |
| Tensor arena | 160 * 1024 = 163,840 bytes, 32-byte aligned |
| Op resolver | `MicroMutableOpResolver<4>` (Quantize, Conv2D, Mean, FullyConnected) |
| Input / output dtype | uint8 / uint8 |
| Test set | 30 fixed MNIST images with labels (`firmware/proj2_final_best/test_images_30.h`) |
| Warm-up | 10 inferences on image 0, not timed |
| Timed runs | 30 (one per test image) |
| Timer | `micros()` |

## Timed region

Per image, in `RunBatch30Release()`:

```
t0 = micros()
memcpy(input_tensor, image, 784)      // input copy
interpreter->Invoke()                 // full model
pred = argmax(output_tensor[0..9])    // uint8 argmax
t1 = micros()
```

So the reported latency is end-to-end for one classification, not `Invoke()` alone. Nothing else runs inside the region: no Serial output, no per-layer timers, no prediction printing (`PROJECT2_PRINT_PRED` defaults to 0; the kernel profiling counters are compiled out with `PROJECT2_ENABLE_KERNEL_PROFILING=0` and `PROJECT2_DISABLE_KERNEL_TIMER=1`).

## Reported statistics

After the 30 runs the sketch prints:

```
BATCH30_STATS,mode=PROJECT2_AGGRESSIVE_SAFE,avg_us=<avg>,min_us=<min>,max_us=<max>,runs=30
ACCURACY,mode=PROJECT2_AGGRESSIVE_SAFE,correct=<n>,total=30,accuracy=<n/30>
```

`avg_us` is the arithmetic mean of the 30 per-image times (integer math, two fixed decimals). Accuracy is the number of correct argmax predictions out of 30.

## Acceptance rule

A candidate build was accepted only if:

1. `ACCURACY` reported 30/30, identical to the baseline build, and
2. `BATCH30_STATS avg_us` was lower than the previous accepted build.

Rule 1 is what rejected the 92.51 ms build (22/30). Rule 2 is what stopped Stage 1 style changes once they stopped paying off.

## Final result (from the project report)

| | Baseline | Final |
|---|---:|---:|
| avg latency | 239.60 ms | 95.17 ms |
| accuracy | 30/30 | 30/30 |

Raw Serial captures were not retained with the submission, so this repository does not contain log files. Re-running the sketch on the board reproduces the two lines above.

## What the accuracy check does and does not cover

- It covers the full deployment path: input copy, quantize, all five conv layers, mean, FC, output quantize, argmax.
- It is a 30-image fixed set, chosen to make latency comparisons stable, not a full MNIST evaluation. An optimization that changed outputs on images outside this set would not be caught. Because the kept optimizations are exact reformulations (same accumulation, same requantization multipliers, skipping only terms that are mathematically zero), no output difference is expected; the rejected build is the case where that assumption broke and the check caught it.
