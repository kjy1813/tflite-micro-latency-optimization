# Optimization History

All latencies are the 30-image batch average from the same measurement protocol (see [validation.md](validation.md)). Values are quoted from the final project report; raw Serial logs were not retained.

## Stage 0 - Baseline: 239.60 ms

Stock TFLM inference path with the provided model. Accuracy 30/30.

## Stage 1 - Conventional cleanup: ~136 ms

- Made sure the CMSIS-NN kernel path was actually taken
- Registered only the four ops the model uses (`MicroMutableOpResolver<4>`)
- Removed debug prints, per-prediction Serial output, and per-layer profiling counters from the timed region
- Compiler optimization (`-O3` and related flags in `platform.local.txt`)

After this, further changes of the same kind produced only marginal gains.

## Stage 2 - Reframing the problem

Old question: *how do we do the same computation faster?*
New question: *do we actually have to do all of this computation?*

`Invoke()` was profiled at layer and kernel level (this instrumentation is compiled out of the final build via `PROJECT2_ENABLE_KERNEL_PROFILING=0`). Items examined: generic kernel branches, memory access pattern, requantization cost, per-layer hot paths, activation value distribution.

## Stage 3 - Zero activations

About 54% of the Conv4 / Conv5 input activations were at the quantization zero point (int8 -128, real value 0). Every MAC that consumes such an activation contributes exactly zero. Those MACs can be skipped with no change in the output.

## Stage 4 - Fixed-shape specialization

Since the graph and all tensor shapes are fixed, the generic kernels' flexibility could be traded for shape-specific fast paths:

1. Conv1 direct fixed-shape path on the original filter tensor (`PROJECT2_CONV1_DIRECT_SPARSE`)
2. Conv2..Conv5 pair-major prepacked filter layout, `SMLAD` dual MAC, contiguous output-channel accumulation (`PROJECT2_CONV{2,3,4}_PAIRMAJOR_SPARSE_ALWAYS`, `PROJECT2_CONV5_SPARSE_V1`)
3. Zero-contribution pair skip inside the conv kernels
4. Fixed fast paths for Quantize, Mean, and Fully Connected
5. `-O3`
6. ARM DSP / CMSIS-NN kept for everything not covered by a fast path
7. Loop-unrolling handled per kernel where it helped, rather than as a global compiler flag (the sketch keeps `#pragma GCC optimize("unroll-loops")` at sketch scope)

Each item has its own `PROJECT2_*` compile flag so it could be enabled and measured on its own.

## Stage 5 - Rejected: 92.51 ms at 22/30

An "aggressive" variant went further: branch-free inner loops and hard-coded requantization parameters. It measured 92.51 ms, the lowest of any build, but accuracy fell to 22/30 (73.3%).

Decision: the fastest build was **not** selected. The changes were split apart and re-verified individually; those that caused the accuracy regression were dropped.

The rejected build's source is not part of the submission archive and is not included here.

## Stage 6 - Final valid build: 95.17 ms at 30/30

| | |
|---|---|
| Average latency | 95.17 ms |
| Accuracy | 30/30 (100%) |
| Absolute reduction vs baseline | 144.43 ms |
| Relative reduction | 60.28% |
| Speedup | 2.52x |
| Model file | unchanged |

This is the build in `firmware/`, `tflm_patches/`, and `platform/`. Its mode string in the Serial output is `PROJECT2_AGGRESSIVE_SAFE`: the "aggressive" kernel set minus the changes that broke correctness.
