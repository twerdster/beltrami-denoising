# Native Metal validation

This document defines checks for the native image-denoising engine; it does not imply that every listed size or setting has already passed. Compare identical numerical inputs against the released Python implementation before using photo previews as evidence.

## Numerical contract

- Preserve natural H,W coordinates, grayscale/RGB channel order, pixel intensity scale and alpha-channel policy. Keep decoding, EXIF orientation and optional preview resizing separate from the numerical engine test.
- Match the chosen historical variant, beta, alpha, iteration count and patch radius. A radius of 2 is a 5×5 patch. The original patch sums are unnormalized.
- Backward/forward derivatives replicate the boundary pixel. Patch sums use zero padding, not replication, reflection or valid-only normalization. Test these operators independently.
- Record scalar precision. Metal FP32 is a different numerical implementation from the historical double-precision experiment. Compare with both a CPU FP32 implementation and the Python historical reference.
- Match the experiment's input/output clipping: the 2011 table protocol clips noisy inputs and preserves numerical flow output, while a photo-display preview clips and rounds only for display. Do not compute validation metrics from the preview PNG.

Record max absolute error, RMSE, mean absolute error, finite counts and PSNR difference against the reference, using float64 metric accumulation and peak 255. Declare tolerances in the result record; they are not supplied by the original paper. A visual match is insufficient, and matching an average PSNR can hide a boundary error. Do not silently add an epsilon to an invalid determinant or replace the flow with another denoiser.

## Geometry and content matrix

| Test group | H×W | Purpose |
|---|---|---|
| Tiny/singleton | 1×1, 1×17, 19×1, 2×3, 7×9 | Derivative and patch boundaries |
| Odd and asymmetric | 31×47, 127×129, 129×127 | Row strides, axis and dispatch-edge checks |
| Standard sizes | 128×128, 256×256, 512×512, 1024×1024 | Scaling and memory |
| Rectangular | 321×481, 481×321 | Orientation and non-square buffers |
| Dispatch/tile edges | Sizes just below, equal to and above a workgroup/tile boundary | Guarded threads and partial tiles |

Use constant images, horizontal/vertical ramps, isolated impulses at the center and all four edges/corners, checkerboards, a procedural smooth surface with an intensity step, and deterministic Gaussian noise. Include out-of-range float inputs for the raw engine, and separately test the 2011 clipped-input wrapper. Test grayscale and an RGB image with three deliberately different channels.

Check radius 0, 1, 2, 3 and a radius larger than a tiny image. Check 0, 1, 10, 40 and 150 iterations where supported. Zero iterations must preserve the numerical input under the selected wrapper. A constant image must remain constant under patch flow. Small integer fixtures permit exact indexing/patch-sum checks before floating nonlinear calculations.

Validate intermediate derivatives, squared/cross derivatives, three metric components, determinant, inverse square root, flux/divergence and one complete update. Then compare multiple iterations. This localizes a kernel error and distinguishes it from accumulated FP32 rounding. Shared scratch buffers must not be read before their producing GPU command finishes; consecutive parameter changes and cancellation/restart must not reuse stale output.

## Full images and tiling

Prefer full-frame GPU buffers for the iterative flow. A single update's dependency radius is patch radius plus 1 in each coordinate: the backward derivative's support is [-1,0], patch aggregation expands it by the patch radius, and the forward divergence's [0,1] support completes the symmetric range. K complete updates can therefore depend on a halo as large as K×(radius+1). A fixed small overlap around a tile does not make an entire multi-iteration tile independent of its neighbors.

If tiling is necessary, either retain the complete dependency context or exchange correctly updated halos between every iteration. Preserve the true image boundary rules. Compare tiled and full-frame output in the same precision, with separate error statistics along seams, edges and corners. Blending seam pixels does not prove numerical equivalence.

The native engine's paper-oriented check should use the same downloaded image hashes, historical noise and fixed parameter tables as the Python reproduction. Record whether printed two-decimal paper scores still match; do not transfer that claim automatically from the double-precision Python program to FP32 Metal. Table 1's clean-reference parameter search is an oracle experiment and should not be confused with an automatic photo-denoising setting.

## GPU and timing evidence

A successfully created Metal pipeline is capability evidence. A completed Metal command buffer with measured GPU work is execution evidence. Record command-buffer completion and GPU start/end times where available; values read before completion are not a valid duration. Use Metal GPU captures/Instruments with app signposts to inspect dispatches, synchronization and transfers.

Separate shader/pipeline creation, texture/buffer allocation, input upload, warm kernel work, output readback and preview rendering. Report an end-to-end time with its boundaries alongside GPU time. Retain reusable pipelines/buffers and avoid a CPU wait between kernels unless required. Benchmark Release builds on the physical phone, with actual processed dimensions, iteration count, patch radius, variant, precision, sample count, median/tail latency, memory and thermal state. A resized preview does not establish full-resolution throughput.

Run repeated checks with the same input while changing parameters, rotating the displayed image and loading another image. Require finite predictions and responsive cancellation; report unsupported sizes or resource failures explicitly. Compare large-image speed only with matching work and output precision.

Core ML's `.all` setting, if used by a separate learned-model component, is an allowed-device configuration rather than proof of ANE dispatch. Core ML compute plans are anticipated placement; actual hardware activity requires profiling. Apple's [Core ML profiling demonstration](https://developer.apple.com/videos/play/wwdc2022/10027/) shows the distinction between requests and GPU/Neural Engine activity. This Metal flow should be identified by its measured Metal execution, without an ANE claim.
