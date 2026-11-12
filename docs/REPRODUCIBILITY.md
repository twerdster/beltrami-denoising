# Reproducibility scope and limits

This release makes the recovered Beltrami image-denoising implementations usable in Python and records the protocols needed to assess historical numerical agreement. Reproduction at a paper's printed precision, agreement with saved archival arrays, and a fresh end-to-end rerun are different claims. Each experiment should identify which evidence it provides.

## SSVM 2011 image experiments

The historical experiments use clipped Gaussian-noise inputs, double-precision patch flow, and the specific saved parameter tables used by the original table-generating script. A later convenience distribution is not interchangeable with that script.

- Table 1 performs separate beta optimization against the clean reference for each listed iteration/patch-size setting, as specified by its caption. It is an oracle parameter experiment, not a test-label-free denoising benchmark.
- Table 3 uses fixed archived beta/alpha tables, with no new fitting. Patch radius 2 means a 5-by-5 patch; radius 0 is the regular-flow comparison.
- Published color examples use their saved beta and alpha values. Substituting a common alpha changes the experiment.
- The recovered MATLAB legacy seed-0 stream is quantized: it was reconstructed from saved float32 clean/noisy pairs. Matching rounded paper scores does not prove bit-for-bit recovery of the unavailable original double-precision random stream. Octave's seed-0 generator is not a substitute for MATLAB's historical generator.
- Summed-area-table and direct-convolution backends have the same mathematical patch sums but different floating-point accumulation order. Timing on current hardware tests scaling with patch size; it does not reproduce the paper's absolute iPhone 4 runtime.

Images are fetched separately and verified. Numeric parameters and a noise-only stream do not contain the source photographs.

## Deep Patch-Beltrami thesis experiments

The recovered models implement the later thesis experiment, with unclipped Gaussian inputs, float32 arithmetic, five archived stages, and final clipping to [0,255]. Eight checkpoints correspond to noise standard deviations 5, 10, 15, 20, 25, 30, 40 and 50 on the 0-to-255 scale. The recovered files are 40,000-step checkpoints; the thesis describes 100,000 iterations. Releasing the saved files does not imply that training has been repeated.

The original MATLAB/Caffe interface stores the two spatial axes in W,H order, the transpose of an ordinary Python H,W image. The public image interface must transpose into that historical network convention and transpose predictions back. Identical raw tensor axes across different APIs do not necessarily describe the same physical image. Inference must also preserve out-of-range noisy values rather than clipping the input.

The historical wrapper adds fixed sigma-specific intensity offsets before inference and subtracts them afterward. These factors are hardcoded; clean labels are used only for scoring at runtime. The method used to choose the factors was not recovered. Faithful historical scores therefore retain the offsets and disclose their unknown calibration provenance.

When original BSD68 noise residuals are supplied, the loader reconstructs the archived float32 noisy input from the matching clean image and verifies its recorded hash. These residual arrays contain noise values, not clean or noisy images; all source images remain external. A fresh NumPy noise draw is a different experiment, even with a numerically identical seed.

The thesis's portrait caption says 92 images, while its preserved results and reproduced means use all 100 archived portraits. No supported rule for selecting 92 was recovered. Ten portrait IDs also have inconsistent noisy inputs in some later result containers. The original Deep inputs resolved the historical inference comparison, but the private portraits are not redistributed. Public BSD68 and separately supplied portraits must be reported as distinct suites.

The later non-neural Patch-Beltrami baseline uses 50 iterations, patch radius 3 and fixed alpha 0.21. Its beta table was calibrated against clean Lena and then applied globally; it is not an uncalibrated method. Exploratory historical scripts also inspected test images, so absence of historical development exposure is not established. New comparisons must not fit parameters against the clean test labels.

## Modern released-model comparison

The completed comparison with DRUNet, SCUNet, Xformer, revised BERUNet and GRL-B evaluates frozen released models on the same physical archived noisy inputs. It is a comparison of released capabilities, not equal training data or compute, and not an exhaustive ranking of current denoisers.

The primary set is BSD68 at noise standard deviations 15, 25 and 50. Modern official PyTorch interfaces receive natural H,W images; the archived Caffe model receives its historical W,H representation. A fixed transpose adapter preserves the same physical image without augmentation or choosing an orientation by score. Inputs are scaled by 255 where required, without clipping or quantization. Record both archived and canonical natural-image input hashes.

Common output metrics clip predictions to [0,255], retain floating-point values, and compute full-image PSNR using float64 MSE with peak 255. Average per-image dB values with equal image weight. No border crop or integer rounding is used. SSIM uses an 11-by-11 Gaussian window, sigma 1.5, population covariance and data range 255. Raw-output metrics should also be retained. Paired image bootstrap intervals describe variation across these images, not uncertainty over independent noise realizations.

Training exposure and model sizes differ substantially. An exact full-image duplicate check on the recovered historical training set does not rule out crops, near-duplicates, or exposure in external pretrained models. Frozen comparison parameters and API adapters should be set from source documentation rather than selected against clean test images.

The evaluated BERUNet is the author's revised, retrained released checkpoint, not a replication of the original paper's table. GRL-B remains diagnostic: its released checkpoint scores did not recover the reported paper values despite source/configuration checks and numerical parity checks. It must not be presented as a reproduced GRL paper result or used to claim superiority over published GRL. Archived BM3D/NLM outputs are rescored historical outputs, not newly executed inference.

Historical result summaries do not make the full private reproduction workspace part of this public package. A complete modern rerun additionally requires the external source revisions, checkpoints, input/noise fixtures and method-specific inference settings. Timing from shared hardware and different wrappers is descriptive rather than a controlled optimized speed ranking.

## Point-cloud scope

The recovered 3D work was investigated separately. Its local fitting and update components were checked numerically, and preserved output arrays allowed the thesis's 18 numerical entries to be recomputed. That is saved-output metric reconstruction, not a fresh rerun of all historical parameter searches.

Fresh full-mesh tests demonstrated functional denoising with normals estimated from noisy points. The historical figures were not exactly reproduced: necessary random realizations, beta choices and normal provenance were not fully recoverable. Some archived demonstrations use clean ground-truth normals. The historical maximum-distance metric also takes an extra square root; a corrected distance must be distinguished from that historical value. The independently minimized columns in the thesis table need not describe one common output mesh.

This image-denoising release does not claim a complete 3D figure reproduction and does not redistribute third-party meshes.
