# Beltrami denoising updates

## Updated and ported for modern hardware

The Python package implements the recovered Beltrami and Patch-Beltrami
operators and reproducible experiments without MATLAB; optional Deep inference
executes the original checkpoints through PyTorch. The new SwiftUI app uses
float32 Metal kernels, full-resolution photo import and context-preserving
tiling, with recorded iPhone 12 checks. These are new implementations of the
operators and app integration; the historical double-precision paper protocol
and native float32 validation remain separate.

### Code and setup

- [Python image operators](../src/beltrami/classical.py),
  [point-cloud updates](../src/beltrami/pointcloud.py) and
  [Deep inference](../src/beltrami/deep/).
- [Python installation and demo](../README.md#quick-start) and
  [reproduction conventions](../docs/REPRODUCIBILITY.md).
- [iOS installation and engine](../ios/README.md), [Metal source](../ios/BeltramiMetal/)
  and [Deep reproduction guide](../docs/deep-reproduction.md).

### Validation and limits

[Python tests](../tests/) accompany the saved
[paper results](../docs/results/ssvm-summary.json). Table 1 selects beta against
the clean reference; agreement at printed precision does not establish a
bit-identical historical random sequence. The [point-cloud guide](../docs/POINT_CLOUDS.md)
states its narrower reproduction coverage.

[iOS numerical validation](../ios/VALIDATION.md) and the
[device record](../ios/benchmarks/iphone12.json) document float32 references,
image geometry, measured work and timing boundaries. The Metal results do not
transfer the separate double-precision paper scores to the phone.

[Project reference, citations and licensing](../README.md)
