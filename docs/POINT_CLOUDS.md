# Point-cloud filtering

The point-cloud port estimates local frames from noisy neighbors with PCA, fits local height grids with regularized least squares, applies the Patch-Beltrami update, and updates all points simultaneously. It requires NumPy and SciPy; no mesh files or reference normals are needed.

Save coordinates as a finite `(N, 3)` NumPy array. Noise standard deviation is in the same coordinate units:

```bash
beltrami-points noisy-points.npy filtered-points.npy \
  --noise-std 0.01 --iterations 10 --neighbors 24
```

The output is a full-precision `.npy` array and an adjacent JSON report with parameters, neighborhood size, progress and finite checks. `--beta`, `--alpha`, `--grid`, `--radius` and `--multiplier` expose the documented numerical parameters. The default mode follows the 2012 formulation; `--mode demo_new` exposes the later local-scale variant.

The local patch update and gridding were independently checked against the historical MATLAB implementation. Full historical figure replication is not claimed: the paper does not specify every noise/parameter choice, some archived examples use reference normals, and the original third-party meshes are not redistributed. This port uses noisy-data PCA normals. Its fresh runs demonstrate the algorithm, while recovered historical metrics and new end-to-end experiments are distinct evidence.

The grid fitter implements the triangular interpolation and gradient-regularization mathematics used by John D'Errico's `gridfit` method; the MATLAB implementation is not bundled. See [provenance and licensing](../THIRD_PARTY_NOTICES.md).

## Citation

Aaron Wetzler, Guy Rosman, and Ron Kimmel. *Patch-space Beltrami denoising of 3D point clouds*. IEEE 27th Convention of Electrical & Electronics Engineers in Israel (IEEEI), 2012. [Author-hosted paper](https://www.cs.technion.ac.il/wp-content/ron-kimmel/papers/Conference/WetRosKimIEEEI2012.pdf).
