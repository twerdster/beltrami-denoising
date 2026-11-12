# Attribution and external materials

The [MIT license](LICENSE) applies to the Beltrami code and documentation in this repository. It does not relicense images, datasets, third-party software, papers, or externally obtained model weights.

## Beltrami research and released artifacts

The implementations follow the Beltrami and Patch-Beltrami work of Aaron Wetzler and collaborators. The historical image experiments are described in *Efficient Beltrami Flow in Patch-Space* (SSVM 2011) and the later Deep Patch-Beltrami experiments in Aaron Wetzler's 2017 thesis, *Topics in Computational Geometric Vision*. See [the reproducibility notes](docs/REPRODUCIBILITY.md) for the distinction between these experiments.

The companion Deep Patch-Beltrami artifacts are the author-provided Caffe-format trained checkpoints and stage descriptions. They are separate from the Caffe implementation: this repository does not redistribute Caffe C++ source or binaries. The Python reader implements only the file-format subset needed by these checkpoints.

The numeric historical parameter tables and noise-only arrays are research artifacts. A noise array is not a clean image or an archived noisy photograph. Benchmark images must be obtained separately. The respective download manifests record sources and checksums.

## Non-local means comparator

Non-local means was introduced by Antoni Buades, Bartomeu Coll and Jean-Michel Morel in *A Non-Local Algorithm for Image Denoising* (CVPR 2005). The historical SSVM comparison used a MATLAB implementation credited to José Vicente Manjón Herrera and Antoni Buades, dated March 9, 2006.

The public package supplies a vectorized NumPy/SciPy implementation of the relevant mathematical procedure, including the historical weighted patch kernel, symmetric patch boundaries, valid candidate centers and maximum-neighbor self-weight. The historical MATLAB source and its comments are not included. Numerical agreement with that comparator is a validation result, not a claim that its original source has been relicensed.

## Point-cloud local surface fitting

The point-cloud port implements the triangular interpolation and gradient-regularized least-squares mathematics used by John D'Errico's [Surface Fitting using gridfit](https://www.mathworks.com/matlabcentral/fileexchange/8998-surface-fitting-using-gridfit). The compact NumPy/SciPy implementation covers the fixed uniform-grid settings required by the Beltrami experiment; it is not the complete MATLAB package. The original `gridfit.m`, its documentation and other third-party point-cloud utilities are not bundled or relicensed.

## Images and datasets

The standard test photographs used by the experiments are not bundled and are not covered by this repository's MIT license. Download tools obtain the exact required versions from the recorded public sources and verify their hashes. A public download location identifies provenance; it does not establish that an image is public domain or grant new redistribution rights.

The BSD68 image set derives from the Berkeley Segmentation Dataset. See the [Berkeley dataset page](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) and the upstream image source recorded by the fetcher. Use the dataset's own terms and attribution. The dataset page requests citation of David Martin, Charless Fowlkes, Doron Tal and Jitendra Malik, *A Database of Human Segmented Natural Images and its Application to Evaluating Segmentation Algorithms and Measuring Ecological Statistics*, ICCV 2001. Portrait images from the private historical archive are not redistributed; portrait experiments require separately supplied data.

No original paper PDFs, private correspondence, personal backup manifests, or third-party point-cloud meshes are part of the release.

## Optional comparison models

The modern comparison concerns frozen releases of separate research projects. Their source and pretrained weights are not included in the MIT core. The upstream projects retain their own terms:

| Project | Upstream source license | Source |
|---|---|---|
| DRUNet / DPIR | MIT | [DPIR](https://github.com/cszn/DPIR), [license](https://github.com/cszn/DPIR/blob/master/LICENSE) |
| SCUNet | Apache-2.0 | [SCUNet](https://github.com/cszn/SCUNet), [license](https://github.com/cszn/SCUNet/blob/main/LICENSE) |
| Xformer | No license file found in the audited source revision | [Xformer](https://github.com/gladzhang/Xformer/tree/ecc1e910b923f35c8d46f5ffbc8ed2bc603f6567) |
| BERUNet | MIT | [BERUNet](https://github.com/Xin-Ge/BERUNet-denoiser), [license](https://github.com/Xin-Ge/BERUNet-denoiser/blob/main/LICENSE) |
| GRL | MIT | [GRL](https://github.com/ofsoundof/GRL-Image-Restoration), [license](https://github.com/ofsoundof/GRL-Image-Restoration/blob/main/LICENSE) |

Source-license identification does not separately establish a checkpoint or its training data's terms. Obtain external model files from the originating project and retain its notices. Xformer code is not vendored or represented as MIT-licensed by this release.

## Runtime dependencies

NumPy, SciPy, Pillow, h5py and optional PyTorch are installed as dependencies rather than copied into the source tree. Their original licenses and notices remain in the installed distributions. Package metadata specifies which dependencies each component needs.
