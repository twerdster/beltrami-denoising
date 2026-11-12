# Project state

`beltrami-denoising` provides a Python package, `beltrami`, and command-line
tools for the Beltrami and Patch-Beltrami research of Aaron Wetzler and
collaborators. Package version: 1.1.0. The MIT license covers the repository's
code; external photographs, data and models retain their separate terms.

## Classical image experiments

- `src/beltrami/classical.py`: grayscale/RGB explicit flows, integral,
  direct-convolution and separable patch sums, historical precision variants.
- `src/beltrami/nonlocal_means.py`: the numerical NLM comparison procedure.
- `src/beltrami/experiments/ssvm.py`: published Tables 1/3 and color examples.
- Seven exact benchmark PNGs are downloaded from the BM3D authors' website
  and verified against both file and decoded-pixel SHA256 hashes. No benchmark
  photograph is bundled.
- Author parameter values and a compressed, noise-only historical random
  sequence are package resources. The recovered sequence retains float32
  quantization uncertainty; displayed-value agreement is not bit-exact
  reconstruction of unavailable historical random doubles.
- A fresh Linux package run reproduced all 20 Table 1, all 36 Table 3 and all
  three color values at published precision. Maximum differences are
  .004391, .004823 and .003912 dB, respectively. Table 1 uses clean-reference
  beta optimization; Table 3 and color use fixed historical parameters.
- Fifteen focused classical tests pass, including synthetic original-source
  reference arrays, boundary behavior, noise ordering and the CLI.
- An isolated Linux virtual environment successfully built and installed the
  1.0.0 wheel, loaded its packaged numeric resources and ran the installed
  demo. Core, SSVM and six point-cloud tests passed with Python 3.13.5,
  NumPy 2.5.4, SciPy 1.18.1 and Pillow 13.0.0.
- Final isolated-wheel validation passed all 24 tests, including the Deep
  protobuf reader, atomic download-integrity check and pinned manifests.

```bash
pip install -e '.[test]'
pytest tests/test_classical.py tests/test_ssvm.py
beltrami demo
beltrami ssvm --download
```

The demo is procedural and needs no external images. Numerical output uses
the original 0–255 scale and preserves floating-point values; previews alone
are rounded and clipped for display. Other components and release assets are
described in the main README and their component documentation.

## Deep Patch-Beltrami and release assets

- `src/beltrami/deep/` reads the inference-only Caffe checkpoints directly and
  executes their five stage graphs with PyTorch. No Caffe runtime is needed.
- `beltrami-deep reproduce bsd68` fetches hash-pinned public clean images and
  noise-only residuals, reconstructs and verifies all 544 original float32
  inputs, and checks all eight thesis Table 4.6 means.
- A fresh Linux run with the released checkpoint files and newly downloaded
  PNGs reproduced all eight means. The run reserved 666 MiB of GPU memory.
  Per-case results are in `docs/results/deep-bsd68.json`.
- The model export removes non-inference metadata. Independent protobuf
  inspection verified all eight files' shapes and learned float32 payloads
  byte-for-byte against the originals. Five deployment graphs are unchanged.
- The two versioned GitHub release assets have SHA256 hashes:
  - `deep-beltrami-models-v1.tar.gz`:
    `88a65088b7852683657fca0c65dc4cb864ed5328231b36519ddbae4f6da8146b`
  - `bsd68-original-noise-v1.h5`:
    `0c1c0227c71e90f5f08eead3ee18e5e5c9217187b4f5a7240f7376d45f460190`
- Every residual was independently checked to reconstruct its original noisy
  float32 input exactly. No clean/noisy photographs are included in the assets.
- The original 40,000-step models and grayscale offsets are preserved.
  Portrait evaluation requires separately supplied data and reports 100
  images, with the thesis's 92-image caption discrepancy disclosed.

## Point clouds and public distribution

- `src/beltrami/pointcloud.py` provides noisy-neighbor PCA frames, uniform-grid
  local surface fitting and simultaneous 3D updates. `beltrami-points --help`
  lists the parameters; `docs/POINT_CLOUDS.md` states the validation limits.
- The point-cloud code is functional; complete historical 3D figure
  reproduction is not claimed and meshes are not distributed.
- Both point-cloud modes are byte-identical to the prior implementation on
  a two-iteration synthetic fixture. Tests also cover planar invariance,
  local fitting, CLI round trips and undefined-input rejection.
- The README, MIT license, citations and attribution notices describe the
  public package. Synthetic previews and numerical/noise-only fixtures are
  tracked; downloaded images, caches, environments and outputs are excluded.
- GitHub Actions runs the core tests and CLI smoke checks on Python
  3.10, 3.12 and 3.13. Deep inference is verified separately with release
  assets rather than requiring model downloads in each CI job.
- All three GitHub Actions Python jobs passed for the initial public release.

## Native iOS release 1.1.0

- `ios/BeltramiApp.xcodeproj` builds the SwiftUI app for iOS 18+ with Xcode
  16.1+. A procedural sample, full-resolution upright RGB photo import,
  before/after comparison and sharing are provided. Imported photos are not
  automatically written to the diagnostic output folder.
- `ios/BeltramiMetal/` contains the reusable float32 Metal engine and kernels.
  Radius-zero flow skips patch sums. Reused buffers, threadgroup patch sums
  and one command-buffer sequence avoid per-iteration CPU synchronization.
- Arbitrary positive grayscale/RGB dimensions are supported subject to
  available memory. A 192 MiB GPU workspace budget triggers exact spatial
  and temporal tiling; CPU pixel arrays and previews require extra memory.
- Eighteen Python float32 references passed on the physical iPhone 12/A14,
  including forced tiling and singleton/odd dimensions. Worst absolute
  error was 3.05176e-5. Mac forced-tile/full-frame comparisons were bit-identical.
- A 26-case physical-device benchmark includes 2048x1536 at40 iterations and
  4032x3024 at5 iterations without resizing. Warm three-run medians, thermal
  state, work parameters and errors are recorded in `ios/benchmarks/`.
- The UI image adapter checks all eight orientations at two UIImage scales,
  including corner/channel order and preview round trips, on the phone.
- `ios/README.md` documents signing, installation, fixtures, API, timing and
  memory. The root README starts with an actual device demo. The historical
  double-precision paper protocol remains separate from native float32 checks.
- GitHub Actions includes an unsigned Release iOS compilation job. No
  development team, device identifier, generated fixtures or build binaries
  are committed. Source code and procedural demos are MIT licensed.

## README presentation (2026-11-12)

The README adds factual language, platform and existing-license badges while
preserving the original demonstration figures and all runtime, reproduction,
validation and attribution details. Keep badge claims aligned with these docs;
a presentation-only update does not establish a new build or runtime result.
