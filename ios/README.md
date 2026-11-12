# Beltrami for iPhone

The SwiftUI app runs the classical grayscale and RGB Beltrami flow in native
float32 Metal kernels. It accepts arbitrary positive image dimensions,
including odd and non-square images. Photo processing preserves the decoded
pixel dimensions; it does not downsample to a model input size.

## Build and run

Use Xcode 16.1 or newer with the iOS 18 SDK and an iOS 18 device. Open
[`BeltramiApp.xcodeproj`](BeltramiApp.xcodeproj), select the `BeltramiApp`
scheme, choose your development team in Signing & Capabilities, and select
your connected, paired iPhone. Enable Developer Mode on the phone when
requested. Set a unique bundle identifier if your signing account requires
one. Select the Release configuration for timing measurements, then run.

The app can generate a procedural noisy image or load a photo. Choose the
iteration count, beta, and local/5×5/7×7 patch setting before processing.
The local setting is radius 0, 5×5 is radius 2, and 7×7 is radius 3. The UI
uses alpha 0.21. The native API exposes alpha as well as beta, radius and
iteration count. The synthetic noise level changes the demo input; it does
not perform an automatic search for beta.

The command-line build needs no Python package or downloaded assets:

```bash
ios/scripts/build_ios.sh
```

This compiles an unsigned Release arm64 app. For a physical-device build,
set `TEAM_ID` to your Apple development team identifier. `BUNDLE_ID` is an
optional signing override. Xcode must already be signed in to your account.

```bash
TEAM_ID="$MY_APPLE_TEAM_ID" ios/scripts/build_ios.sh
```

Build products go to `.build-ios/Build/Products/Release-iphoneos/BeltramiApp.app`.
Set `BELTRAMI_IOS_BUILD_DIR` to use another build directory. Signing
identifiers and generated build products are not part of the source tree.

## Numerical behavior and memory

[`BeltramiMetal.swift`](BeltramiMetal/BeltramiMetal.swift) accepts row-major
interleaved grayscale or RGB arrays on the original 0–255 intensity scale:

```swift
let engine = try BeltramiMetal()
let result = try engine.denoise(pixels, width: width, height: height,
    channels: 3, beta: 800, alpha: 0.21, iterations: 40, radius: 2)
```

The API preserves floating-point input and output values without clipping.
Photo decoding normalizes orientation and flattens transparency on white;
preview rendering rounds and clips to 8 bits. Raw `.f32` outputs, rather
than preview PNGs, are used for numerical comparisons.

The implementation uses replicated derivative boundaries and zero-padded,
unnormalized patch sums. Shared threadgroup memory accelerates the two box
passes. Buffers and pipelines are reused, and the complete full-frame
iteration sequence is submitted without a CPU wait between iterations.
Radius 0 skips patch aggregation. Metal fast math is disabled in the
project to preserve the specified float32 arithmetic.

The default GPU workspace budget is 192 MiB. Full-frame workspaces use
96 bytes per pixel, or 80 bytes per pixel for radius 0. Larger images use
temporal blocks of at most 16 iterations and exact dependency halos of
`blockIterations × (radius + 1)` around each tile. The updated tile cores
are assembled between temporal blocks. This preserves the flow without
resizing or blending seams. The budget covers GPU workspace; CPU image
arrays, photo decoding and previews use additional memory. Reported
workspace bytes include retained reusable buffers, so a later small image
may report the workspace allocated for an earlier larger image.

This is a custom Metal GPU implementation. The historical paper tables use
a separate double-precision protocol in the [Python reproduction](../README.md).
Their printed scores are not automatically a claim about this float32 app.

## Reference fixtures and device validation

The 18 procedural reference cases cover grayscale/RGB, singleton and
one-dimensional images, small and odd rectangles, constant images,
radius 0/2/3, up to 150 iterations, and forced tiling. They contain no
photographs and require no network access. To generate them and build:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
ios/scripts/build_ios.sh --fixtures
```

Add `TEAM_ID` to the build command for device installation. The fixture
generator uses the active interpreter; `BELTRAMI_PYTHON` can select another
one. To prepare fixtures without building, run:

```bash
python ios/scripts/prepare_fixtures.py
```

The generator computes the Python `efficient` variant with `precision="single"`
and `backend="conv2"`. It writes ignored files beneath
`BeltramiApp/Fixtures/<id>/{input,expected}.f32` and a manifest containing
dimensions, parameters and SHA256 hashes. Existing flat fixtures from
`Tests/generate_fixtures.py` can instead be converted using only Python's
standard library:

```bash
python3 ios/scripts/prepare_fixtures.py --flat-dir /path/to/flat-fixtures
```

Install the signed app and run the fixture comparisons using `devicectl`.
Set `DEVICE_ID` to the identifier of your paired device from
`xcrun devicectl list devices`. Use your overridden bundle identifier if
you changed it during signing.

```bash
xcrun devicectl device install app --device "$DEVICE_ID" \
  .build-ios/Build/Products/Release-iphoneos/BeltramiApp.app
xcrun devicectl device process launch --device "$DEVICE_ID" \
  --terminate-existing --console org.twerdster.BeltramiDenoising --validate
mkdir -p results/ios
xcrun devicectl device copy from --device "$DEVICE_ID" \
  --domain-type appDataContainer \
  --domain-identifier org.twerdster.BeltramiDenoising \
  --source Documents/benchmark-fp32-gpu.json \
  --destination results/ios/validation.json
```

`--validate` runs the bundled cases and records full-array maximum absolute
and RMS errors, finite status, parameters, memory and timings. It uses one
warm repeat by default. A complete fixture run contains 18 cases; check
that count when reviewing the report. `--benchmark` runs those cases plus eight larger
procedural performance cases, using three warm repeats after a cold run.
Both modes write `Documents/benchmark-fp32-gpu.json` and individual raw
float32 outputs. Each invocation replaces that report. Copy it before
starting another mode if you want to retain both results.

Use `--case tile_gray,tile_rgb` to select cases and `--repeats 10` to change
the warm repeat count. `--stay-open` keeps the app open after completion.
The JSON's `median_ms` is the median warm engine wall time, including data
packing, upload, GPU work and readback. It excludes photo decoding and UI
rendering. `gpu_ms` contains all warm GPU durations; nested `statistics`
describes the final warm invocation. Numerical errors should be assessed
alongside the recorded finite status; see the [validation protocol](VALIDATION.md).

The macOS harness also checks all reference arrays and compares forced
tiling with full-frame Metal output:

```bash
ios/Tests/run_mac_validation.sh
```

It requires the Python package and a Metal-capable Mac. It builds native
shaders and a Swift command-line runner, enforces maximum absolute error
0.01 and RMS error 0.001 on the 0–255 scale, and writes `report.json` under
the printed temporary build directory. Set `BELTRAMI_METAL_BUILD_DIR` to
retain artifacts at a chosen location.

## Physical iPhone measurements

On an iPhone 12 (A14), iOS 18.0.1, Release build, all 18 reference cases
produced finite outputs. Maximum absolute error against the Python
float32 reference was `3.05176e-5` intensity units. The forced grayscale
and RGB tiling cases used 45 and 63 tile executions respectively, with
64,896 bytes of GPU workspace; both had maximum error `1.52588e-5`.
Forced tiling and full-frame Metal were also bit-identical in the macOS
harness.

Representative grayscale engine timings below are medians of three warm
runs, with beta 800, alpha 0.21 and radius 2. The device reported nominal
thermal state at the start and end, with Low Power Mode off. Three repeats
do not establish sustained throughput or tail latency.

| Image dimensions | Iterations | Warm median wall time | Tiled |
|---|---:|---:|---|
| 128×128 | 40 | 7.01 ms | No |
| 256×256 | 40 | 16.44 ms | No |
| 512×512 | 40 | 65.27 ms | No |
| 1024×768 | 40 | 246.27 ms | No |
| 2048×1536 | 40 | 1,132.40 ms | Yes |
| 4032×3024 | 5 | 637.68 ms | Yes |

The two large cases stayed below the 192 MiB GPU workspace budget
(201,283,584 bytes). Timings are specific to these inputs and parameters;
the 12-megapixel case uses fewer iterations. These measurements establish
Metal GPU execution, with no Neural Engine claim.

## Image import checks and demo capture

`--check-image-io` runs16 assertions on the phone: all eight EXIF-style
orientations at UIImage scales1 and2, with corner/channel ordering and
preview round trips. It exits nonzero on failure and writes
`Documents/image-io-checks.json` on success. All16 passed on the test phone;
[the sanitized result](benchmarks/image-io.json) is included.

`--auto-run --screenshot` processes the procedural sample and writes its
raw floats, display PNGs, metrics and an actual app-view capture to Documents.
The [README demo metadata](../docs/images/ios-demo.json) records that run.
Imported photos remain in memory for processing; sharing is an explicit action.

Device validation exits nonzero on a missing/empty suite, a non-finite
prediction or reference errors above max0.01/RMS0.001 intensity units.
[The26-case benchmark](benchmarks/iphone12.json) includes all18 references
plus eight performance sizes. Its three-repeat latency values describe
that measurement, not sustained throughput or tail latency.
