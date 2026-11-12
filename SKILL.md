# Development and reproduction

Read `SUMMARY.md`, `README.md`, `THIRD_PARTY_NOTICES.md` and
`docs/REPRODUCIBILITY.md` before changing experiment behavior. Preserve the
separation between code licensing, external images and model artifacts.

## Classical workflow

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest tests/test_classical.py tests/test_ssvm.py
beltrami demo --output results/demo
beltrami ssvm --download --output results/ssvm
```

The SSVM command downloads only the seven required public PNGs, validates
checksums, recomputes metrics and fails if a requested result differs at
published precision. `--tables 3 color` omits the Table 1 oracle study.
`--save-arrays` retains full numerical outputs. Downloads and generated
results belong outside tracked source and must not be relabeled as MIT data.

Keep historical conventions explicit: clipped noisy input, double precision,
unnormalized patch sums, replicated derivative boundaries, zero-padded patch
sums and unclipped numerical outputs. A radius of 2 means a 5-by-5 patch.
Later single-precision flows are a separate variant. Do not silently change
the random sequence, image versions, parameter tables, clipping or metrics.

Run focused tests after relevant changes. Recompute affected experiments
when numerical behavior changes; unchanged saved metrics do not validate a
new numerical implementation. Update `SUMMARY.md`, this file and public
documentation before committing behavior or workflow changes.

## Deep and point-cloud workflow

```bash
pip install -e '.[deep,test]'
pytest
beltrami-deep reproduce bsd68 --output results/deep-bsd68.json
beltrami-points noisy.npy filtered.npy --noise-std 0.01
```

Deep documentation: `docs/deep-reproduction.md`. Preserve the historical
W,H network interface, unclipped float32 inputs, sigma-specific intensity
offsets, final clipping and full-image float PSNR. Verify learned tensor bytes
when rebuilding model assets; metadata stripping must not alter tensors.
Release asset names, byte lengths, URLs and SHA256 values are pinned in
`src/beltrami/deep/data/release-assets.json`. Keep release files separate from
the source tree. `tools/build_deep_assets.py` takes explicit source paths and
validates noise reconstruction and tensor fidelity.

Use `--device cpu` without CUDA, `--limit 2 --sigmas 25` for a partial smoke
test, and `--cache-dir` or `--asset-dir` for controlled/offline storage.
Only a complete eight-sigma, 68-image run is the full BSD68 reproduction.
Portraits require user-supplied data. Do not describe a partial suite as a
complete published result.

The 3D CLI consumes an N-by-3 `.npy` point array and emits an array plus JSON.
Keep noisy-data PCA normals and historical figure provenance explicit.
Its later `demo_new` mode requires positive noise and distinct local
neighbors. It is not an exact historical mesh-figure reproduction command.

Before public releases, inspect the tracked tree and assets for unintended
private data, absolute machine paths and external materials. Keep attribution,
license boundaries, command examples and result claims synchronized. No
training is needed to reproduce the released inference results.

For release verification, build and install a wheel into an isolated virtual
environment, unset `PYTHONPATH`, and run all tests from outside the source
checkout. Confirm package resources load from the installed distribution.
Check the three Python CI jobs, public repository visibility and an anonymous
clone. Validate public asset downloads with a fresh cache; a local asset-dir
test alone does not verify the published download path.

## Native iOS workflow

Read `ios/README.md` and `ios/VALIDATION.md` before changing native behavior.
Build with `ios/scripts/build_ios.sh` (unsigned), or set `TEAM_ID` to sign
using an existing Apple development account. Do not commit local signing
settings, device identifiers, generated fixtures/models, caches or builds.
Use Release builds on a physical phone for latency evidence; keep sample
counts, thermal state, precision and processed dimensions with the results.

Run `--check-image-io` on the phone after image-adapter changes. It must
pass all16 orientation/scale checks. `--validate` must contain the expected
reference cases and pass declared finite/error gates; an empty suite fails.
Use `--benchmark`, `--case` and `--repeats` for controlled timings. Preserve
reports before another invocation replaces the same Documents filename.
`--auto-run --screenshot` exports only the procedural demo and an app-view
capture for documentation. Imported photos are shared only on user action.

Generate18 references with `python ios/scripts/prepare_fixtures.py` or
`ios/scripts/build_ios.sh --fixtures`. Run `ios/Tests/run_mac_validation.sh`
for full-frame/forced-tile numerical controls. Preserve unnormalized sums,
replicated derivatives, zero patch boundaries and exact temporal halos.
The historical double-precision Python scores are separate from float32
Metal fidelity; do not transfer their table claim without rerunning it.

## Documentation maintenance

Read SUMMARY.md, then this file, README, linked setup/validation records and
relevant source before revising claims. Write concrete inputs, operations and
outputs; keep current implementation, historical aims and recorded evidence
separate. Label synthetic/untrained figures and schematics. Keep metric units,
implementation, hardware, sample count and measurement scope with the claim.
Source inspection or file export does not establish numerical execution or
training. Preserve original attribution, licenses and validation limitations.

Rebuild distributions when README metadata or bundled guides change; compare
installed documentation with source. For substantial prose revisions, request
an actual Claude review if directed, verify its factual suggestions and review
the revision again. Never claim a review or runtime check that did not complete.
Review and synchronize both context files before every commit.
