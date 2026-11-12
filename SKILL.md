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
