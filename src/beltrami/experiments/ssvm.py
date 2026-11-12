"""SSVM 2011 Tables 1/3 and the three published color examples.

The bundled random sequence and parameter values reproduce the historical
protocol. Benchmark images are downloaded separately from the BM3D authors'
public site and verified with SHA256 before use.
"""

from __future__ import annotations

import hashlib
from importlib.resources import files
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar

from ..classical import patch_beltrami, psnr
from ..nonlocal_means import nonlocal_means


GRAYSCALE_NAMES = ("Cameraman256", "Lena512", "barbara", "house")


def _resource(name):
    return files("beltrami").joinpath("data", name)


def parameters():
    return json.loads(_resource("ssvm_parameters.json").read_text())


def image_sources():
    return json.loads(_resource("image_sources.json").read_text())


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_image(path, source):
    if _sha256(path) != source["file_sha256"]:
        raise ValueError(f"SHA256 mismatch for {path}; expected {source['file_sha256']}")
    with Image.open(path) as image:
        array = np.asarray(image)
    if list(array.shape) != source["shape"]:
        raise ValueError(f"Unexpected image shape for {path}: {array.shape}")
    if hashlib.sha256(array.tobytes()).hexdigest() != source["pixels_sha256"]:
        raise ValueError(f"Decoded pixel hash mismatch for {path}")


def fetch_images(directory, names=None):
    """Fetch verified public benchmark PNGs, retaining valid existing files.

    Downloaded images retain their original rights; they are not covered by
    the Python package's code license. No image is included in the package.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    selected = set(names) if names is not None else None
    downloaded = []
    for source in image_sources():
        if selected is not None and source["name"] not in selected:
            continue
        target = directory / (source["name"] + ".png")
        if not target.exists():
            temporary = target.with_suffix(".png.part")
            try:
                request = Request(source["url"], headers={"User-Agent": "beltrami-denoising/1.0"})
                with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
                _validate_image(temporary, source)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        _validate_image(target, source)
        downloaded.append(str(target))
    return downloaded


def load_image(directory, name):
    source = next(s for s in image_sources() if s["name"] == name)
    path = Path(directory) / (name + ".png")
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run 'beltrami fetch-ssvm --data-dir {directory}' first")
    _validate_image(path, source)
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.float64)


def legacy_noise():
    metadata = parameters()["noise"]
    resource = _resource(metadata["file"])
    data = resource.read_bytes()
    if hashlib.sha256(data).hexdigest() != metadata["sha256"]:
        raise ValueError("Bundled legacy-noise file SHA256 mismatch")
    with resource.open("rb") as stream, np.load(stream, allow_pickle=False) as archive:
        noise = archive["noise"]
    if (noise.dtype != np.float64 or noise.size != metadata["count"] or
            hashlib.sha256(noise.tobytes()).hexdigest() != metadata["array_sha256"]):
        raise ValueError("Bundled legacy-noise array does not match its manifest")
    return noise


def corrupt(clean, sigma, noise=None):
    """Reset the historical stream for each image; clip noisy input to 0..255."""
    noise = legacy_noise() if noise is None else np.asarray(noise)
    if clean.size > noise.size:
        raise ValueError("Image exceeds the bundled historical noise sequence")
    return np.clip(clean + sigma * noise[:clean.size].reshape(clean.shape, order="F"), 0, 255)


def _write_json(path, value):
    temporary = Path(path).with_suffix(".json.part")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _record(method, name, sigma, score, target, **extra):
    return {"method": method, "image": name, "sigma": sigma,
            "paper_psnr_db": target, "psnr_db": score,
            "difference_db": score-target, "matches_published": round(score, 2) == target,
            **extra}


def _preview(output, name, noisy, estimate, clean):
    array = np.concatenate((noisy, estimate, clean), axis=1)
    Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8)).save(output / (name + ".png"))


def reproduce_table3(directory, output, noise, save_arrays=False):
    """Use fixed historical beta values and the specified NLM comparator."""
    params = parameters()
    rows = []
    for method, radius in (("patch", 2), ("regular", 0), ("nlm", None)):
        for index, sigma in enumerate(params["sigmas"]):
            for image_index, name in enumerate(GRAYSCALE_NAMES):
                clean = load_image(directory, name)
                noisy = corrupt(clean, sigma, noise)
                started = time.perf_counter()
                if method == "nlm":
                    estimate = nonlocal_means(noisy, search_radius=5, patch_radius=2, bandwidth=sigma)
                    extra = {"search_radius": 5, "patch_radius": 2, "bandwidth": sigma}
                else:
                    beta = params[method + "_betas"][index]
                    estimate = patch_beltrami(noisy, beta, .21, 150, radius,
                                              backend="integral", variant="efficient")
                    extra = {"beta": beta, "alpha": .21, "iterations": 150, "radius": radius}
                target = params["table3_targets"][method][str(sigma)][image_index]
                row = _record(method, name, sigma, psnr(clean, estimate), target,
                              seconds=time.perf_counter()-started, parameters_fitted=False,
                              finite=bool(np.isfinite(estimate).all()), **extra)
                rows.append(row)
                if save_arrays:
                    np.savez_compressed(output / f"table3_{method}_{name}_{sigma}.npz",
                                        clean=clean, noisy=noisy, estimate=estimate)
                if method == "patch" and sigma == 20:
                    _preview(output, f"table3_{name}_{sigma}", noisy, estimate, clean)
                print(json.dumps(row), flush=True)
                _write_json(output / "table3.json", rows)
    return rows


def reproduce_color(directory, output, noise, save_arrays=False):
    rows = []
    for settings in parameters()["color"]:
        name, sigma = settings["name"], settings["sigma"]
        clean = load_image(directory, name)
        noisy = corrupt(clean, sigma, noise)
        started = time.perf_counter()
        estimate = patch_beltrami(noisy, settings["beta"], settings["alpha"], 10, 3,
                                  backend="integral", variant="efficient")
        row = _record("color", name, sigma, psnr(clean, estimate), settings["target"],
                      beta=settings["beta"], alpha=settings["alpha"], iterations=10, radius=3,
                      seconds=time.perf_counter()-started, parameters_fitted=False,
                      finite=bool(np.isfinite(estimate).all()))
        rows.append(row)
        if save_arrays:
            np.savez_compressed(output / f"color_{name}_{sigma}.npz", clean=clean, noisy=noisy, estimate=estimate)
        _preview(output, f"color_{name}_{sigma}", noisy, estimate, clean)
        print(json.dumps(row), flush=True)
    _write_json(output / "color.json", rows)
    return rows


def reproduce_table1(directory, output, noise):
    """Per-cell clean-reference beta optimization, as the caption specifies."""
    clean = load_image(directory, "Cameraman256")
    noisy = corrupt(clean, 20, noise)
    rows = []
    for count, targets in parameters()["table1_targets"].items():
        iterations = int(count)
        for radius, target in enumerate(targets):
            evaluations = []

            def objective(log_beta):
                beta = float(np.exp(log_beta))
                estimate = patch_beltrami(noisy, beta, .21, iterations, radius,
                                          backend="integral", variant="efficient")
                score = psnr(clean, estimate)
                evaluations.append({"beta": beta, "psnr_db": score})
                return -score

            optimum = minimize_scalar(objective, method="bounded",
                                      bounds=(np.log(1e-3), np.log(1e8)),
                                      options={"xatol": .0005, "maxiter": 60})
            row = _record("table1", "Cameraman256", 20, float(-optimum.fun), target,
                          iterations=iterations, radius=radius, beta=float(np.exp(optimum.x)),
                          alpha=.21, success=bool(optimum.success), evaluations=evaluations,
                          parameters_fitted=True, fit_reference="same clean Cameraman image")
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "evaluations"}), flush=True)
            _write_json(output / "table1.json", rows)
    return rows


def reproduce(directory="data/ssvm", output="results/ssvm", tables=("1", "3", "color"),
              download=False, save_arrays=False):
    """Recompute requested published results and fail on a displayed-value mismatch."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tables = tuple(tables)
    if not tables or not set(tables) <= {"1", "3", "color"}:
        raise ValueError("tables must contain '1', '3', or 'color'")
    names = set(GRAYSCALE_NAMES if "3" in tables else ())
    if "1" in tables:
        names.add("Cameraman256")
    if "color" in tables:
        names.update(row["name"] for row in parameters()["color"])
    if download:
        fetch_images(directory, names)
    for name in names:
        load_image(directory, name)
    noise = legacy_noise()
    results = {}
    if "1" in tables:
        results["table1"] = reproduce_table1(directory, output, noise)
    if "3" in tables:
        results["table3"] = reproduce_table3(directory, output, noise, save_arrays)
    if "color" in tables:
        results["color"] = reproduce_color(directory, output, noise, save_arrays)
    summary = {"paper": parameters()["paper"], "paper_url": parameters()["paper_url"],
               "protocol": {"input_scale": "0..255", "input_clipping": "0..255",
                            "output_clipping": "none", "precision": "float64",
                            "noise": parameters()["noise"],
                            "parameter_source": parameters()["source_parameters"],
                            "table1_fitting": "oracle beta optimization against clean reference per cell",
                            "table3_color_fitting": "none; historical shared parameters"},
               "experiments": {key: {"cases": len(rows),
                                      "matches": sum(r["matches_published"] for r in rows),
                                      "max_absolute_difference_db": max(abs(r["difference_db"]) for r in rows)}
                               for key, rows in results.items()},
               "passed": all(row["matches_published"] and row.get("finite", True) and
                             row.get("success", True) for rows in results.values() for row in rows)}
    _write_json(output / "summary.json", summary)
    if not summary["passed"]:
        raise RuntimeError(f"Published-precision mismatch; inspect {output / 'summary.json'}")
    return summary
