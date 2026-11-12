"""Command-line interface for classical Beltrami denoising."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .classical import patch_beltrami, psnr


def _save_preview(path, image):
    Image.fromarray(np.clip(np.rint(image), 0, 255).astype(np.uint8)).save(path)


def demo(output):
    """Create a deterministic synthetic example without network access."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    row, col = np.mgrid[:128, :160]
    clean = 110 + 35*np.sin(row/13) + 25*np.cos(col/17)
    clean += 30 * ((row-64)**2 + (col-80)**2 < 28**2)
    noisy = clean + np.random.default_rng(2011).normal(0, 15, clean.shape)
    estimate = patch_beltrami(noisy, 800, .21, 40, 2, backend="integral", variant="efficient")
    np.savez_compressed(output / "arrays.npz", clean=clean, noisy=noisy, estimate=estimate)
    _save_preview(output / "comparison.png", np.concatenate((noisy, estimate, clean), axis=1))
    metrics = {"noisy_psnr_db": psnr(clean, noisy), "denoised_psnr_db": psnr(clean, estimate),
               "sigma": 15, "beta": 800, "alpha": .21, "iterations": 40, "radius": 2,
               "preview_order": ["noisy", "denoised", "clean"], "finite": bool(np.isfinite(estimate).all())}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(prog="beltrami", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    quick = commands.add_parser("demo", help="run a small synthetic example without downloads")
    quick.add_argument("--output", type=Path, default=Path("results/demo"))
    denoise = commands.add_parser("denoise", help="filter a grayscale/RGB image or .npy array")
    denoise.add_argument("input", type=Path)
    denoise.add_argument("output", type=Path, help=".npy file for full-precision output")
    denoise.add_argument("--beta", type=float, required=True)
    denoise.add_argument("--alpha", type=float, default=.21)
    denoise.add_argument("--iterations", type=int, default=150)
    denoise.add_argument("--radius", type=int, default=2)
    denoise.add_argument("--backend", choices=("integral", "conv2", "separable"), default="integral")
    denoise.add_argument("--variant", choices=("efficient", "pbframework"), default="efficient")
    denoise.add_argument("--precision", choices=("single", "double"))
    denoise.add_argument("--preview", type=Path)
    fetch = commands.add_parser("fetch-ssvm", help="download the seven hash-checked public benchmark images")
    fetch.add_argument("--data-dir", type=Path, default=Path("data/ssvm"))
    suite = commands.add_parser("ssvm", help="reproduce Tables 1/3 and published color examples")
    suite.add_argument("--data-dir", type=Path, default=Path("data/ssvm"))
    suite.add_argument("--output", type=Path, default=Path("results/ssvm"))
    suite.add_argument("--tables", nargs="+", choices=("1", "3", "color"), default=["1", "3", "color"])
    suite.add_argument("--download", action="store_true", help="download missing public benchmark images")
    suite.add_argument("--save-arrays", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "demo":
        result = demo(args.output)
    elif args.command == "fetch-ssvm":
        from .experiments.ssvm import fetch_images
        result = {"images": fetch_images(args.data_dir)}
    elif args.command == "ssvm":
        from .experiments.ssvm import reproduce
        result = reproduce(args.data_dir, args.output, args.tables, args.download, args.save_arrays)
    else:
        if args.output.suffix != ".npy":
            parser.error("denoise output must be .npy; use --preview for a PNG")
        if args.input.suffix == ".npy":
            image = np.load(args.input, allow_pickle=False)
        else:
            with Image.open(args.input) as opened:
                image = np.asarray(opened)
        estimate = patch_beltrami(image, args.beta, args.alpha, args.iterations, args.radius,
                                  backend=args.backend, variant=args.variant, precision=args.precision)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, estimate)
        if args.preview:
            args.preview.parent.mkdir(parents=True, exist_ok=True)
            _save_preview(args.preview, estimate)
        result = {"output": str(args.output), "shape": list(estimate.shape), "dtype": str(estimate.dtype),
                  "finite": bool(np.isfinite(estimate).all())}
    print(json.dumps(result, indent=2))
