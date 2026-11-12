"""Procedural float32 reference arrays for the Swift/Metal validation harness.

Run in an environment with beltrami-denoising installed. No photographs,
network downloads or model assets are required.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from beltrami import patch_beltrami


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [
        ("singleton_gray", 1, 1, 1, 0, 4),
        ("singleton_rgb", 1, 1, 3, 3, 4),
        ("one_row", 17, 1, 1, 2, 7),
        ("one_column", 1, 19, 3, 3, 7),
        ("small_rgb", 3, 2, 3, 3, 5),
        ("small_gray", 2, 3, 1, 2, 5),
        ("local_gray", 19, 17, 1, 0, 10),
        ("local_rgb", 17, 19, 3, 0, 10),
        ("odd_gray", 31, 29, 1, 2, 20),
        ("odd_rgb", 29, 31, 3, 3, 20),
        ("long_gray", 65, 63, 1, 2, 150),
        ("long_rgb", 67, 65, 3, 3, 30),
        ("constant_gray", 43, 37, 1, 3, 25),
        ("constant_rgb", 43, 37, 3, 3, 25),
        ("tile_gray", 43, 41, 1, 2, 7),
        ("tile_rgb", 43, 41, 3, 3, 7),
        ("nonsquare_gray", 129, 127, 1, 2, 10),
        ("nonsquare_rgb", 127, 129, 3, 3, 10),
    ]
    rng = np.random.default_rng(2011)
    manifest = []
    for name, width, height, channels, radius, iterations in cases:
        shape = (height, width) if channels == 1 else (height, width, channels)
        array = rng.normal(110, 30, shape).astype(np.float32)
        if name.startswith("constant"):
            array[...] = 73 if channels == 1 else np.array([20, 110, 240], np.float32)
        beta = 100 if radius == 0 else 800
        expected = patch_beltrami(array, beta, .21, iterations, radius,
                                  variant="efficient", precision="single", backend="conv2")
        array.astype("<f4").tofile(args.output / f"{name}.input.f32")
        expected.astype("<f4").tofile(args.output / f"{name}.expected.f32")
        manifest.append({"name": name, "width": width, "height": height,
                         "channels": channels, "radius": radius, "iterations": iterations,
                         "beta": beta, "alpha": .21,
                         "input": f"{name}.input.f32", "expected": f"{name}.expected.f32",
                         "forceTiling": name.startswith("tile_")})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(manifest)} procedural float32 reference cases")


if __name__ == "__main__":
    main()
