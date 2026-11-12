#!/usr/bin/env python3
"""Prepare procedural Python float32 references for the iOS app.

Without --flat-dir, the current interpreter must have beltrami-denoising
installed. Adapting an existing Tests/generate_fixtures.py output requires
only the Python standard library. No images or model files are downloaded.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


REPOSITORY = Path(__file__).resolve().parents[2]


def local_file(directory, filename):
    """Accept only a filename directly inside the flat fixture directory."""
    if not isinstance(filename, str) or Path(filename).name != filename or filename in ("", ".", ".."):
        raise ValueError(f"Invalid fixture filename: {filename!r}")
    path = directory / filename
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing regular fixture file: {filename}")
    return path


def prepare(flat_dir, output):
    manifest = json.loads((flat_dir / "manifest.json").read_text())
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("Expected a nonempty flat fixture manifest list")
    cases = []
    copies = []
    identifiers = set()
    for row in manifest:
        name = row["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in identifiers:
            raise ValueError(f"Invalid or duplicate fixture ID: {name!r}")
        identifiers.add(name)
        for key in ("width", "height", "channels", "radius", "iterations"):
            if type(row[key]) is not int:
                raise ValueError(f"{name}: {key} must be an integer")
        if row["width"] < 1 or row["height"] < 1 or row["channels"] not in (1, 3):
            raise ValueError(f"{name}: require positive dimensions and 1 or 3 channels")
        if row["radius"] < 0 or row["iterations"] < 0:
            raise ValueError(f"{name}: radius and iterations must be nonnegative")
        beta, alpha = float(row["beta"]), float(row["alpha"])
        if not math.isfinite(beta) or beta <= 0 or not math.isfinite(alpha) or alpha < 0:
            raise ValueError(f"{name}: invalid beta or alpha")
        byte_count = row["width"] * row["height"] * row["channels"] * 4
        case = {"id": name, **{key: row[key] for key in
                ("width", "height", "channels", "beta", "alpha", "radius", "iterations")},
                "forceTiling": bool(row.get("forceTiling", False))}
        for kind in ("input", "expected"):
            source = local_file(flat_dir, row[kind])
            if source.stat().st_size != byte_count:
                raise ValueError(f"{name}: {kind} has incorrect float32 byte count")
            filename = f"{kind}.f32"
            case[kind] = {"file": filename, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            copies.append((source, output / name / filename))
        cases.append(case)

    # Validate every source before replacing the manifest. Existing unrelated
    # folders are retained; the manifest selects which cases the app executes.
    output.mkdir(parents=True, exist_ok=True)
    for source, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    result = {"schema": 1, "reference": {"implementation": "beltrami.patch_beltrami",
              "variant": "efficient", "precision": "single", "backend": "conv2",
              "layout": "H,W,channels; row-major; little-endian float32",
              "intensity_peak": 255}, "cases": cases}
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output / "manifest.json")
    print(f"Prepared {len(cases)} procedural float32 cases in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPOSITORY / "ios/BeltramiApp/Fixtures")
    parser.add_argument("--flat-dir", type=Path,
                        help="adapt an existing flat reference directory without importing beltrami")
    args = parser.parse_args()
    if args.flat_dir is not None:
        prepare(args.flat_dir.resolve(), args.output.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="beltrami-ios-fixtures-") as temporary:
            subprocess.run([sys.executable, str(REPOSITORY / "ios/Tests/generate_fixtures.py"), temporary],
                           check=True)
            prepare(Path(temporary), args.output.resolve())


if __name__ == "__main__":
    main()
