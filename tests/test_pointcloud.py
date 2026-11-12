import json

import numpy as np
import pytest

from beltrami import patch_beltrami
from beltrami.pointcloud import denoise_points, gridfit, main, patch_step


def _surface():
    x, y = np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 7))
    z = .1 * np.sin(2*x) * np.cos(3*y)
    z += np.random.default_rng(2012).normal(0, .01, z.shape)
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def test_patch_update_matches_image_flow():
    z = np.random.default_rng(2012).normal(0, .2, (9, 11))
    expected = patch_beltrami(z, .7, .19, 1, 2, backend="conv2", variant="efficient") - z
    np.testing.assert_allclose(patch_step(z, .7, .19, 2), expected, atol=2e-15, rtol=2e-12)


def test_gridfit_preserves_a_plane():
    x, y = np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 7))
    points = np.column_stack((x.ravel(), y.ravel(), (.2*x-.3*y+.5).ravel()))
    fitted, _, _ = gridfit(points, np.linspace(-1, 1, 7))
    np.testing.assert_allclose(fitted, .2*x-.3*y+.5, atol=5e-13, rtol=1e-12)


def test_noiseless_planar_invariance_and_no_input_mutation():
    x, y = np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 7))
    points = np.column_stack((x.ravel(), y.ravel(), (.2*x-.1*y+.3).ravel()))
    unchanged = points.copy()
    actual, metadata = denoise_points(points, 0, iterations=2, neighbors=12, grid=7, radius=1)
    np.testing.assert_array_equal(points, unchanged)
    np.testing.assert_allclose(actual, points, atol=1e-12, rtol=1e-12)
    assert metadata["neighbors"] >= 12


def test_cli_roundtrip(tmp_path):
    points = _surface()
    source, target = tmp_path / "points.npy", tmp_path / "filtered.npy"
    np.save(source, points)
    expected, _ = denoise_points(points, .02, iterations=2, neighbors=12, grid=7, radius=1)
    main([str(source), str(target), "--noise-std", ".02", "--iterations", "2",
          "--neighbors", "12", "--grid", "7", "--radius", "1"])
    actual = np.load(target)
    np.testing.assert_array_equal(actual, expected)
    report = json.loads(target.with_suffix(".json").read_text())
    assert report["status"] == "complete" and report["finite"]
    assert report["points"] == len(points) and len(report["history"]) == 2


def test_demo_new_rejects_undefined_noise_and_local_radius():
    points = _surface()
    with pytest.raises(ValueError, match="positive|noise"):
        denoise_points(points, 0, mode="demo_new")
    repeated = np.vstack((np.zeros((4, 3)), points))
    with pytest.raises(ValueError, match="repeated|neighbor|radius|duplicate"):
        denoise_points(repeated, .02, neighbors=12, mode="demo_new")


def test_invalid_points_and_zero_global_radius():
    with pytest.raises(ValueError):
        denoise_points(np.zeros((3, 3)), .02)
    with pytest.raises(ValueError, match="radius|repeated"):
        denoise_points(np.zeros((8, 3)), .02, neighbors=8)
