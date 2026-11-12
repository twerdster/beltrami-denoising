from pathlib import Path

import numpy as np
import pytest

from beltrami import beltrami_original, patch_beltrami, psnr
from beltrami.classical import box_sum, dminus, dplus
from beltrami.cli import demo, main
from beltrami.nonlocal_means import nonlocal_means


def test_box_sums_and_derivative_boundaries():
    array = np.arange(35, dtype=float).reshape(5, 7)
    for radius in (0, 1, 3, 8):
        direct = np.empty_like(array)
        for row in range(5):
            for col in range(7):
                direct[row, col] = array[max(0, row-radius):row+radius+1,
                                         max(0, col-radius):col+radius+1].sum()
        for backend in ("conv2", "integral", "separable"):
            np.testing.assert_array_equal(box_sum(array, radius, backend), direct)
    np.testing.assert_array_equal(dminus(array, 0)[0], 0)
    np.testing.assert_array_equal(dminus(array, 1)[:, 0], 0)
    np.testing.assert_array_equal(dplus(array, 0)[-1], 0)
    np.testing.assert_array_equal(dplus(array, 1)[:, -1], 0)


@pytest.mark.parametrize("shape", [(9, 11), (9, 11, 3), (1, 8), (8, 1), (1, 1, 3)])
def test_constant_images(shape):
    array = np.full(shape, 72.5)
    for backend in ("conv2", "integral", "separable"):
        for variant in ("efficient", "pbframework"):
            np.testing.assert_array_equal(
                patch_beltrami(array, 100, .21, 4, 3, backend=backend, variant=variant), array)


def test_octave_reference_arrays():
    """Synthetic arrays evaluated with the original MATLAB flow in Octave."""
    fixture = Path(__file__).parent / "data/flow_reference.npz"
    with np.load(fixture, allow_pickle=False) as data:
        for variant, precision in (("efficient", "double"), ("pbframework", "single")):
            actual = patch_beltrami(data["A"], 800, .21, 4, 2,
                                    backend="integral", variant=variant, precision=precision)
            expected = data[variant]
            np.testing.assert_allclose(actual, expected, atol=2e-10 if precision == "double" else 3e-5,
                                       rtol=2e-12 if precision == "double" else 1e-6)


def test_original_source_ignores_alpha():
    array = np.arange(36, dtype=float).reshape(6, 6)
    np.testing.assert_array_equal(beltrami_original(array, 100, .01, 3),
                                  beltrami_original(array, 100, .4, 3))


def test_input_errors_and_psnr():
    with pytest.raises(ValueError):
        patch_beltrami(np.zeros((4, 4, 4)), 100)
    with pytest.raises(ValueError):
        patch_beltrami(np.zeros((4, 4)), 0)
    with pytest.raises(ValueError):
        patch_beltrami(np.zeros((4, 4)), 100, ws=-1)
    assert psnr(np.array([0], dtype=np.uint8), np.array([255], dtype=np.uint8)) == 0


def test_nlm_keeps_constant_and_singleton_images():
    for shape in ((8, 9), (1, 1), (1, 5)):
        image = np.full(shape, 73.0)
        np.testing.assert_allclose(nonlocal_means(image, bandwidth=20), image, atol=1e-12)


def test_nlm_octave_reference_arrays():
    fixture = Path(__file__).parent / "data/nlm_reference.npz"
    with np.load(fixture, allow_pickle=False) as data:
        for index in range(1, 5):
            actual = nonlocal_means(data[f"input{index}"], bandwidth=float(data[f"h{index}"]))
            np.testing.assert_allclose(actual, data[f"expected{index}"], atol=1e-10, rtol=1e-12)


def test_demo_and_cli(tmp_path):
    metrics = demo(tmp_path / "demo")
    assert metrics["finite"]
    assert metrics["denoised_psnr_db"] > metrics["noisy_psnr_db"] + 2
    assert (tmp_path / "demo/comparison.png").is_file()
    np.save(tmp_path / "input.npy", np.full((3, 4), 42.0))
    main(["denoise", str(tmp_path / "input.npy"), str(tmp_path / "output.npy"),
          "--beta", "100", "--iterations", "2"])
    np.testing.assert_array_equal(np.load(tmp_path / "output.npy"), 42)
