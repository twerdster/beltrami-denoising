import hashlib

import numpy as np
from PIL import Image
import pytest

from beltrami.experiments import ssvm


def test_noise_and_parameter_manifest():
    params = ssvm.parameters()
    noise = ssvm.legacy_noise()
    assert noise.shape == (512*512*3,)
    assert len(params["table1_targets"]) == 4
    assert len(params["table3_targets"]) == 3
    assert params["patch_betas"][1] == pytest.approx(128.85700242394654)
    assert all(len(row["file_sha256"]) == 64 for row in ssvm.image_sources())


def test_noise_column_major_reshape_and_input_clipping():
    clean = np.full((2, 3), 100.0)
    noise = np.arange(6, dtype=float)
    expected = np.array([[100, 102, 104], [101, 103, 105]])
    np.testing.assert_array_equal(ssvm.corrupt(clean, 1, noise), expected)
    np.testing.assert_array_equal(ssvm.corrupt(clean, 1000, noise), [[100, 255, 255], [255, 255, 255]])


def test_cached_download_checks_hash_without_network(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.png"
    array = np.arange(12, dtype=np.uint8).reshape(3, 4)
    Image.fromarray(array).save(path)
    source = {"name": "synthetic", "url": "https://example.invalid/unused.png",
              "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "pixels_sha256": hashlib.sha256(array.tobytes()).hexdigest(), "shape": [3, 4]}
    monkeypatch.setattr(ssvm, "image_sources", lambda: [source])
    monkeypatch.setattr(ssvm, "urlopen", lambda *a, **kw: pytest.fail("cache should avoid network"))
    assert ssvm.fetch_images(tmp_path) == [str(path)]
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        ssvm.fetch_images(tmp_path)
