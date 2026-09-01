"""The KITTI lidar-to-image projection, checked against its own docstring.

`_kitti_lidar2img` says:

    3x4 projection: lidar2img = P2 @ R0_rect @ Tr_velo_to_cam

`make_iterator(dataset="kitti"|"waymokitti", ...)` reaches it through
`iter_kitti_like`, which builds it for every frame that has a calib file.

Pure numpy: the calibration values below are a real KITTI `calib.txt` written
out as arrays, so no dataset download and no GPU. Nothing here touches mmdet3d.

    pytest DeepDataMiningLearning/bevdet/tests/test_kitti_lidar2img.py
"""
import os
import sys

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from DeepDataMiningLearning.bevdet.dataset_resolver import (  # noqa: E402
    _kitti_lidar2img, _read_kitti_calib,
)

# KITTI object 000000 calib.txt, split into the three matrices the reader
# returns. R0_rect and Tr_velo_to_cam are 4x4 there, as _read_kitti_calib
# builds them.
P2 = np.array([[721.5377, 0.0, 609.5593, 44.85728],
               [0.0, 721.5377, 172.854, 0.2163791],
               [0.0, 0.0, 1.0, 0.002745884]], dtype=np.float32)

R0 = np.eye(4, dtype=np.float32)
R0[:3, :3] = np.array([[0.9999239, 0.00983776, -0.00744505],
                       [-0.0098698, 0.9999421, -0.00427846],
                       [0.00740253, 0.00435161, 0.9999631]], dtype=np.float32)

TR = np.eye(4, dtype=np.float32)
TR[:3, :] = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                      [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
                      [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01]],
                     dtype=np.float32)

CALIB = {"P2": P2, "R0_rect": R0, "Tr_velo_to_cam": TR}


def test_the_inputs_have_the_shapes_the_reader_produces():
    """Guards the premise: P2 is 3x4 and the other two are 4x4."""
    assert P2.shape == (3, 4)
    assert R0.shape == (4, 4)
    assert TR.shape == (4, 4)


def test_it_returns_a_projection_instead_of_raising():
    m = _kitti_lidar2img(CALIB)
    assert m.shape == (3, 4)
    assert m.dtype == np.float32
    assert np.isfinite(m).all()


def test_it_equals_the_product_its_docstring_states():
    np.testing.assert_allclose(_kitti_lidar2img(CALIB), P2 @ (R0 @ TR),
                               rtol=1e-5, atol=1e-4)


def test_a_lidar_point_lands_in_front_of_the_camera():
    """A point 10 m ahead on the lidar x axis should project with w > 0."""
    m = _kitti_lidar2img(CALIB)
    uvw = m @ np.array([10.0, 0.0, 0.0, 1.0], dtype=np.float32)
    assert uvw[2] > 0, "depth came out behind the camera"
    u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
    assert 0 < u < 1242 and 0 < v < 375, f"({u:.1f}, {v:.1f}) is off the KITTI image"


def test_a_point_behind_the_lidar_projects_behind_the_camera():
    m = _kitti_lidar2img(CALIB)
    assert (m @ np.array([-10.0, 0.0, 0.0, 1.0], dtype=np.float32))[2] < 0


def test_moving_along_the_lidar_y_axis_moves_the_pixel_horizontally():
    """Sanity on the axis convention: KITTI lidar +y is to the left."""
    m = _kitti_lidar2img(CALIB)

    def u_of(y):
        uvw = m @ np.array([10.0, y, 0.0, 1.0], dtype=np.float32)
        return uvw[0] / uvw[2]

    assert u_of(2.0) < u_of(0.0) < u_of(-2.0)


def test_the_rectification_translation_is_not_discarded():
    """The 4x4 R0_rect can carry a translation; slicing it to 3x4 loses it."""
    shifted = R0.copy()
    shifted[:3, 3] = [0.5, -0.25, 0.75]
    a = _kitti_lidar2img({**CALIB, "R0_rect": shifted})
    b = _kitti_lidar2img(CALIB)
    assert not np.allclose(a, b), "a translation in R0_rect changed nothing"


def test_it_survives_the_reader_end_to_end(tmp_path):
    """_read_kitti_calib -> _kitti_lidar2img, the path iter_kitti_like takes."""
    def row(name, m):
        return name + ": " + " ".join(f"{v:.12e}" for v in m.ravel())

    calib = tmp_path / "000000.txt"
    calib.write_text("\n".join([
        row("P0", P2), row("P1", P2), row("P2", P2), row("P3", P2),
        row("R0_rect", R0[:3, :3]),
        row("Tr_velo_to_cam", TR[:3, :]),
        row("Tr_imu_to_velo", TR[:3, :]),
    ]) + "\n", encoding="utf-8")

    m = _kitti_lidar2img(_read_kitti_calib(str(calib)))
    assert m.shape == (3, 4)
    np.testing.assert_allclose(m, P2 @ (R0 @ TR), rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("missing", ["P2", "R0_rect", "Tr_velo_to_cam"])
def test_an_incomplete_calib_still_raises_from_the_reader(missing, tmp_path):
    """Unchanged behaviour: the reader, not the projection, reports this."""
    def row(name, m):
        return name + ": " + " ".join(f"{v:.12e}" for v in m.ravel())

    rows = {"P2": row("P2", P2),
            "R0_rect": row("R0_rect", R0[:3, :3]),
            "Tr_velo_to_cam": row("Tr_velo_to_cam", TR[:3, :])}
    del rows[missing]
    calib = tmp_path / "bad.txt"
    calib.write_text("\n".join(rows.values()) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Incomplete calib"):
        _read_kitti_calib(str(calib))
