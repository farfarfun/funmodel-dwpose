# DWposePredict.__init__ 会自动下载权重并加载 ONNX 会话，测试中通过 monkeypatch
# 跳过 load()，直接构造对象后注入假的 pose_estimation，覆盖 predict/draw_image
# 的坐标归一化、可见性过滤与绘制路径，不依赖真实网络/模型文件。
import numpy as np
import pytest

from funmodel.dwpose import DWposePredict


class _FakePoseEstimation:
    """返回固定关键点坐标的假推理器，模拟 Wholebody.__call__ 的输出形状。"""

    def __call__(self, img: np.ndarray):
        candidate = np.ones((1, 134, 2), dtype=np.float64) * 10.0
        subset = np.ones((1, 134), dtype=np.float64)
        subset[:, :5] = 0.1  # 前 5 个关键点标记为不可见，用于覆盖过滤逻辑
        return candidate, subset


@pytest.fixture
def predictor(monkeypatch: pytest.MonkeyPatch) -> DWposePredict:
    monkeypatch.setattr(DWposePredict, "load", lambda self, *a, **k: None)
    p = DWposePredict()
    p.pose_estimation = _FakePoseEstimation()
    return p


def test_predict_normalizes_coordinates_by_image_size(predictor: DWposePredict) -> None:
    img = np.zeros((100, 50, 3), dtype=np.uint8)  # H=100, W=50
    pose, out_img = predictor.predict(img, draw=False)

    assert pose["height"] == 100
    assert pose["width"] == 50
    # 取第 10 个关键点（不在前 5 个低置信度掩码范围内）
    # x 坐标除以宽度 50，y 坐标除以高度 100
    assert pose["bodies"][0, 10, 0] == pytest.approx(10.0 / 50)
    assert pose["bodies"][0, 10, 1] == pytest.approx(10.0 / 100)
    # predict() 内部对输入图像做了 copy()，返回的是副本而非原对象
    assert out_img is not img
    np.testing.assert_array_equal(out_img, img)


def test_predict_marks_low_confidence_points_invisible(predictor: DWposePredict) -> None:
    pose, _ = predictor.predict(np.zeros((10, 10, 3), dtype=np.uint8))
    # subset < 0.3 的前 5 个关键点应被置为 -1
    assert (pose["bodies"][0, :5] == -1).all()


def test_predict_with_draw_returns_canvas_image(predictor: DWposePredict) -> None:
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    pose, drawn = predictor.predict(img, draw=True)

    assert drawn.shape == (pose["height"], pose["width"], 3)
    assert drawn.dtype == np.uint8


def test_predict_keypoint_group_slicing_is_disjoint(predictor: DWposePredict) -> None:
    pose, _ = predictor.predict(np.zeros((10, 10, 3), dtype=np.uint8))
    assert pose["bodies"].shape[1] == 18
    assert pose["hand1"].shape[1] == 21
    assert pose["hand2"].shape[1] == 21
    assert pose["faces"].shape[1] == 68
    assert pose["foots"].shape[1] == 6
