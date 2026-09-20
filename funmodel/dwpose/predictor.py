from typing import Any

import numpy as np
from funget import simple_download
from fundrive.drives.oss import public_oss_url
from funmodel.core.predict import ImagePredictModel
from funmodel.dwpose.core import util
from funmodel.dwpose.core.wholebody import Wholebody


class DWposePredict(ImagePredictModel):
    """基于 DWPose 的人体全身姿态估计器，封装模型下载、推理与骨架绘制。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pose_estimation: Wholebody | None = None
        self.load()

    def load(self, *args: Any, **kwargs: Any) -> None:
        """下载 DWPose 的检测与姿态 ONNX 权重（若本地缓存缺失）并初始化推理器。"""
        onnx_det = f"{self.cache_path}/yolox_l.onnx"
        onnx_pose = f"{self.cache_path}/dw-ll_ucoco_384.onnx"
        simple_download(
            url=public_oss_url(path="models/dwpose/yolox_l.onnx"), filepath=onnx_det
        )
        simple_download(
            url=public_oss_url(path="models/dwpose/dw-ll_ucoco_384.onnx"),
            filepath=onnx_pose,
        )
        self.pose_estimation = Wholebody(onnx_det=onnx_det, onnx_pose=onnx_pose)

    def draw_image(
        self, image: np.ndarray, result: dict[str, Any], *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """将姿态估计结果绘制为骨架图像。

        Args:
            image: 原始图像（本实现未直接使用，绘制在空白画布上）。
            result: `predict` 返回的姿态字典，包含 height/width/bodies/hand1/hand2/faces。

        Returns:
            绘制了身体、双手、面部关键点的画布图像。
        """
        canvas = np.zeros(shape=(result["height"], result["width"], 3), dtype=np.uint8)
        canvas = util.draw_bodypose(canvas, result["bodies"])
        canvas = util.draw_handpose(canvas, result["hand1"])
        canvas = util.draw_handpose(canvas, result["hand2"])
        canvas = util.draw_facepose(canvas, result["faces"])
        return canvas

    def predict(
        self, ori_img: np.ndarray, draw: bool = False, *args: Any, **kwargs: Any
    ) -> tuple[dict[str, Any], np.ndarray]:
        """对输入图像做全身姿态估计。

        Args:
            ori_img: BGR 格式的输入图像（cv2 读取格式）。
            draw: 是否额外返回绘制了骨架的图像。

        Returns:
            (pose, image) 二元组：pose 为归一化坐标的关键点字典；
            image 为原图，若 draw=True 则替换为绘制了骨架的图像。
        """
        ori_img = ori_img.copy()
        H, W, C = ori_img.shape
        candidate, subset = self.pose_estimation(ori_img)
        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)

        un_visible = subset < 0.3
        candidate[un_visible] = -1

        pose = dict(
            height=H,
            width=W,
            bodies=candidate[:, :18],
            hand1=candidate[:, 92:113],
            hand2=candidate[:, 113:134],
            faces=candidate[:, 24:92],
            foots=candidate[:, 18:24],
            nums=candidate.shape[0],
        )

        if draw:
            ori_img = self.draw_image(image=ori_img, result=pose)
        return pose, ori_img
