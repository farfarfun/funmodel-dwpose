# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
from typing import Any
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from . import util
from .wholebody import Wholebody

def draw_pose(pose: dict[str, Any], H: int, W: int) -> np.ndarray:
    """将完整姿态结果绘制到空白画布。

    Args:
        pose: 包含人体、手部和面部关键点的姿态结果。
        H: 输出画布高度。
        W: 输出画布宽度。

    Returns:
        绘制姿态骨架后的图像。
    """
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate)

    canvas = util.draw_handpose(canvas, hands)

    canvas = util.draw_facepose(canvas, faces)

    return canvas


class DWposeDetector:
    """执行全身姿态估计并生成骨架图的检测器。"""

    def __init__(self) -> None:
        """初始化全身姿态估计模型。"""

        self.pose_estimation = Wholebody()

    def __call__(self, oriImg: np.ndarray) -> np.ndarray:
        """检测输入图像中的人体姿态并绘制骨架。

        Args:
            oriImg: BGR 格式的输入图像。

        Returns:
            与输入图像尺寸一致的姿态骨架图。
        """
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)

            un_visible = subset<0.3
            candidate[un_visible] = -1

            body = candidate[:, :18].copy()

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            
            bodies = dict(candidate=body)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            return draw_pose(pose, H, W)
