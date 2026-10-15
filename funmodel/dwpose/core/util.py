# Copyright 2023 IDEA. All Rights Reserved.
# Copyright 2024 farfarfun
#
# SPDX-License-Identifier: Apache-2.0
#
# 本文件取自 IDEA-Research/DWPose（分支 onnx）的
# ControlNet-v1-1-nightly/annotator/dwpose/util.py，按 Apache License 2.0 使用。
# 本文件 10 个函数全部存在于上游，farfarfun 未新增任何函数，修改仅为中文
# docstring、类型标注与 black 格式化（去注释后 token 级相似度约 91%）。
#
# 注意：本文件的移植链路不止 Apache-2.0 一条。其绘制与手部检测部分上溯至
# CMU OpenPose，该项目许可证标题为「ACADEMIC OR NON-PROFIT ORGANIZATION
# NONCOMMERCIAL RESEARCH USE ONLY」（GitHub 归类 NOASSERTION），既非
# Apache-2.0 亦非 MIT，无法通过署名满足：
#   - handDetect    ← CMU src/openpose/hand/handDetector.cpp 的 Python 转写，
#                     函数内保留该 C++ 源码的逐字注释（见 handDetect 处标注）
#   - draw_bodypose ← CMU COCO-18 骨架：19 组 limbSeq 连接表 + 18 色调色板
#   - draw_handpose ← CMU hand model 的 21 点手部骨架边表
#   - faceDetect    ← lllyasviel/ControlNet（上游注释 "Written by Lvmin"），Apache-2.0
#   - transfer      ← Hzzone/pytorch-openpose（该仓库无 LICENSE 文件）；本包仅走
#                     ONNX 推理，此函数在包内无任何调用点，为死代码
#
# 逐项许可证核实见 NOTICE 与 THIRD_PARTY_LICENSES.md；CMU OpenPose 相关的分发权
# 问题在 https://github.com/farfarfun/todo-list/issues/894 跟踪，尚未结论。

import math
from typing import Any, Mapping

import cv2
import matplotlib
import numpy as np


def smart_resize(x: np.ndarray, s: tuple[int, int]) -> np.ndarray:
    """按目标尺寸缩放二维图像或多通道数组。

    Args:
        x: 待缩放的图像数组。
        s: 目标尺寸，格式为（高，宽）。

    Returns:
        缩放后的图像数组。
    """
    Ht, Wt = s
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    if Co == 3 or Co == 1:
        k = float(Ht + Wt) / float(Ho + Wo)
        return cv2.resize(
            x,
            (int(Wt), int(Ht)),
            interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4,
        )
    else:
        return np.stack([smart_resize(x[:, :, i], s) for i in range(Co)], axis=2)


def smart_resize_k(x: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """按水平和垂直比例缩放图像。

    Args:
        x: 待缩放的图像数组。
        fx: 水平方向缩放比例。
        fy: 垂直方向缩放比例。

    Returns:
        缩放后的图像数组。
    """
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    Ht, Wt = Ho * fy, Wo * fx
    if Co == 3 or Co == 1:
        k = float(Ht + Wt) / float(Ho + Wo)
        return cv2.resize(
            x,
            (int(Wt), int(Ht)),
            interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4,
        )
    else:
        return np.stack([smart_resize_k(x[:, :, i], fx, fy) for i in range(Co)], axis=2)


def padRightDownCorner(
    img: np.ndarray, stride: int, padValue: int | float
) -> tuple[np.ndarray, list[int]]:
    """在图像右侧和下方填充像素，使尺寸成为步长的整数倍。

    Args:
        img: 待填充的图像数组。
        stride: 尺寸对齐步长。
        padValue: 填充像素值。

    Returns:
        填充后的图像和四个方向的填充量。
    """
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def transfer(model: Any, model_weights: Mapping[str, Any]) -> dict[str, Any]:
    """将外部权重名称转换为模型状态字典使用的名称。

    Args:
        model: 提供 ``state_dict`` 方法的模型。
        model_weights: 以外部参数名为键的权重映射。

    Returns:
        可供模型加载的权重字典。
    """
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights[
            ".".join(weights_name.split(".")[1:])
        ]
    return transfered_model_weights


def draw_bodypose(canvas: np.ndarray, bodies: Any) -> np.ndarray:
    """在画布上绘制人体关键点和骨架。

    Args:
        canvas: 待绘制的图像数组。
        bodies: 人体关键点集合。

    Returns:
        绘制人体骨架后的画布。
    """
    H, W, C = canvas.shape
    bodies = np.array(bodies)

    stickwidth = 4

    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        [3, 17],
        [6, 18],
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]
    for body in bodies:
        for i in range(17):
            index = np.array([np.array(limbSeq[i]) - 1])
            Y = body[index.astype(int), 0][0] * float(W)
            X = body[index.astype(int), 1][0] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            if X.min() < 0 or Y.min() < 0:
                continue
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)
    for body in bodies:
        for i in range(18):
            x, y = body[i][0:2]
            x = int(x * W)
            y = int(y * H)
            if x < 0 or y < 0:
                continue
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas


def draw_handpose(
    canvas: np.ndarray, hands: Any, eps: float = 0.01
) -> np.ndarray:
    """在画布上绘制手部关键点和连线。

    Args:
        canvas: 待绘制的图像数组。
        hands: 手部关键点集合。
        eps: 判定关键点有效的最小坐标值。

    Returns:
        绘制手部姿态后的画布。
    """
    H, W, C = canvas.shape

    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]

    for peaks in hands:
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
                    * 255,
                    thickness=2,
                )

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(
    canvas: np.ndarray, all_lmks: Any, eps: float = 0.01
) -> np.ndarray:
    """在画布上绘制面部关键点。

    Args:
        canvas: 待绘制的图像数组。
        all_lmks: 面部关键点集合。
        eps: 判定关键点有效的最小坐标值。

    Returns:
        绘制面部关键点后的画布。
    """
    H, W, C = canvas.shape
    for lmks in all_lmks:
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas


# ---------------------------------------------------------------------------
# 许可证标注（farfarfun 补充，非上游内容）
#
# 以下 handDetect 是 CMU OpenPose `src/openpose/hand/handDetector.cpp` 的
# Python 逐行转写，并且把该 C++ 源码的注释一起搬了过来：函数体内以
# `# handRectangle.…` / `# const auto distance…` 开头的几行即 CMU 原文
# 第 25-29 行与第 34-35 行；常量 `ratioWristElbow = 0.33` 对应 CMU 第 20 行
# `const auto ratioWristElbow = 0.33f;`。
#
# CMU OpenPose 的许可证为「学术/非营利机构非商业研究专用」，不是 Apache-2.0，
# 也不是 MIT，因此这一部分的许可证状况**无法通过本仓库的署名补正来满足**。
# 事实核实见 THIRD_PARTY_LICENSES.md，处置在
# https://github.com/farfarfun/todo-list/issues/894 跟踪，尚未结论。
# ---------------------------------------------------------------------------
# detect hand according to body pose keypoints
# please refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/hand/handDetector.cpp
def handDetect(
    candidate: np.ndarray, subset: np.ndarray, oriImg: np.ndarray
) -> list[list[int | bool]]:
    """根据人体关键点估算手部检测区域。

    Args:
        candidate: 候选人体关键点坐标。
        subset: 每个人体使用的关键点索引。
        oriImg: 原始图像。

    Returns:
        手部方框列表，每项包含左上角、边长和是否为左手。
    """
    # right hand: wrist 4, elbow 3, shoulder 2
    # left hand: wrist 7, elbow 6, shoulder 5
    ratioWristElbow = 0.33
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]
    for person in subset.astype(int):
        # if any of three not detected
        has_left = np.sum(person[[5, 6, 7]] == -1) == 0
        has_right = np.sum(person[[2, 3, 4]] == -1) == 0
        if not (has_left or has_right):
            continue
        hands = []
        # left hand
        if has_left:
            left_shoulder_index, left_elbow_index, left_wrist_index = person[[5, 6, 7]]
            x1, y1 = candidate[left_shoulder_index][:2]
            x2, y2 = candidate[left_elbow_index][:2]
            x3, y3 = candidate[left_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, True])
        # right hand
        if has_right:
            right_shoulder_index, right_elbow_index, right_wrist_index = person[
                [2, 3, 4]
            ]
            x1, y1 = candidate[right_shoulder_index][:2]
            x2, y2 = candidate[right_elbow_index][:2]
            x3, y3 = candidate[right_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, False])

        for x1, y1, x2, y2, x3, y3, is_left in hands:
            # pos_hand = pos_wrist + ratio * (pos_wrist - pos_elbox) = (1 + ratio) * pos_wrist - ratio * pos_elbox
            # handRectangle.x = posePtr[wrist*3] + ratioWristElbow * (posePtr[wrist*3] - posePtr[elbow*3]);
            # handRectangle.y = posePtr[wrist*3+1] + ratioWristElbow * (posePtr[wrist*3+1] - posePtr[elbow*3+1]);
            # const auto distanceWristElbow = getDistance(poseKeypoints, person, wrist, elbow);
            # const auto distanceElbowShoulder = getDistance(poseKeypoints, person, elbow, shoulder);
            # handRectangle.width = 1.5f * fastMax(distanceWristElbow, 0.9f * distanceElbowShoulder);
            x = x3 + ratioWristElbow * (x3 - x2)
            y = y3 + ratioWristElbow * (y3 - y2)
            distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
            # x-y refers to the center --> offset to topLeft point
            # handRectangle.x -= handRectangle.width / 2.f;
            # handRectangle.y -= handRectangle.height / 2.f;
            x -= width / 2
            y -= width / 2  # width = height
            # overflow the image
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            width1 = width
            width2 = width
            if x + width > image_width:
                width1 = image_width - x
            if y + width > image_height:
                width2 = image_height - y
            width = min(width1, width2)
            # the max hand box value is 20 pixels
            if width >= 20:
                detect_result.append([int(x), int(y), int(width), is_left])

    """
    return value: [[x, y, w, True if left hand else False]].
    width=height since the network require squared input.
    x, y is the coordinate of top left 
    """
    return detect_result


# Written by Lvmin
def faceDetect(
    candidate: np.ndarray, subset: np.ndarray, oriImg: np.ndarray
) -> list[list[int]]:
    """根据人体关键点估算面部检测区域。

    Args:
        candidate: 候选人体关键点坐标。
        subset: 每个人体使用的关键点索引。
        oriImg: 原始图像。

    Returns:
        面部方框列表，每项包含左上角和边长。
    """
    # left right eye ear 14 15 16 17
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]
    for person in subset.astype(int):
        has_head = person[0] > -1
        if not has_head:
            continue

        has_left_eye = person[14] > -1
        has_right_eye = person[15] > -1
        has_left_ear = person[16] > -1
        has_right_ear = person[17] > -1

        if not (has_left_eye or has_right_eye or has_left_ear or has_right_ear):
            continue

        head, left_eye, right_eye, left_ear, right_ear = person[[0, 14, 15, 16, 17]]

        width = 0.0
        x0, y0 = candidate[head][:2]

        if has_left_eye:
            x1, y1 = candidate[left_eye][:2]
            d = max(abs(x0 - x1), abs(y0 - y1))
            width = max(width, d * 3.0)

        if has_right_eye:
            x1, y1 = candidate[right_eye][:2]
            d = max(abs(x0 - x1), abs(y0 - y1))
            width = max(width, d * 3.0)

        if has_left_ear:
            x1, y1 = candidate[left_ear][:2]
            d = max(abs(x0 - x1), abs(y0 - y1))
            width = max(width, d * 1.5)

        if has_right_ear:
            x1, y1 = candidate[right_ear][:2]
            d = max(abs(x0 - x1), abs(y0 - y1))
            width = max(width, d * 1.5)

        x, y = x0, y0

        x -= width
        y -= width

        if x < 0:
            x = 0

        if y < 0:
            y = 0

        width1 = width * 2
        width2 = width * 2

        if x + width > image_width:
            width1 = image_width - x

        if y + width > image_height:
            width2 = image_height - y

        width = min(width1, width2)

        if width >= 20:
            detect_result.append([int(x), int(y), int(width)])

    return detect_result


# get max index of 2d array
def npmax(array: np.ndarray) -> tuple[int, int]:
    """返回二维数组最大值的行列索引。

    Args:
        array: 待查找的二维数组。

    Returns:
        最大值所在的行索引和列索引。
    """
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j
