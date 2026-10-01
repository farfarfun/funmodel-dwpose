
import cv2
import numpy as np
import onnxruntime as ort

def preprocess(
    img: np.ndarray,
    out_bbox: np.ndarray | list[list[float]],
    input_size: tuple[int, int] = (192, 256),
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """为 RTMPose 推理预处理图像。

    Args:
        img: 输入图像。
        out_bbox: 待估计姿态的人体边界框。
        input_size: 模型输入尺寸，格式为 (宽, 高)。

    Returns:
        预处理图像、边界框中心和缩放尺寸。
    """
    # 获取图像尺寸。
    img_shape = img.shape[:2]
    out_img, out_center, out_scale = [], [], []
    if len(out_bbox) == 0:
        out_bbox = [[0, 0, img_shape[1], img_shape[0]]]
    for i in range(len(out_bbox)):
        x0 = out_bbox[i][0]
        y0 = out_bbox[i][1]
        x1 = out_bbox[i][2]
        y1 = out_bbox[i][3]
        bbox = np.array([x0, y0, x1, y1])

        # 计算边界框中心和缩放尺寸。
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)

        # 执行仿射变换。
        resized_img, scale = top_down_affine(input_size, scale, center, img)

        # 按模型参数归一化图像。
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        resized_img = (resized_img - mean) / std

        out_img.append(resized_img)
        out_center.append(center)
        out_scale.append(scale)

    return out_img, out_center, out_scale


def inference(
    sess: ort.InferenceSession, img: list[np.ndarray]
) -> list[list[np.ndarray]]:
    """执行 RTMPose 模型推理。

    Args:
        sess: ONNX Runtime 会话。
        img: 已预处理的图像批次。

    Returns:
        模型输出列表。
    """
    all_out = []
    # 构造模型输入。
    for i in range(len(img)):
        input = [img[i].transpose(2, 0, 1)]

        # 收集模型输出名称。
        sess_input = {sess.get_inputs()[0].name: input}
        sess_output = []
        for out in sess.get_outputs():
            sess_output.append(out.name)

        # 执行推理。
        outputs = sess.run(sess_output, sess_input)
        all_out.append(outputs)

    return all_out


def postprocess(
    outputs: list[list[np.ndarray]],
    model_input_size: tuple[int, int],
    center: list[np.ndarray],
    scale: list[np.ndarray],
    simcc_split_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """后处理 RTMPose 模型输出。

    Args:
        outputs: 模型输出。
        model_input_size: 模型输入尺寸。
        center: 边界框中心坐标 (x, y)。
        scale: 边界框尺寸 (宽, 高)。
        simcc_split_ratio: SimCC 分割比例。

    Returns:
        缩放回原图坐标的关键点和置信度。
    """
    all_key = []
    all_score = []
    for i in range(len(outputs)):
        # 解码 SimCC 输出。
        simcc_x, simcc_y = outputs[i]
        keypoints, scores = decode(simcc_x, simcc_y, simcc_split_ratio)

        # 将关键点缩放回原图坐标。
        keypoints = keypoints / model_input_size * scale[i] + center[i] - scale[i] / 2
        all_key.append(keypoints[0])
        all_score.append(scores[0])

    return np.array(all_key), np.array(all_score)


def bbox_xyxy2cs(bbox: np.ndarray,
                 padding: float = 1.) -> tuple[np.ndarray, np.ndarray]:
    """将边界框从 (x1,y1,x2,y2) 转换为 (中心, 尺寸)。

    Args:
        bbox: 形状为 (4,) 或 (n, 4) 的边界框。
        padding: 尺寸扩展比例，默认值为 1.0。

    Returns:
        边界框中心和尺寸数组。
    """
    # 将单个边界框转换为批量形状。
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    # 计算边界框中心和尺寸。
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale


def _fix_aspect_ratio(bbox_scale: np.ndarray,
                      aspect_ratio: float) -> np.ndarray:
    """扩展尺寸以匹配指定的宽高比。

    Args:
        bbox_scale: 图像尺寸 (宽, 高)。
        aspect_ratio: 宽高比。

    Returns:
        调整后的图像尺寸。
    """
    w, h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(w > h * aspect_ratio,
                          np.hstack([w, w / aspect_ratio]),
                          np.hstack([h * aspect_ratio, h]))
    return bbox_scale


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """按弧度旋转二维点。

    Args:
        pt: 二维点坐标 (x, y)。
        angle_rad: 旋转角度，单位为弧度。

    Returns:
        旋转后的二维点。
    """
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """根据两个二维点计算仿射变换所需的第三个点。

    Args:
        a: 第一个二维点。
        b: 第二个二维点。

    Returns:
        第三个二维点。
    """
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c


def get_warp_matrix(center: np.ndarray,
                    scale: np.ndarray,
                    rot: float,
                    output_size: tuple[int, int],
                    shift: tuple[float, float] = (0., 0.),
                    inv: bool = False) -> np.ndarray:
    """计算将边界框区域变换到目标尺寸的仿射矩阵。

    Args:
        center: 边界框中心坐标。
        scale: 边界框的宽度和高度。
        rot: 旋转角度，单位为度。
        output_size: 目标输出尺寸。
        shift: 相对于宽度和高度的平移比例。
        inv: 是否计算从目标到源图像的逆变换。

    Returns:
        形状为 ``(2, 3)`` 的仿射变换矩阵。
    """
    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    # 计算仿射变换矩阵。
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0., src_w * -0.5]), rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    # 获取原图中源矩形的角点。
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    # 获取模型输入中目标矩形的角点。
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return warp_mat


def top_down_affine(input_size: tuple[int, int], bbox_scale: np.ndarray, bbox_center: np.ndarray,
                    img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """通过仿射变换提取边界框区域作为模型输入。

    Args:
        input_size: 模型输入尺寸。
        bbox_scale: 图像边界框尺寸。
        bbox_center: 图像边界框中心。
        img: 原始图像。

    Returns:
        仿射变换后的图像和边界框尺寸。
    """
    w, h = input_size
    warp_size = (int(w), int(h))

    # 将边界框调整到固定宽高比。
    bbox_scale = _fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)

    # 计算仿射矩阵。
    center = bbox_center
    scale = bbox_scale
    rot = 0
    warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

    # 执行仿射变换。
    img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

    return img, bbox_scale


def get_simcc_maximum(simcc_x: np.ndarray,
                      simcc_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """从 SimCC 表示中获取最大响应位置和数值。

    Args:
        simcc_x: 形状为 ``(N, K, Wx)`` 的横轴 SimCC 表示。
        simcc_y: 形状为 ``(N, K, Wy)`` 的纵轴 SimCC 表示。

    Returns:
        最大响应坐标和对应响应值。
    """
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    # 获取最大值位置。
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    # 获取 x、y 方向的共同最大响应。
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1

    # 恢复输出形状。
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals


def decode(simcc_x: np.ndarray, simcc_y: np.ndarray,
           simcc_split_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """根据 SimCC 分割比例解码关键点坐标。

    Args:
        simcc_x: 模型预测的 x 方向 SimCC。
        simcc_y: 模型预测的 y 方向 SimCC。
        simcc_split_ratio: SimCC 分割比例。

    Returns:
        关键点坐标和置信度。
    """
    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints /= simcc_split_ratio

    return keypoints, scores


def inference_pose(
    session: ort.InferenceSession, out_bbox: np.ndarray, ori_img: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """执行姿态模型推理并返回关键点和置信度。

    Args:
        session: 姿态估计 ONNX Runtime 会话。
        out_bbox: 人体检测边界框。
        ori_img: BGR 格式的输入图像。

    Returns:
        姿态关键点坐标和对应置信度。
    """
    h, w = session.get_inputs()[0].shape[2:]
    model_input_size = (w, h)
    resized_img, center, scale = preprocess(ori_img, out_bbox, model_input_size)
    outputs = inference(session, resized_img)
    keypoints, scores = postprocess(outputs, model_input_size, center, scale)

    return keypoints, scores
