# funmodel-dwpose

基于 [DWPose](https://github.com/IDEA-Research/DWPose)（含 CMU OpenPose / pytorch-openpose 移植代码）封装的人体全身姿态估计模型，作为 `funmodel` 命名空间下的一个子包发布（`import funmodel.dwpose`），提供开箱即用的关键点检测与骨架绘制能力。

## 安装

```bash
pip install funmodel-dwpose
# 或
uv add funmodel-dwpose
```

## 最小示例

```python
import cv2

from funmodel.dwpose import DWposePredict

predictor = DWposePredict()
image = cv2.imread("input.jpg")
result, drawn_image = predictor.predict(image, draw=True)
cv2.imwrite("output.jpg", drawn_image)
```

首次调用 `DWposePredict()` 会自动从对象存储下载 ONNX 权重（`yolox_l.onnx`、`dw-ll_ucoco_384.onnx`）到本地缓存目录（`~/.funmodel/<model_name>/`）。

## 第三方代码来源

`funmodel/dwpose/core/` 下的检测、姿态估计与绘制代码移植自：

- [CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)（原始实现）
- [Hzzone/pytorch-openpose](https://github.com/Hzzone/pytorch-openpose)（二次移植）
- [IDEA-Research/DWPose](https://github.com/IDEA-Research/DWPose)（ONNX 推理管线）

以上上游项目均为 MIT/兼容协议，移植代码保留原始来源说明，与本仓库统一的 MIT 协议兼容。

---

## 关于 farfarfun

[farfarfun](https://github.com/farfarfun) 是一个专注于实用工具库的开源组织，
涵盖云存储、数据处理、AI、多媒体与开发工具链等方向。

- 🏠 组织主页：<https://github.com/farfarfun>
- 📦 PyPI：<https://pypi.org/user/niuliangtao/>
- 📧 联系：farfarfun@qq.com

本项目基于 [MIT](LICENSE) 协议开源。
