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

## 许可证

本仓库**不是**自研实现，`funmodel/dwpose/core/` 下的 5 个模块全部取自
[IDEA-Research/DWPose](https://github.com/IDEA-Research/DWPose)（Apache-2.0）。
因此本仓库沿用上游协议，以 **[Apache License 2.0](LICENSE)** 开源，而不是组织默认的 MIT；
上游署名见 [NOTICE](NOTICE)。

## 第三方代码来源

`funmodel/dwpose/core/` 下的检测、姿态估计与绘制代码取自：

- [IDEA-Research/DWPose](https://github.com/IDEA-Research/DWPose)（本仓库的直接上游，提供全部 ONNX 推理与绘制代码，Apache-2.0）
- [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose)（`onnxpose.py` 的 SimCC 解码与仿射变换工具，Apache-2.0）
- [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)（`onnxdet.py` 的 NMS 与检测后处理，Apache-2.0）
- [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)（`util.py` 的 `faceDetect` 与面部/手部绘制，Apache-2.0）
- [Hzzone/pytorch-openpose](https://github.com/Hzzone/pytorch-openpose)（`util.py` 的 `transfer()`，该仓库**未提供任何 LICENSE 文件**）
- [CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)（`util.py` 的 `handDetect` 与骨架/调色板常量，**许可证为学术/非营利非商业研究专用**，GitHub 归类为 `NOASSERTION`，既非 Apache-2.0 也非 MIT）

前四项为 Apache-2.0，其署名义务已通过 `LICENSE`、`NOTICE` 与逐文件版权头履行。
后两项的许可证状况**无法通过署名满足**：pytorch-openpose 未声明许可证，CMU OpenPose
的协议禁止分发且不授予再授权权。该部分的处置（含 PyPI 上已发布版本）在
[farfarfun/todo-list#894](https://github.com/farfarfun/todo-list/issues/894) 跟踪，
**尚未结论**。逐项实测证据与许可证原文引用见
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)，引用方请据此自行评估。

---

## 关于 farfarfun

[farfarfun](https://github.com/farfarfun) 是一个专注于实用工具库的开源组织，
涵盖云存储、数据处理、AI、多媒体与开发工具链等方向。

- 🏠 组织主页：<https://github.com/farfarfun>
- 📦 PyPI：<https://pypi.org/user/niuliangtao/>
- 📧 联系：farfarfun@qq.com

本项目基于 [Apache-2.0](LICENSE) 协议开源（衍生自第三方 Apache-2.0 代码，沿用上游协议，
非组织默认的 MIT；详见上方「许可证」与 [NOTICE](NOTICE)）。
