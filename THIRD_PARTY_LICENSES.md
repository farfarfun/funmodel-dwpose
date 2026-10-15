# 第三方代码许可证说明

`funmodel/dwpose/core/` 下的 5 个模块全部来自上游，不是在上游启发下自行实现的。
本文件记录**实测得到的**溯源证据与各上游的**实际许可证**，供引用方自行评估合规风险。

每条结论的核实方式：上游许可证原文用 `curl` 从各仓库 raw 地址下载后逐条比对；
代码用 `sha256sum` 逐文件比对，并在剔除注释与 docstring 后做 token 级比对；
已发布产物从 PyPI 下载 wheel 后 `unzip` 展开比对。

## 一、溯源证据

本仓库直接的上游是 **IDEA-Research/DWPose**（分支 `onnx`）的
`ControlNet-v1-1-nightly/annotator/dwpose/` 目录。

| 本仓库文件 | 上游对应文件 | 实测结果 |
|---|---|---|
| `core/__init__.py` | `annotator/dwpose/__init__.py` | 已发布的 **0.0.11 与上游逐字节相同**（sha256 `a0925acf…`，1983 字节）。当前 master 增加了中文 docstring、类型标注，并移除 subset 索引重排逻辑 |
| `core/onnxdet.py` | `annotator/dwpose/onnxdet.py` | 已发布的 **0.0.11 与上游逐字节相同**（sha256 `4c82a928…`，4331 字节）。当前 master 仅增加中文 docstring、类型标注与格式化，逻辑未变（去注释后 token 级相似度约 96%） |
| `core/onnxpose.py` | `annotator/dwpose/onnxpose.py` | 已发布的 **0.0.11 与上游逐字节相同**（sha256 `16fb69ab…`，11608 字节）。当前 master 同上（去注释后约 94%） |
| `core/util.py` | `annotator/dwpose/util.py` | **全部 10 个函数均存在于上游，无一个是自研**；差异仅为中文 docstring、类型标注与 black 格式化（去注释后 token 级相似度约 91%） |
| `core/wholebody.py` | `annotator/dwpose/wholebody.py` | 去注释后 token 级相似度约 89%；差异为自加的 CUDA ExecutionProvider 选择 |
| `predictor.py` | — | 封装层（下载、推理入口、结果整形）为自研，但关键点切片常量沿用 OpenPose 顺序，且 `draw_image()` 直接调用 `util.draw_bodypose/draw_handpose/draw_facepose` |

DWPose 之上还有两条可追到具体文件的链路：

- `core/onnxdet.py` 的 `nms` / `multiclass_nms` / `demo_postprocess` 来自
  **Megvii YOLOX** 的 `yolox/utils/demo_utils.py`。实测 DWPose 版本的 `nms` 与
  `demo_postprocess` 在归一化比对后与 YOLOX **完全一致**。
- `core/onnxpose.py` 的 `get_simcc_maximum` / `bbox_xyxy2cs` / `get_warp_matrix` /
  `top_down_affine` / `_fix_aspect_ratio` / `_rotate_point` / `_get_3rd_point` 来自
  **OpenMMLab MMPose**。DWPose 自己的 LICENSE 文件末尾也追加了 MMPose 的
  Apache-2.0 声明。

### CMU OpenPose 的直接证据

`core/util.py` 的 `handDetect` 是 CMU OpenPose
`src/openpose/hand/handDetector.cpp` 的 Python 逐行转写，**连该 C++ 源码的注释
都一起搬了过来**。实测对照（CMU 行号取自上游 `master`）：

| 本仓库 `core/util.py` | CMU `handDetector.cpp` |
|---|---|
| `ratioWristElbow = 0.33` | 第 20 行 `const auto ratioWristElbow = 0.33f;` |
| 注释 `# handRectangle.x = posePtr[wrist*3] + ratioWristElbow * (…)` | 第 25 行原文 |
| 注释 `# handRectangle.y = posePtr[wrist*3+1] + ratioWristElbow * (…)` | 第 26 行原文 |
| 注释 `# const auto distanceWristElbow = getDistance(…)` | 第 27 行原文 |
| 注释 `# const auto distanceElbowShoulder = getDistance(…)` | 第 28 行原文 |
| 注释 `# handRectangle.width = 1.5f * fastMax(distanceWristElbow, 0.9f * distanceElbowShoulder);` | 第 29 行原文 |
| 注释 `# handRectangle.x -= handRectangle.width / 2.f;` | 第 34 行原文 |
| 注释 `# handRectangle.y -= handRectangle.height / 2.f;` | 第 35 行原文 |

此外 `draw_bodypose` 的 19 组 `limbSeq` 肢体连接表与 18 色调色板就是 OpenPose
COCO-18 骨架常量，`draw_handpose` 的 21 点手部骨架边表就是 OpenPose hand model。

本文件此前的措辞是「在算法结构上可追溯至 CMU OpenPose 的原始设计」。
**该措辞弱于实际证据**：实际情况是 C++ 源码的逐行转写外加逐字注释拷贝，
而不只是算法谱系上的近似。本次已据实改写。

## 二、各上游的实际许可证

| 上游项目 | 实际许可证（实测） | 署名义务是否已履行 |
|---|---|---|
| [IDEA-Research/DWPose](https://github.com/IDEA-Research/DWPose)（本仓库的直接上游） | **Apache-2.0**，附录已填 `Copyright 2023 IDEA. All Rights Reserved.` | 已在 `LICENSE` / `NOTICE` / 逐文件头部履行 |
| [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) | **Apache-2.0**，`Copyright 2018-2020 Open-MMLab.`，逐文件头部 `# Copyright (c) OpenMMLab. All rights reserved.` | 同上 |
| [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | **Apache-2.0**，附录已填 `Copyright (c) 2021-2022 Megvii Inc. All rights reserved.` | 同上 |
| [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) | **Apache-2.0**（其 LICENSE 附录未填写，仍为 `Copyright [yyyy] [name of copyright owner]`） | 同上 |
| [Hzzone/pytorch-openpose](https://github.com/Hzzone/pytorch-openpose) | **无 LICENSE 文件**（GitHub API `license: null`）→ 默认保留全部权利，未授予使用/修改/再分发权 | 无法通过署名满足，见第四节 |
| [CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) | **非标准开源协议**，见下 | 无法通过署名满足，见第四节 |

### CMU OpenPose 许可证条款

来源 `https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/LICENSE`
（实际下载核对，9686 字节）。标题三行：

```
OPENPOSE: MULTIPERSON KEYPOINT DETECTION
SOFTWARE LICENSE AGREEMENT
ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
```

关键条款（逐字引用）：

- **GRANT OF LICENSE**：授予的是 `noncommercial research purposes, without the
  right to sublicense` 的非独占、不可转让许可；"Software" 的定义显式包含
  `all or any derivatives and/or modifications created or made by You`。
- **DERIVATIVES**：`You agree that all and any such derivatives and
  modifications will be owned by Licensor`，且 `you may not otherwise use,
  distribute or copy such derivatives and modifications`。
- **USES NOT PERMITTED**：不得 `distribute, copy or use` 除本协议明确允许之外；
  不得使用 "OpenPose" / "Carnegie Mellon" 名称或标识。
- **TERM**：违约即自动终止，终止后须 `delete any and all copies`。
- **GOVERNING LAW**：宾夕法尼亚州，Allegheny County 管辖。

GitHub API 将该许可证归类为 `spdx_id: "NOASSERTION"` / `name: "Other"`，
即不被识别为任何标准开源协议。

商业授权渠道在 OpenPose README 中给出（不在 LICENSE 文件内）：FlintBox 页面
`https://cmu.flintbox.com/`，并需抄送 Yaser Sheikh `yaser@cs.cmu.edu`。

## 三、本仓库自身的许可证沿革

- 本仓库原为 **Apache-2.0**。`d939bea`（2026-08-26，提交信息「按 SPEC.md §12.4，
  全组织统一 MIT。原为 Apache-2.0。」）把 `LICENSE` 换成了 MIT，**该动作删掉了
  仓库内唯一一份上游 Apache-2.0 文本**，使 DWPose / MMPose / YOLOX / ControlNet
  的 Apache-2.0 §4 署名义务处于未履行状态。本次恢复 Apache-2.0 即为补上这一项。
- 实测发现：`d939bea` 删掉的那份 Apache-2.0 文本（11357 字节，
  sha256 `c71d239d…`）与 **ControlNet 的 LICENSE 文件逐字节相同**，也与
  PyPI 上 `0.0.11` 的 `dist-info/LICENSE` 逐字节相同——三者都是**附录未填写的
  空白样板**（`Copyright [yyyy] [name of copyright owner]`）。也就是说，
  已公开分发两年半的 `0.0.11` 对 CMU / IDEA / Open-MMLab / Megvii / ControlNet
  **零署名**。
- 本次恢复的 `LICENSE` 采用 `https://www.apache.org/licenses/LICENSE-2.0.txt`
  的正文（逐字节核对），并把附录填写为 `Copyright 2024 farfarfun`，
  **不沿用那份空白样板**。上游署名集中记录在 `NOTICE`。

## 四、尚未解决的问题

以下两项**不是**署名问题，无法通过 `NOTICE` 与逐文件版权头解决，仍在
[farfarfun/todo-list#894](https://github.com/farfarfun/todo-list/issues/894)
跟踪，尚无结论：

1. **CMU OpenPose 的分发权**。其许可证禁止分发、不授予再授权权、并约定衍生物
   所有权归 Licensor；而「把 C++ 转写成 Python 是否构成该协议下的 derivative」
   是法律判断，不是可查证的事实。本文件只给出第一节的逐字对照证据，不下结论。
2. **pytorch-openpose 无许可证**。`util.py` 的 `transfer()` 来自该上游，是
   caffe→pytorch 的权重改名函数；本包只走 ONNX 推理路径，该函数在包内无任何
   调用点，属死代码。

此外还有一个**与代码许可证完全独立、本次未审计**的合规面：`predictor.py`
从 farfarfun 自有对象存储重分发 ONNX 权重（`yolox_l.onnx`、
`dw-ll_ucoco_384.onnx`）。模型权重的再分发条款（DWPose 权重涉及
COCO-WholeBody 数据集条款，YOLOX 权重来自 Megvii）需另行核实。
