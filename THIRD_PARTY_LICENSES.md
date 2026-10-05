# 第三方代码许可证说明

`funmodel/dwpose/core/` 下的检测、姿态估计与绘制代码按 `core/__init__.py` 开头注释记录的移植链路，
依次来自以下上游项目。本文件逐项核实其**实际许可证**（而不是笼统假设为 MIT），供引用方评估合规风险。

| 上游项目 | 实际许可证 | 与 MIT 的关系 |
|---|---|---|
| [CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)（原始实现，身体/手部关键点算法与绘制逻辑的最初来源） | **OpenPose: Multiperson Keypoint Detection Software License Agreement**——仅限学术或非营利机构的非商业研究用途；Carnegie Mellon University 保留所有权，不授予再许可权；商业使用需向 CMU 另行购买商用许可。 | **不兼容**。这不是 MIT 或 MIT 兼容协议，禁止未经许可的商业使用/再分发。 |
| [Hzzone/pytorch-openpose](https://github.com/Hzzone/pytorch-openpose)（将 CMU 算法用 PyTorch 重新实现，二次移植） | 该仓库**未提供任何 LICENSE 文件**（GitHub API 查询 `license: null`）。 | 默认版权保留，未明确授予使用/修改/再分发权利。 |
| [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)（在 pytorch-openpose 基础上补充面部/手部检测，三/四次编辑） | Apache License 2.0。 | 与 MIT 相容（均为宽松许可），但需保留 Apache-2.0 的版权与 NOTICE 声明。 |
| [IDEA-Research/DWPose](https://github.com/IDEA-Research/DWPose)（提供本仓库实际使用的 ONNX 推理管线与权重导出流程） | Apache License 2.0。 | 与 MIT 相容，需保留版权与 NOTICE 声明。 |

## 结论

- `core/onnxdet.py`、`core/onnxpose.py`、`core/wholebody.py` 的推理管线主要来自 DWPose/ControlNet
  的 Apache-2.0 代码路径，与本仓库 MIT 协议兼容。
- `core/util.py` 中的绘制逻辑（`draw_bodypose`/`draw_handpose`/`draw_facepose`/`handDetect`）在算法结构上
  可追溯至 CMU OpenPose 的原始设计（`handDetect` 函数注释直接引用了 CMU 源码路径），但本仓库实际持有的是
  pytorch-openpose/ControlNet 的 PyTorch 重新实现，并非 CMU 的原始源码或预训练权重。
- 即便如此，由于 pytorch-openpose 未提供任何明确许可证、且该实现在算法谱系上直接衍生自 CMU 的
  非商业研究许可代码，**能否以 MIT 协议对外商用分发仍存在未消除的合规风险**，不能简单断言「上游均为
  MIT/兼容协议」。
- 是否需要替换 `core/util.py` 中可追溯至 CMU OpenPose 的绘制/检测实现、申请 CMU 商用许可，或改为明确
  声明仅限非商业使用，属于需要仓库所有者权衡的业务/法务决策，本次审计仅做事实核实与文档更正，未改动
  任何推理/绘制代码逻辑。
