# Changelog

## 0.0.12

### 新增

- 新增 `NOTICE`，按 Apache License 2.0 第 4(d) 条列出 DWPose（IDEA）、MMPose
  （Open-MMLab）、YOLOX（Megvii）、ControlNet 的版权声明与出处，并一并记录
  pytorch-openpose（无 LICENSE）与 CMU OpenPose（非商业研究专用）两条非
  Apache-2.0 链路。`NOTICE` 已加入 `license-files`，随发行物一起分发。
- `funmodel/dwpose/` 下全部 Python 文件新增逐文件版权头与
  `SPDX-License-Identifier: Apache-2.0`，并注明各文件的上游出处与本仓库的改动
  范围；`util.py` 的 `handDetect` 处单独标注其为 CMU `handDetector.cpp` 的
  Python 转写、且保留了该 C++ 源码的逐字注释。

### 修复

- `example/example.py` 的摄像头示例原先在模块顶层无条件执行，`import` 该模块即会触发摄像头访问；
  现收敛到 `if __name__ == "__main__":`，消除导入期副作用。
- `script/build.sh` 原先在 `funbuild build --multi` 失败后无条件用裸 `pip install` 安装
  `funpypi`/`funbuild` 并重试，掩盖真实失败原因；改为检测 `funbuild` 是否已安装、`set -euo pipefail`
  使失败原样非零退出，不再做隐式兜底安装。

### 变更

- 完善姿态推理接口的类型标注与输入边界测试。
- 将 `funget` 依赖下限提升至 `1.1.63`，与组织正式入口保持一致。
- 核实第三方代码的实际许可证（新增 `THIRD_PARTY_LICENSES.md`），修正 README 中「上游均为
  MIT/兼容协议」的不准确表述；CMU OpenPose 实为非商业研究许可、pytorch-openpose 未声明许可证，
  是否继续以 MIT 分发可追溯至 CMU OpenPose 的绘制/检测实现待仓库所有者决策。
- 删除未被任何构建流程引用的残留版本文件 `script/__version__.md`（版本号唯一来源为
  `pyproject.toml`）。
- **开源协议由 MIT 改回 Apache-2.0**（`LICENSE`、`pyproject.toml`、README 同步）。
  `d939bea` 为执行 SPEC.md §12.4「全组织统一 MIT」把 `LICENSE` 换成 MIT，该动作删掉了
  仓库内唯一一份上游 Apache-2.0 文本，使 DWPose / MMPose / YOLOX / ControlNet 的
  Apache-2.0 §4 署名义务处于未履行状态。本仓库的 `core/` 代码全部取自上游，按「衍生自
  第三方代码的沿用上游协议」处理，故恢复 Apache-2.0。恢复的文本取自
  `https://www.apache.org/licenses/LICENSE-2.0.txt` 并填写附录版权行，不沿用
  `d939bea` 删除的那份附录空白样板（经实测该样板与 ControlNet 的 LICENSE 及 PyPI 上
  `0.0.11` 的 `dist-info/LICENSE` 逐字节相同）。
- 改写 `THIRD_PARTY_LICENSES.md`：原措辞「在算法结构上可追溯至 CMU OpenPose」弱于实际
  证据，改为据实表述——已发布的 `0.0.11` 中 `core/__init__.py`、`onnxdet.py`、
  `onnxpose.py` 与 DWPose 上游**逐字节相同**，`util.py` 的 10 个函数全部来自上游，
  `handDetect` 内含 CMU `handDetector.cpp` 的**逐字 C++ 源码注释**。同时补上原先遗漏的
  MMPose 与 YOLOX 两条链路，并说明各条结论的实测核实方式。
- README 的协议声明与 `LICENSE` 对齐（原先声明 MIT 与实际文件不符）。

### 未解决

- CMU OpenPose 的分发权问题（其协议禁止分发、不授予再授权权）与 pytorch-openpose 无
  许可证问题**不是署名问题，无法通过 `NOTICE` 解决**，需法律判断，在
  [farfarfun/todo-list#894](https://github.com/farfarfun/todo-list/issues/894) 跟踪。
- `predictor.py` 从自有对象存储重分发 ONNX 权重，属与代码许可证独立的合规面，本次未审计。

### 废弃

- 无

> 0.0.12 之前的版本未维护 CHANGELOG，历史变更请参考 `git log`。
