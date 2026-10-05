# Changelog

## 0.0.12

### 新增

- 无

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

### 废弃

- 无

> 0.0.12 之前的版本未维护 CHANGELOG，历史变更请参考 `git log`。
