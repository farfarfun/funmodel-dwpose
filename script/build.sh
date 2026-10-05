#!/usr/bin/env bash
set -euo pipefail

if ! command -v funbuild >/dev/null 2>&1; then
    echo "错误：未找到 funbuild，请先通过 uv 安装（例如 uv tool install funbuild 或 uv add --dev funbuild），不自动用 pip 兜底安装。" >&2
    exit 1
fi

funbuild build --multi
