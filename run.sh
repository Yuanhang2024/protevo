#!/usr/bin/env sh
# protevo 一键启动（Linux / macOS）：自动创建 venv、安装依赖、进入命令行交互台
# 用法：sh run.sh   （或 chmod +x run.sh && ./run.sh）
cd "$(dirname "$0")" || exit 1

PY="${PROTEVO_PY:-}"
if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)' >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1 && python -c 'import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)' >/dev/null 2>&1; then
  PY=python
fi
if [ -z "$PY" ]; then
  echo "[错误] 未找到 Python 3.10+，请先安装"
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "[1/3] 创建虚拟环境 .venv ..."
  "$PY" -m venv .venv || exit 1
fi
VPY=.venv/bin/python

if ! "$VPY" -c 'import protevo, pandas, sklearn, scipy' >/dev/null 2>&1; then
  echo "[2/3] 安装依赖（首次约几分钟，后续跳过）..."
  "$VPY" -m pip install -q -r requirements.txt || { echo "[错误] 依赖安装失败，请检查网络"; exit 1; }
else
  echo "[2/3] 依赖已就绪，跳过安装"
fi

echo "[3/3] 启动 protevo 交互台（quit 退出；help 查看命令）..."
exec "$VPY" -m protevo.tui
