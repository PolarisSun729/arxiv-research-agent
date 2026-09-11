#!/usr/bin/env bash
# 复用真实 ASGI 安全回归；隔离凭据和数据库由 Python 入口统一负责。
set -euo pipefail

# 只用 Bash 内建命令解析目录，兼容从 PowerShell 直接调用的 Git Bash。
SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
if [[ "$SCRIPT_DIR" == "${BASH_SOURCE[0]}" ]]; then
    SCRIPT_DIR=.
fi
SCRIPT_DIR="$(cd -- "$SCRIPT_DIR" && pwd)"
if [[ -n "${SECURITY_TEST_PYTHON:-}" ]]; then
    PYTHON_BIN="$SECURITY_TEST_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
else
    PYTHON_BIN=python
fi

exec "$PYTHON_BIN" -X utf8 "$SCRIPT_DIR/scripts/test_security.py" "$@"
