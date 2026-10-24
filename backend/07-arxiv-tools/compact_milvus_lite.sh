#!/usr/bin/env bash
# Milvus Lite 离线整理入口（Linux 生产环境）。
#
# 调用方式:
#   由 deploy/arxiv-milvus-compact.service 以 root 执行，deploy/arxiv-milvus-compact.timer 每天凌晨触发；
#   手动执行：sudo systemctl start arxiv-milvus-compact.service，日志：journalctl -u arxiv-milvus-compact
#
# 工作流程（整理逻辑见 compact_milvus_lite.py）:
#   1. 服务运行中先只读检查（--check）：没有删除行要清、也没有超大文件时直接结束，不停服务；
#   2. 需要整理时停 Web 服务（Milvus Lite 只允许一个进程打开），以应用用户身份整理；
#   3. 无论整理成功与否，只要服务原本在运行就重新启动它。服务原本就停着（例如人工维护中）时不去启动。
#
# 为什么用 root 跑这个外壳：停、启服务需要 root；Python 部分用 runuser 切到应用用户，新文件的属主和服务一致。
#
# 可通过环境变量覆盖的参数（未设置时用下面的默认值）:
#   PYTHON_EXE  Python 解释器，默认 release 根目录下的 .venv/bin/python
#   APP_USER    运行整理脚本的用户，默认 arxiv
#   SERVICE     Web 服务单元名，默认 arxiv-agent.service

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_EXE="${PYTHON_EXE:-$BACKEND_DIR/../.venv/bin/python}"
APP_USER="${APP_USER:-arxiv}"
SERVICE="${SERVICE:-arxiv-agent.service}"
COMPACT_SCRIPT="$SCRIPT_DIR/compact_milvus_lite.py"

if [[ ! -x "$PYTHON_EXE" ]]; then
    echo "Python executable not found or not executable: $PYTHON_EXE" >&2
    exit 1
fi

run_as_app() {
    runuser -u "$APP_USER" -- "$@"
}

was_active=false
if systemctl is-active --quiet "$SERVICE"; then
    was_active=true
fi

if [[ "$was_active" == true ]]; then
    set +e
    run_as_app "$PYTHON_EXE" "$COMPACT_SCRIPT" --check
    check_status=$?
    set -e
    if [[ $check_status -eq 0 ]]; then
        echo "Nothing to compact; $SERVICE keeps running."
        exit 0
    fi
    if [[ $check_status -ne 3 ]]; then
        echo "Compaction check failed with exit code $check_status" >&2
        exit "$check_status"
    fi
    echo "Stopping $SERVICE for compaction."
    systemctl stop "$SERVICE"
    # 整理失败或被中断也要把服务拉起来。
    trap 'echo "Starting $SERVICE."; systemctl start "$SERVICE"' EXIT
fi

run_as_app "$PYTHON_EXE" "$COMPACT_SCRIPT"
