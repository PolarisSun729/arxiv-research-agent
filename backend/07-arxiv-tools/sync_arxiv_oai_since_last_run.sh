#!/usr/bin/env bash
# arXiv OAI-PMH 增量同步入口（Linux 生产环境）。
#
# 调用方式:
#   由 deploy/arxiv-oai-sync.service 执行，deploy/arxiv-oai-sync.timer 每天定时触发；
#   手动执行：sudo systemctl start arxiv-oai-sync.service，日志：journalctl -u arxiv-oai-sync
#
# 工作流程（日期计算、游标读写和断点续传都在 sync_arxiv_oai.py 里完成）:
#   1. 从游标的下一天同步到昨天，只同步已经完整结束的日子，不碰今天；
#   2. 按天逐个同步，每天完成后立即推进游标；中断或某天失败时游标停在最后一个成功日，
#      下次运行从失败的那天继续；中间停了几天也会在下次运行时一次补齐；
#   3. 游标缺失时以 OAI 库中最大 datestamp 那天为起点；没有记录的日期（noRecordsMatch）按成功处理；
#   4. 运行中和结束时都会更新首页读取的摘要文件。
#
# 可通过环境变量覆盖的参数（未设置时用下面的默认值）:
#   PYTHON_EXE        Python 解释器，默认 release 根目录下的 .venv/bin/python
#   INTERVAL_SECONDS  两次请求之间的间隔秒数，默认 5（服务层最低 3.5）
#   TIMEOUT_SECONDS   单次 HTTP 请求超时秒数，默认 60
#   MAX_RETRIES       每页请求的最大尝试次数，默认 5
#   LOG_LEVEL         日志级别，默认 INFO

set -euo pipefail

# Linux 生产增量入口：脚本位于 release/current/backend/07-arxiv-tools，
# 但游标和摘要必须写入 backend/data（部署器会将其链接到 shared/backend/data）。
# 这样每次发布切换 release 目录后，游标和摘要都不会丢。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
STATE_DIR="$BACKEND_DIR/data/arxiv-oai-sync"
# 游标文件：只有一行 YYYY-MM-DD，表示已成功同步到的最后一天（含当天）。
STATE_FILE="$STATE_DIR/sync_arxiv_oai_since_last_run.state"
# 摘要文件：每次运行的进度和结果，由后端 /sync-status 接口读取并展示在首页。
META_FILE="$STATE_DIR/sync_arxiv_oai_since_last_run.meta.json"
# 发布布局把 venv 链接放在 release 根目录（backend 的上一级）。
PYTHON_EXE="${PYTHON_EXE:-$BACKEND_DIR/../.venv/bin/python}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-5}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-60}"
MAX_RETRIES="${MAX_RETRIES:-5}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

if [[ ! -x "$PYTHON_EXE" ]]; then
    echo "Python executable not found or not executable: $PYTHON_EXE" >&2
    exit 1
fi

# exec 让 Python 直接接管进程：systemd 停止服务时信号直达同步进程，退出码也原样交给 systemd。
exec "$PYTHON_EXE" "$SCRIPT_DIR/sync_arxiv_oai.py" \
    --state-file "$STATE_FILE" --meta-file "$META_FILE" \
    --interval "$INTERVAL_SECONDS" --timeout "$TIMEOUT_SECONDS" \
    --max-retries "$MAX_RETRIES" --log-level "$LOG_LEVEL"
