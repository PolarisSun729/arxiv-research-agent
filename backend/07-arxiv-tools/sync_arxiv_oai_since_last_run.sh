#!/usr/bin/env bash
set -euo pipefail

# Linux 生产增量入口：脚本位于 release/current/backend/07-arxiv-tools，
# 但游标和摘要必须写入 backend/data（部署器会将其链接到 shared/backend/data）。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
STATE_DIR="$BACKEND_DIR/data/arxiv-oai-sync"
STATE_FILE="$STATE_DIR/sync_arxiv_oai_since_last_run.state"
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
mkdir -p "$STATE_DIR"

yesterday="$(date -d 'yesterday' +%F)"
two_days_ago="$(date -d '2 days ago' +%F)"
last_until=""
if [[ -f "$STATE_FILE" ]]; then
    IFS= read -r last_until < "$STATE_FILE" || true
fi
if [[ ! "$last_until" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || ! date -d "$last_until" +%F >/dev/null 2>&1; then
    last_until="$two_days_ago"
fi
from_date="$(date -d "$last_until + 1 day" +%F)"
if [[ "$from_date" > "$yesterday" ]]; then
    echo "No new full day to sync; last successful until: $last_until"
    exit 0
fi

echo "Running incremental arXiv OAI-PMH sync from $from_date to $yesterday"
"$PYTHON_EXE" "$SCRIPT_DIR/sync_arxiv_oai.py" \
    --from "$from_date" --until "$yesterday" --meta-file "$META_FILE" \
    --interval "$INTERVAL_SECONDS" --timeout "$TIMEOUT_SECONDS" \
    --max-retries "$MAX_RETRIES" --log-level "$LOG_LEVEL"

# 仅在同步成功后推进游标，并通过 rename 避免并发读取半行日期。
temporary="$STATE_FILE.tmp.$$"
printf '%s\n' "$yesterday" > "$temporary"
mv -f -- "$temporary" "$STATE_FILE"
