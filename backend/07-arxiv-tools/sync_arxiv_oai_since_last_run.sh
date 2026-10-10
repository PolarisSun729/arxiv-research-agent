#!/usr/bin/env bash
# arXiv OAI-PMH 增量同步入口（Linux 生产环境）。
#
# 调用方式:
#   由 deploy/arxiv-oai-sync.service 执行，deploy/arxiv-oai-sync.timer 每天定时触发；
#   手动执行：sudo systemctl start arxiv-oai-sync.service，日志：journalctl -u arxiv-oai-sync
#
# 工作流程:
#   1. 从游标文件读出“上次成功同步到哪一天”（last_until）；
#   2. 本次同步区间 = [last_until + 1 天, 昨天]，只同步已经完整结束的日子，不碰今天；
#   3. 调用 sync_arxiv_oai.py 抓取该区间的论文并写库，同时更新首页读取的摘要文件；
#   4. 只有同步成功（退出码 0）才把游标推进到“昨天”，失败时游标不动，下次会重试同一区间。
#   因为是“补齐到昨天”，中间停了几天也会在下次运行时一次补齐。
#
# 可通过环境变量覆盖的参数（未设置时用下面的默认值）:
#   PYTHON_EXE        Python 解释器，默认 release 根目录下的 .venv/bin/python
#   INTERVAL_SECONDS  两次请求之间的间隔秒数，默认 5（服务层最低 3.5）
#   TIMEOUT_SECONDS   单次 HTTP 请求超时秒数，默认 60
#   MAX_RETRIES       每页请求的最大尝试次数，默认 5
#   LOG_LEVEL         日志级别，默认 INFO
#
# 依赖 GNU date（-d 参数），只能在 Linux 上运行；Windows 本地请用同目录的 .cmd 版本。

# -e：任一命令失败就退出（同步失败时不会执行最后的游标推进）；
# -u：引用未定义变量即报错；-o pipefail：管道中任一环节失败都算失败。
set -euo pipefail

# Linux 生产增量入口：脚本位于 release/current/backend/07-arxiv-tools，
# 但游标和摘要必须写入 backend/data（部署器会将其链接到 shared/backend/data）。
# 这样每次发布切换 release 目录后，游标和摘要都不会丢。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
STATE_DIR="$BACKEND_DIR/data/arxiv-oai-sync"
# 游标文件：只有一行 YYYY-MM-DD，表示已成功同步到的最后一天（含当天）。
STATE_FILE="$STATE_DIR/sync_arxiv_oai_since_last_run.state"
# 摘要文件：每次运行的结果（成功/失败、新增篇数等），由后端 /sync-status 接口读取并展示在首页。
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

# %F 等价于 %Y-%m-%d。“昨天”是本次同步的终点，“前天”用于游标缺失时的默认值。
yesterday="$(date -d 'yesterday' +%F)"
two_days_ago="$(date -d '2 days ago' +%F)"
last_until=""
if [[ -f "$STATE_FILE" ]]; then
    # 文件没有结尾换行时 read 会返回非 0，用 || true 防止 set -e 直接退出。
    IFS= read -r last_until < "$STATE_FILE" || true
fi
# 游标不存在、格式不对或不是合法日期（例如 2026-02-30）时，按“已同步到前天”处理，
# 也就是首次运行只同步昨天一天，不会意外回溯抓取大量历史数据。需要补历史请手动运行 sync_arxiv_oai.py。
if [[ ! "$last_until" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || ! date -d "$last_until" +%F >/dev/null 2>&1; then
    last_until="$two_days_ago"
fi
from_date="$(date -d "$last_until + 1 day" +%F)"
# YYYY-MM-DD 格式的字符串比较等价于日期比较；起点晚于昨天说明已经是最新，无需同步。
if [[ "$from_date" > "$yesterday" ]]; then
    echo "No new full day to sync; last successful until: $last_until"
    exit 0
fi

echo "Running incremental arXiv OAI-PMH sync from $from_date to $yesterday"
# 同步失败（退出码非 0）时，set -e 会让脚本在这里退出，后面的游标推进不会执行，
# systemd 也会把这次运行记为失败。
"$PYTHON_EXE" "$SCRIPT_DIR/sync_arxiv_oai.py" \
    --from "$from_date" --until "$yesterday" --meta-file "$META_FILE" \
    --interval "$INTERVAL_SECONDS" --timeout "$TIMEOUT_SECONDS" \
    --max-retries "$MAX_RETRIES" --log-level "$LOG_LEVEL"

# 仅在同步成功后推进游标，并通过 rename 避免并发读取半行日期。
# 临时文件名带上当前进程号 $$，避免多个实例同时运行时互相覆盖临时文件。
temporary="$STATE_FILE.tmp.$$"
printf '%s\n' "$yesterday" > "$temporary"
mv -f -- "$temporary" "$STATE_FILE"
