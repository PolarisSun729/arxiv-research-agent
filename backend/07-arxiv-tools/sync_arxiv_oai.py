"""按天从 arXiv OAI-PMH 接口同步论文元数据到本地 SQLite，支持断点续传。

作用:
    把 [起点, 截止日] 拆成逐天的 ListRecords 请求，按目标分类过滤后 upsert 到本地 OAI 库
    （默认 backend/06-database/arxiv_oai.db 的 arxiv_oai_papers 表），写库时同步更新 FTS5 和分类索引。
    只有 ARXIV_DATA_SOURCE=local 时后端才会读这个库。编排逻辑在 services/arxiv/oai_incremental_sync.py。
    落库成功的论文会同时用当前 embedding 配置生成论文向量，写入 Milvus 的 arxiv_paper_embeddings，
    推荐的兴趣簇召回只在这个 collection 里找论文。--dry-run / --count-only 不生成向量。

向量保留窗口:
    只给首次提交日期（created，缺失时用 updated / oai_datestamp）在最近
    ARXIV_OAI_EMBEDDING_RETENTION_MONTHS 个月（按每月 30 天算，默认 6）内的论文生成向量。
    同步成功结束后删除窗口外的论文向量，任何用户点赞或点踩过的论文除外；清理失败只记日志，不影响退出码。

向量失败:
    embedding 或向量库出错只计入摘要里的 embedding_errors，不算当天失败，游标照常推进；
    漏掉的论文用 backfill_arxiv_oai_embeddings.py 补齐（它会跳过已有向量的论文，重复运行也安全）。

断点续传:
    游标默认是 backend/data/arxiv-oai-sync/sync_arxiv_oai_since_last_run.state（--state-file 可改）。
    每天的全部页同步完成后立即把游标写成这一天；中断或失败时游标停在最后一个
    成功日，下次运行从失败的那天继续。某一天失败会立即停止，不会跳过它去同步后面的日期。
    游标缺失时以库内最大 oai_datestamp 那天为起点（那天可能只同步了一部分，重同步是幂等的）。
    arXiv 对没有记录的日期（如周末）返回 noRecordsMatch，按“成功、新增 0 条”处理。

谁来调用:
    - Linux 生产：sync_arxiv_oai_since_last_run.sh（由 deploy/arxiv-oai-sync.timer 每天触发）
    - Windows 本地：sync_arxiv_oai_since_last_run.cmd（双击运行）
    - 手动补数据：直接运行本脚本，指定任意日期区间

常用示例（在 backend 目录下执行）:
    # 增量同步：从游标的下一天同步到昨天（外层脚本就是这样调用的）
    python 07-arxiv-tools/sync_arxiv_oai.py
    # 手动补一段区间：与游标相连时会顺带推进游标，补更早的旧区间则不动游标
    python 07-arxiv-tools/sync_arxiv_oai.py --from 2026-10-01 --until 2026-10-07
    # 演练 / 只计数：完整走一遍抓取和过滤，不写库、不动游标
    python 07-arxiv-tools/sync_arxiv_oai.py --from 2026-10-01 --until 2026-10-07 --dry-run
    python 07-arxiv-tools/sync_arxiv_oai.py --from 2026-10-01 --until 2026-10-07 --count-only

相关配置（环境变量，见 backend/utils/config.py）:
    ARXIV_OAI_ENDPOINT           OAI-PMH 地址，默认 https://oaipmh.arxiv.org/oai
    ARXIV_OAI_TARGET_CATEGORIES  目标分类，逗号分隔，默认 cs.CL,cs.LG,cs.IR,cs.AI；
                                 论文只要挂了其中任一分类（包括 cross-list）就会保留
    OAI_SQLITE_DATABASE_PATH     OAI 库路径，默认 backend/06-database/arxiv_oai.db
    HTTP_PROXY / HTTPS_PROXY     requests 会自动读取，国内网络访问 arXiv 时需要设置
    ARXIV_OAI_EMBEDDING_BATCH_SIZE  每次 embedding 请求的文本条数，默认 20
    ARXIV_OAI_EMBEDDING_RETENTION_MONTHS  论文向量保留窗口（月），默认 6

退出码:
    0 = 全部日期成功（含没有新的完整日期可同步）；1 = 某一天同步失败，或另一个同步正在运行；
    2 = 参数错误，或既没有游标、库也为空而无法确定起点；130 = 被 Ctrl+C 中断。
"""

import argparse
import logging
import signal
import sys
import time
from datetime import date
from pathlib import Path


# 本脚本可能从任意工作目录启动，这里把 backend 目录加入 sys.path，
# 保证下面能以 services.xxx 的形式导入后端模块，并沿用后端同一套配置和路径规则。
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.arxiv.arxiv_oai_service import (  # noqa: E402
    ArxivOaiDatabaseService,
    ArxivOaiSyncService,
    prune_out_of_window_embeddings,
)
from services.arxiv.oai_incremental_sync import IncrementalSyncOptions, run_incremental_sync  # noqa: E402
from utils.config import OAI_SQLITE_CONFIG  # noqa: E402
from utils.storage_paths import ARXIV_OAI_SYNC_STATE_FILE, resolve_backend_artifact_path  # noqa: E402

logger = logging.getLogger(__name__)


def _parse_date(value: str) -> date:
    """argparse 的类型转换函数：校验 YYYY-MM-DD 格式。"""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date format: {value!r}; expected YYYY-MM-DD") from exc


def build_parser() -> argparse.ArgumentParser:
    """定义命令行参数。日期是闭区间，按 OAI 记录的 datestamp（最后修改日期）匹配，而非首次提交日期。"""
    parser = argparse.ArgumentParser(description="Sync arXiv metadata from OAI-PMH into arxiv_oai_papers, day by day.")
    parser.add_argument("--from", dest="from_date", type=_parse_date, help="Start date (YYYY-MM-DD). Default: day after the cursor.")
    parser.add_argument("--until", dest="until_date", type=_parse_date, help="End date (YYYY-MM-DD). Default: yesterday.")
    # 相对路径以 backend 目录为基准，与后端读取的位置保持一致。
    parser.add_argument(
        "--state-file",
        default=str(ARXIV_OAI_SYNC_STATE_FILE),
        help="Cursor file holding the last fully synced day.",
    )
    # 增量脚本会传固定路径 backend/data/arxiv-oai-sync/sync_arxiv_oai_since_last_run.meta.json；
    # 手动补数据时一般不传，以免覆盖首页显示的“最近一次同步”信息。
    parser.add_argument("--meta-file", help="Optional JSON file storing the latest run summary for the dashboard.")
    parser.add_argument(
        "--lock-file",
        default=f"{OAI_SQLITE_CONFIG['database_path']}.sync.lock",
        help="Lock file that prevents two syncs from writing at the same time.",
    )
    # --dry-run：统计中的 records_written 会按“本会写入”的条数累加，便于预估写入量。
    parser.add_argument("--dry-run", action="store_true", help="Parse and log results without writing to the database.")
    # --count-only：只累加 records_matched，records_written 保持 0。
    parser.add_argument("--count-only", action="store_true", help="Count matching papers only. Skip database writes.")
    # arXiv 对 OAI 抓取频率很敏感，服务层会把间隔强制抬到至少 3.5 秒，传更小的值不会生效；天与天之间同样间隔。
    parser.add_argument("--interval", type=float, default=5.0, help="Sleep interval between requests in seconds.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds.")
    # 每一页请求的最大尝试次数；遇到 429/503 会优先遵守服务端的 Retry-After，否则按指数退避（上限 60 秒）。
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum retry attempts for each page request.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    return parser


def _raise_keyboard_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def main() -> int:
    # systemctl stop 发送的是 SIGTERM；转成与 Ctrl+C 相同的中断路径，摘要才会记成“被中断”而不是一直停在 running。
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    parser = build_parser()
    args = parser.parse_args()
    if args.count_only and args.dry_run:
        parser.error("--count-only cannot be combined with --dry-run")

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    service = ArxivOaiSyncService(
        request_interval_seconds=args.interval,
        request_timeout_seconds=args.timeout,
        max_retries=args.max_retries,
        # 正式同步时一并写论文向量，推荐的兴趣簇召回才能看到新论文；演练和只计数不写库，也就不连向量库、不花钱。
        with_embeddings=not (args.dry_run or args.count_only),
    )
    options = IncrementalSyncOptions(
        from_date=args.from_date,
        until_date=args.until_date,
        state_file=resolve_backend_artifact_path(args.state_file),
        meta_file=resolve_backend_artifact_path(args.meta_file) if args.meta_file else None,
        lock_file=resolve_backend_artifact_path(args.lock_file),
        dry_run=args.dry_run,
        count_only=args.count_only,
    )
    exit_code = run_incremental_sync(
        options,
        sync_day=lambda day: service.sync(day, day, dry_run=args.dry_run, count_only=args.count_only),
        latest_datestamp=lambda: ArxivOaiDatabaseService().get_latest_oai_datestamp(),
        pause=lambda: time.sleep(service.request_interval_seconds),
    )
    if service.embedding_handler is not None and exit_code == 0:
        # 清理移出保留窗口的向量只影响推荐召回范围，失败不改变同步本身的退出码，下次同步会再清。
        try:
            pruned = prune_out_of_window_embeddings(service.embedding_handler)
            logger.info("Pruned %s paper embeddings older than %s", pruned, service.embedding_handler.min_paper_date)
        except Exception:
            logger.exception("Failed to prune out-of-window paper embeddings")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
