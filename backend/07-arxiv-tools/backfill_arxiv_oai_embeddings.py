"""扫描本地 OAI 库，给还没有论文向量的论文补 embedding，写入 Milvus 的 arxiv_paper_embeddings。

背景:
    推荐的兴趣簇召回只在 arxiv_paper_embeddings 里找论文，没有向量的论文永远召回不到。
    日常的 sync_arxiv_oai.py 落库时会顺手生成向量，但下面几种情况会留下没有向量的论文：
      1. 同步开始写向量之前就已经入库的历史论文；
      2. 同步时 embedding 服务或向量库出错（摘要里的 embedding_errors 大于 0）；
      3. 换了 embedding 模型或清空了 collection。

运行行为:
    - 只扫描首次提交日期在保留窗口内的论文（ARXIV_OAI_EMBEDDING_RETENTION_MONTHS，默认 4 个月，
      按每月 30 天算），和同步脚本、推荐近期论文池口径一致；
    - 按入库顺序从新到旧扫描 arxiv_oai_papers，每页先查向量库，只给缺向量的论文生成 embedding；
    - 已有向量的论文直接跳过，所以没有断点文件：中断或部分失败后重新运行即可接着补；
    - 只看 arxiv_id 在 collection 里有没有向量，不区分是哪个 embedding 模型生成的；
    - 和同步任务共用同一个锁文件，避免两边同时给同一篇论文写向量；
    - 补完后删除窗口外的论文向量（用户点赞或点踩过的论文除外）；--dry-run 只统计将删除多少，
      被 --limit 提前截停或中途失败时不清理。

常用示例（在 backend 目录下执行）:
    python 07-arxiv-tools/backfill_arxiv_oai_embeddings.py --dry-run      # 只统计缺多少、将删多少，不花钱也不删
    python 07-arxiv-tools/backfill_arxiv_oai_embeddings.py                # 补齐窗口内全部，再清理窗口外
    python 07-arxiv-tools/backfill_arxiv_oai_embeddings.py --limit 2000   # 本次最多补 2000 篇

相关配置（环境变量，见 backend/utils/config.py）:
    EMBEDDING_PROVIDER / EMBEDDING_MODEL 等 EMBEDDING_*   embedding 配置，和后端、同步脚本用同一套
    MILVUS_URI                        向量库地址
    ARXIV_OAI_EMBEDDING_BATCH_SIZE    每次 embedding 请求的文本条数，默认 20
    ARXIV_OAI_EMBEDDING_RETENTION_MONTHS  论文向量保留窗口（月），默认 4；0 表示不限、也不清理

输出与退出码:
    结束时以 JSON 输出统计，status 为 success / partial / interrupted / failed。
    0 = 全部成功（含没有需要补的论文）；1 = 有论文没补上（重跑即可重试）、向量库或数据库不可用，
    或同步任务正在运行；130 = 被 Ctrl+C 中断。
"""

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict


# 本脚本可能从任意工作目录启动，这里把 backend 目录加入 sys.path，
# 保证能导入后端服务，并和后端使用同一个数据库路径与 embedding 配置。
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.arxiv.arxiv_embedding_handler import EmbeddingStats  # noqa: E402
from services.arxiv.arxiv_oai_service import (  # noqa: E402
    ArxivOaiDatabaseService,
    create_oai_embedding_handler,
    prune_out_of_window_embeddings,
)
from services.arxiv.oai_incremental_sync import SyncLockError, exclusive_sync_lock  # noqa: E402
from utils.config import OAI_SQLITE_CONFIG  # noqa: E402
from utils.storage_paths import resolve_backend_artifact_path  # noqa: E402

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """定义命令行参数。"""
    parser = argparse.ArgumentParser(description="Embed arxiv_oai_papers that have no vector in arxiv_paper_embeddings yet.")
    parser.add_argument("--dry-run", action="store_true", help="Only count papers missing embeddings; call no embedding API.")
    # 按“本次实际送去生成 embedding 的篇数”计，已有向量而跳过的不算；0 表示不限。
    parser.add_argument("--limit", type=int, default=0, help="Embed at most this many papers in this run (default: 0, no limit).")
    # 每页先查一次向量库再批量 embedding、写一次向量库；写向量库每次都会 flush，页太小会拖慢整体速度。
    parser.add_argument("--page-size", type=int, default=500, help="Papers read and checked per page (default: 500).")
    parser.add_argument(
        "--lock-file",
        default=f"{OAI_SQLITE_CONFIG['database_path']}.sync.lock",
        help="Lock file shared with sync_arxiv_oai.py so the two never run at the same time.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    return parser


def _raise_keyboard_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def _print_payload(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> int:
    # systemctl stop / kill 发送的是 SIGTERM；转成与 Ctrl+C 相同的中断路径，照样输出已完成部分的统计。
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    parser = build_parser()
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >= 0")

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    started_at = time.perf_counter()
    stats = EmbeddingStats()
    progress = {"window_start": None, "total_papers": 0, "scanned": 0, "missing_found": 0, "pruned_out_of_window": None}
    status = "success"
    error_message = None

    def summary() -> Dict[str, Any]:
        return {
            "status": status,
            "mode": "dry_run" if args.dry_run else "backfill",
            "limit": args.limit or None,
            **progress,
            **asdict(stats),
            "duration_seconds": round(time.perf_counter() - started_at, 1),
            "error_message": error_message,
        }

    try:
        database_service = ArxivOaiDatabaseService()
        handler = create_oai_embedding_handler()
        window_start = handler.min_paper_date
        progress["window_start"] = window_start
        progress["total_papers"] = database_service.count_papers_since(window_start)
        logger.info(
            "Starting OAI embedding backfill: papers=%s window_start=%s dry_run=%s limit=%s model=%s",
            progress["total_papers"],
            window_start,
            args.dry_run,
            args.limit or None,
            handler.embedding_config.model_name,
        )
        with exclusive_sync_lock(resolve_backend_artifact_path(args.lock_file)):
            reached_limit = False
            for page in database_service.iter_papers_newest_first(since=window_start, page_size=args.page_size):
                progress["scanned"] += len(page)
                # 向量库查询失败直接抛出终止：库不可用时每一页都会失败，继续扫描没有意义。
                missing_papers = handler.find_papers_missing_embeddings(page)
                progress["missing_found"] += len(missing_papers)
                if not args.dry_run and missing_papers:
                    if args.limit:
                        missing_papers = missing_papers[: max(0, args.limit - stats.embeddings_attempted)]
                    handler.store_paper_embeddings_batch(missing_papers, stats)
                logger.info(
                    "Progress: scanned=%s/%s missing=%s written=%s errors=%s cost=%.4f yuan",
                    progress["scanned"],
                    progress["total_papers"],
                    progress["missing_found"],
                    stats.embeddings_written,
                    stats.embedding_errors,
                    stats.embedding_cost_yuan,
                )
                if args.limit and not args.dry_run and stats.embeddings_attempted >= args.limit:
                    logger.info("Reached --limit %s, stopping", args.limit)
                    reached_limit = True
                    break
            # 被 --limit 截停时窗口内还没补完，这次只补不删，等补齐那次再一起清理。
            if not reached_limit:
                progress["pruned_out_of_window"] = prune_out_of_window_embeddings(handler, dry_run=args.dry_run)
                logger.info(
                    "%s %s out-of-window paper embeddings (published before %s)",
                    "Would prune" if args.dry_run else "Pruned",
                    progress["pruned_out_of_window"],
                    window_start,
                )
        if stats.embedding_errors:
            status = "partial"
    except SyncLockError as exc:
        status = "failed"
        error_message = str(exc)
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as exc:
        logger.exception("OAI embedding backfill failed")
        status = "failed"
        error_message = f"{exc.__class__.__name__}: {exc}"

    _print_payload(summary())
    if status == "interrupted":
        return 130
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
