"""根据 arxiv_oai_papers 表全量重建本地 OAI 搜索索引。

背景:
    本地 OAI 库（默认 backend/06-database/arxiv_oai.db）里，arxiv_oai_papers 存论文原始元数据，
    另有两张由它派生出来的索引表，本地搜索（ARXIV_DATA_SOURCE=local）实际查的是这两张：
      - arxiv_oai_papers_fts        FTS5 全文索引，按关键词搜标题、摘要
      - arxiv_oai_paper_categories  分类索引，按 cs.CL 等分类过滤
    日常的 sync_arxiv_oai.py 在写库时会顺手增量更新这两张表，平时不需要运行本脚本。

什么时候需要运行:
    1. 首次导入或批量灌入历史数据后，papers 表有数据但索引是空的；
    2. 升级了索引格式（arxiv_oai_service.py 中的 OAI_INDEX_REBUILD_VERSION 变化）；
    3. 索引损坏，或后端搜索报 “index_missing / index_rebuild_failed”，提示去运行重建命令。

运行行为:
    - 默认（不带 --reset）：上一轮重建处于 failed 状态，或 rebuilding 但心跳已超时（进程被杀），
      就从断点续跑，已提交的批次不会重做；否则清空两张索引表，从头开始新一轮。
    - --reset：无论之前是什么状态，都清空索引从头重建（续跑校验失败时使用）。
    - --status：只读查询当前重建进度和索引覆盖情况，不做任何修改。
    - 同一时间只允许一个重建任务；另一个任务心跳正常时再运行会直接报错退出。
    - 重建过程中搜索接口会返回 “索引重建中” 的错误，结束后自动恢复。

常用示例（在 backend 目录下执行）:
    python 07-arxiv-tools/rebuild_arxiv_oai_search_index.py            # 重建或续跑
    python 07-arxiv-tools/rebuild_arxiv_oai_search_index.py --status   # 查看进度
    python 07-arxiv-tools/rebuild_arxiv_oai_search_index.py --reset    # 强制从头重建

注意:
    重建期间会长时间写 SQLite。如果后端或同步任务也在写同一个库，可能报 database is locked，
    建议先停掉后端（生产环境：sudo systemctl stop arxiv-agent）以及 arxiv-oai-sync 定时任务再运行。
    输出为 JSON，status 为 success 或 failed；退出码 0 表示成功，1 表示失败。
"""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict


# 本脚本可能从任意工作目录启动，这里把 backend 目录加入 sys.path，
# 保证能导入后端服务，并和后端使用同一个数据库路径（OAI_SQLITE_DATABASE_PATH 或默认路径）。
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.arxiv.arxiv_oai_service import (  # noqa: E402
    OAI_INDEX_REBUILD_BATCH_SIZE,
    OAI_INDEX_REBUILD_STALE_SECONDS,
    ArxivOaiDatabaseService,
)


def build_parser() -> argparse.ArgumentParser:
    """定义命令行参数；批大小和心跳超时的默认值与服务层常量保持一致。"""
    parser = argparse.ArgumentParser(description="Rebuild local arXiv OAI search indexes from arxiv_oai_papers.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    parser.add_argument("--reset", action="store_true", help="Clear existing partial indexes and rebuild from scratch.")
    # 每处理这么多篇论文提交一次，并记录断点（last_arxiv_id）。
    # 续跑时批大小必须和中断前那一轮相同，否则会报错，需要改回原值或加 --reset。
    parser.add_argument(
        "--batch-size",
        type=int,
        default=OAI_INDEX_REBUILD_BATCH_SIZE,
        help=f"Papers to commit per rebuild batch (default: {OAI_INDEX_REBUILD_BATCH_SIZE}).",
    )
    # 重建任务会定期更新心跳时间；超过这个秒数没有心跳，就认为原进程已经死掉，允许新进程接管续跑。
    # 调得太小可能会抢走一个仍在正常运行的任务。
    parser.add_argument(
        "--stale-after-seconds",
        type=int,
        default=OAI_INDEX_REBUILD_STALE_SECONDS,
        help=f"Seconds before a rebuilding heartbeat can be taken over (default: {OAI_INDEX_REBUILD_STALE_SECONDS}).",
    )
    parser.add_argument("--status", action="store_true", help="Print current rebuild status without starting rebuild.")
    return parser


def _print_payload(payload: Dict[str, Any]) -> None:
    """以 JSON 形式输出结果到 stdout，便于人工查看，也方便其他脚本解析。"""
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> int:
    """根据参数查询状态或执行重建，返回进程退出码；所有异常都转换成 JSON 输出，不抛到外层。"""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        service = ArxivOaiDatabaseService()
        if args.status:
            # 只读路径：返回状态（ready / rebuilding / rebuild_failed / *_missing 等）、
            # 已处理篇数、总篇数，以及心跳是否已超时。
            result = service.get_oai_search_index_rebuild_status(
                stale_after_seconds=args.stale_after_seconds,
            )
            payload = {
                "status": "success",
                "db_path": service.db_path,
                "result": result,
            }
            _print_payload(payload)
            return 0

        logger.info("Starting local OAI search index rebuild: db_path=%s", service.db_path)
        # 服务层负责续跑判断、分批写入、心跳更新，以及结束后的覆盖校验（确认所有论文都已进入索引）。
        result = service.rebuild_oai_search_index(
            reset=args.reset,
            batch_size=args.batch_size,
            stale_after_seconds=args.stale_after_seconds,
        )
        payload = {
            "status": "success",
            "db_path": service.db_path,
            "result": result,
        }
        logger.info("Local OAI search index rebuild finished: %s", result)
        _print_payload(payload)
        return 0
    except sqlite3.OperationalError as exc:
        message = str(exc)
        logger.exception("Local OAI search index rebuild failed")
        # 数据库被占用时直接失败，避免后台长时间重试继续制造“像卡住了一样”的体验。
        if "locked" in message.lower():
            _print_payload(
                {
                    "status": "failed",
                    "error": "database_locked",
                    "message": "SQLite 数据库当前被占用，请先停止后端或其他写入进程后重试。",
                    "details": message,
                }
            )
            return 1
        _print_payload(
            {
                "status": "failed",
                "error": "sqlite_operational_error",
                "message": message,
            }
        )
        return 1
    except Exception as exc:
        # 其他失败，如 FTS5 不可用、已有任务在运行、续跑参数不一致、结束后覆盖校验不通过。
        # 如果本次已经接手了重建任务，服务层会把状态标记为 failed，排除原因后重新运行即可从断点续跑；
        # 错误信息里提示要 --reset 的（如覆盖校验失败），则需加 --reset 从头重建。
        logger.exception("Local OAI search index rebuild failed")
        _print_payload(
            {
                "status": "failed",
                "error": exc.__class__.__name__,
                "message": str(exc),
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
