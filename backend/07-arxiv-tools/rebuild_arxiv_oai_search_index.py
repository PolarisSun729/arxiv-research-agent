import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict


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
    parser = argparse.ArgumentParser(description="Rebuild local arXiv OAI search indexes from arxiv_oai_papers.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    parser.add_argument("--reset", action="store_true", help="Clear existing partial indexes and rebuild from scratch.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=OAI_INDEX_REBUILD_BATCH_SIZE,
        help=f"Papers to commit per rebuild batch (default: {OAI_INDEX_REBUILD_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--stale-after-seconds",
        type=int,
        default=OAI_INDEX_REBUILD_STALE_SECONDS,
        help=f"Seconds before a rebuilding heartbeat can be taken over (default: {OAI_INDEX_REBUILD_STALE_SECONDS}).",
    )
    parser.add_argument("--status", action="store_true", help="Print current rebuild status without starting rebuild.")
    return parser


def _print_payload(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> int:
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
