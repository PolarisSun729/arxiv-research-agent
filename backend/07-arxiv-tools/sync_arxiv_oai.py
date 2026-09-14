import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.arxiv.arxiv_oai_service import ArxivOaiSyncService  # noqa: E402


def _parse_date(value: str) -> str:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date format: {value!r}; expected YYYY-MM-DD") from exc


def _build_sync_metadata(
    args: argparse.Namespace,
    *,
    started_at: datetime,
    finished_at: datetime,
    exit_code: int,
    summary: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    summary = summary or {}
    status = "success" if exit_code == 0 else "failed"
    mode = "count_only" if args.count_only else "dry_run" if args.dry_run else "sync"
    return {
        "status": status,
        "mode": mode,
        "from_date": args.from_date,
        "until_date": args.until_date,
        "started_at": started_at.astimezone().isoformat(),
        "finished_at": finished_at.astimezone().isoformat(),
        "exit_code": exit_code,
        "last_successful_until": args.until_date if exit_code == 0 else None,
        "records_written": int(summary.get("records_written", 0) or 0),
        "records_matched": int(summary.get("records_matched", 0) or 0),
        "errors": int(summary.get("errors", 0) or 0),
        "error_message": error_message,
        "summary": summary,
    }


def _write_sync_metadata(metadata_file: str, payload: Dict[str, Any]) -> None:
    path = Path(metadata_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 先写同目录临时文件再替换，避免首页恰好读取到半截 JSON。
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync arXiv metadata from OAI-PMH into arxiv_oai_papers.")
    parser.add_argument("--from", dest="from_date", required=True, type=_parse_date, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--until", dest="until_date", required=True, type=_parse_date, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and log results without writing to the database.")
    parser.add_argument("--count-only", action="store_true", help="Count matching papers only. Skip database writes and embeddings.")
    parser.add_argument("--meta-file", dest="meta_file", help="Optional JSON file path used to store the latest sync run summary.")
    parser.add_argument("--interval", type=float, default=5.0, help="Sleep interval between requests in seconds.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum retry attempts for each page request.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    started_at = datetime.now().astimezone()
    exit_code = 1
    summary: Dict[str, Any] = {}
    error_message: Optional[str] = None

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    try:
        if args.from_date > args.until_date:
            parser.error("--from must be less than or equal to --until")
        if args.count_only and args.dry_run:
            parser.error("--count-only cannot be combined with --dry-run")

        service = ArxivOaiSyncService(
            request_interval_seconds=args.interval,
            request_timeout_seconds=args.timeout,
            max_retries=args.max_retries,
        )
        stats = service.sync(args.from_date, args.until_date, dry_run=args.dry_run, count_only=args.count_only)

        summary = stats.to_dict()
        logging.getLogger(__name__).info("Sync finished with summary: %s", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        exit_code = 0 if summary.get("errors", 0) == 0 else 1
        return exit_code
    except Exception as exc:
        error_message = str(exc)
        summary = {"errors": 1, "error_message": error_message}
        logging.getLogger(__name__).exception("Sync failed")
        raise
    finally:
        if args.meta_file:
            finished_at = datetime.now().astimezone()
            payload = _build_sync_metadata(
                args,
                started_at=started_at,
                finished_at=finished_at,
                exit_code=exit_code,
                summary=summary,
                error_message=error_message,
            )
            try:
                _write_sync_metadata(args.meta_file, payload)
                logging.getLogger(__name__).info("Wrote sync metadata file: %s", args.meta_file)
            except Exception:
                logging.getLogger(__name__).exception("Failed to write sync metadata file: %s", args.meta_file)


if __name__ == "__main__":
    raise SystemExit(main())
