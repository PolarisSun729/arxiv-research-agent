import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.arxiv_oai_service import ArxivOaiSyncService  # noqa: E402


def _parse_date(value: str) -> str:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date format: {value!r}; expected YYYY-MM-DD") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync arXiv metadata from OAI-PMH into arxiv_oai_papers.")
    parser.add_argument("--from", dest="from_date", required=True, type=_parse_date, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--until", dest="until_date", required=True, type=_parse_date, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and log results without writing to the database.")
    parser.add_argument("--interval", type=float, default=5.0, help="Sleep interval between requests in seconds.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum retry attempts for each page request.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    if args.from_date > args.until_date:
        parser.error("--from must be less than or equal to --until")

    service = ArxivOaiSyncService(
        request_interval_seconds=args.interval,
        request_timeout_seconds=args.timeout,
        max_retries=args.max_retries,
    )
    stats = service.sync(args.from_date, args.until_date, dry_run=args.dry_run)

    summary = stats.to_dict()
    logging.getLogger(__name__).info("Sync finished with summary: %s", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("errors", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
