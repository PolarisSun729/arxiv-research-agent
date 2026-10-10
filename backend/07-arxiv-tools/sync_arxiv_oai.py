"""按日期区间从 arXiv OAI-PMH 接口同步论文元数据到本地 SQLite。

作用:
    抓取 [--from, --until] 区间内 arXiv 上新增或修改过的论文，按目标分类过滤后
    upsert 到本地 OAI 库（默认 backend/06-database/arxiv_oai.db 的 arxiv_oai_papers 表）。
    写库的同时会增量更新 FTS5 全文索引和分类索引，所以日常同步完不需要再跑
    rebuild_arxiv_oai_search_index.py。只有 ARXIV_DATA_SOURCE=local 时后端才会读这个库。

谁来调用:
    - Linux 生产：sync_arxiv_oai_since_last_run.sh（由 deploy/arxiv-oai-sync.timer 每天触发）
    - Windows 本地：sync_arxiv_oai_since_last_run.cmd（双击运行）
    - 手动补数据：直接运行本脚本，指定任意日期区间

常用示例（在 backend 目录下执行）:
    # 正式同步：写库并更新搜索索引
    python 07-arxiv-tools/sync_arxiv_oai.py --from 2026-10-01 --until 2026-10-07
    # 演练：完整走一遍抓取和过滤，日志里打印“本会写入”的论文，不写库
    python 07-arxiv-tools/sync_arxiv_oai.py --from 2026-10-01 --until 2026-10-07 --dry-run
    # 只统计区间内命中目标分类的论文数，不写库
    python 07-arxiv-tools/sync_arxiv_oai.py --from 2026-10-01 --until 2026-10-07 --count-only

相关配置（环境变量，见 backend/utils/config.py）:
    ARXIV_OAI_ENDPOINT           OAI-PMH 地址，默认 https://oaipmh.arxiv.org/oai
    ARXIV_OAI_TARGET_CATEGORIES  目标分类，逗号分隔，默认 cs.CL,cs.LG,cs.IR,cs.AI；
                                 论文只要挂了其中任一分类（包括 cross-list）就会保留
    OAI_SQLITE_DATABASE_PATH     OAI 库路径，默认 backend/06-database/arxiv_oai.db
    HTTP_PROXY / HTTPS_PROXY     requests 会自动读取，国内网络访问 arXiv 时需要设置

退出码:
    0 = 成功且统计里 errors == 0；1 = 有错误（抓取失败、XML 解析失败、写库失败等）。
    外层增量脚本只在退出码为 0 时推进游标，失败的日期会在下次运行时重新同步。
    upsert 是幂等的，同一区间重复同步不会产生重复数据。
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# 本脚本可能从任意工作目录启动，这里把 backend 目录加入 sys.path，
# 保证下面能以 services.xxx 的形式导入后端模块，并沿用后端同一套配置和路径规则。
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.arxiv.arxiv_oai_service import ArxivOaiSyncService  # noqa: E402


def _parse_date(value: str) -> str:
    """argparse 的类型转换函数：校验 YYYY-MM-DD 格式并规范化输出（如补齐前导零）。"""
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
    """组装写入 --meta-file 的本次运行摘要。

    这个 JSON 由后端 GET /sync-status（routers/paper_router.py 的 _get_sync_status_payload）读取，
    用于在首页展示“上次同步时间、同步到哪一天、新增多少篇、是否出错”。
    顶层的 records_written / records_matched / errors 是从 summary 里提出来的常用字段，
    方便前端直接读取；完整统计仍保留在 summary 中。
    """
    summary = summary or {}
    status = "success" if exit_code == 0 else "failed"
    # 三种模式互斥（main 中已校验），优先级只是为了写法简洁。
    mode = "count_only" if args.count_only else "dry_run" if args.dry_run else "sync"
    return {
        "status": status,
        "mode": mode,
        "from_date": args.from_date,
        "until_date": args.until_date,
        "started_at": started_at.astimezone().isoformat(),
        "finished_at": finished_at.astimezone().isoformat(),
        "exit_code": exit_code,
        # 只有成功时才声明“已同步到 until_date”；失败时为 None，首页会回退读取游标文件里的日期。
        "last_successful_until": args.until_date if exit_code == 0 else None,
        "records_written": int(summary.get("records_written", 0) or 0),
        "records_matched": int(summary.get("records_matched", 0) or 0),
        "errors": int(summary.get("errors", 0) or 0),
        "error_message": error_message,
        "summary": summary,
    }


def _write_sync_metadata(metadata_file: str, payload: Dict[str, Any]) -> None:
    """把运行摘要原子地写入 metadata_file，目录不存在时自动创建。"""
    path = Path(metadata_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 先写同目录临时文件再替换，避免首页恰好读取到半截 JSON。
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        # replace 成功后临时文件已不存在；写入或替换失败时清理残留。
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    """定义命令行参数。日期区间是闭区间，按 OAI 记录的 datestamp（最后修改日期）匹配，而非首次提交日期。"""
    parser = argparse.ArgumentParser(description="Sync arXiv metadata from OAI-PMH into arxiv_oai_papers.")
    parser.add_argument("--from", dest="from_date", required=True, type=_parse_date, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--until", dest="until_date", required=True, type=_parse_date, help="End date in YYYY-MM-DD format.")
    # --dry-run：统计中的 records_written 会按“本会写入”的条数累加，便于预估写入量。
    parser.add_argument("--dry-run", action="store_true", help="Parse and log results without writing to the database.")
    # --count-only：只累加 records_matched，records_written 保持 0。
    # 注意：本脚本创建同步服务时没有传入 embedding / 向量库服务，所以正式同步本身也不会生成向量；
    # help 里的 “embeddings” 指服务层在注入向量依赖时才会执行的那一步。
    parser.add_argument("--count-only", action="store_true", help="Count matching papers only. Skip database writes and embeddings.")
    # 增量脚本会传固定路径 backend/data/arxiv-oai-sync/sync_arxiv_oai_since_last_run.meta.json；
    # 手动补数据时一般不传，以免覆盖首页显示的“最近一次同步”信息。
    parser.add_argument("--meta-file", dest="meta_file", help="Optional JSON file path used to store the latest sync run summary.")
    # arXiv 对 OAI 抓取频率很敏感，服务层会把间隔强制抬到至少 3.5 秒，传更小的值不会生效。
    parser.add_argument("--interval", type=float, default=5.0, help="Sleep interval between requests in seconds.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds.")
    # 每一页请求的最大尝试次数；遇到 429/503 会优先遵守服务端的 Retry-After，否则按指数退避（上限 60 秒）。
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum retry attempts for each page request.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    return parser


def main() -> int:
    """执行一次同步并返回进程退出码；无论成功失败，只要传了 --meta-file 都会写运行摘要。"""
    parser = build_parser()
    args = parser.parse_args()
    started_at = datetime.now().astimezone()
    # 默认按失败处理，只有同步正常结束且无错误时才改成 0，保证异常路径写出的摘要是 failed。
    exit_code = 1
    summary: Dict[str, Any] = {}
    error_message: Optional[str] = None

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    try:
        # parser.error 会打印用法并以退出码 2 结束进程（抛出 SystemExit），finally 仍会执行。
        if args.from_date > args.until_date:
            parser.error("--from must be less than or equal to --until")
        if args.count_only and args.dry_run:
            parser.error("--count-only cannot be combined with --dry-run")

        service = ArxivOaiSyncService(
            request_interval_seconds=args.interval,
            request_timeout_seconds=args.timeout,
            max_retries=args.max_retries,
        )
        # 服务内部按 resumptionToken 自动翻页直到取完，逐条过滤分类，再按页批量 upsert。
        stats = service.sync(args.from_date, args.until_date, dry_run=args.dry_run, count_only=args.count_only)

        summary = stats.to_dict()
        logging.getLogger(__name__).info("Sync finished with summary: %s", summary)
        # 统计结果打印到 stdout，便于手动运行时直接查看，也会进入 systemd journal。
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        # 同步过程中出错不会抛异常，而是累加到 errors；只要有错误就返回 1，让外层不推进游标。
        exit_code = 0 if summary.get("errors", 0) == 0 else 1
        return exit_code
    except Exception as exc:
        # 未预期的异常：记入摘要后继续抛出，让进程以非 0 退出并保留完整堆栈。
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
                # 摘要写失败只记日志，不覆盖同步本身的结果和退出码。
                logging.getLogger(__name__).exception("Failed to write sync metadata file: %s", args.meta_file)


if __name__ == "__main__":
    raise SystemExit(main())
