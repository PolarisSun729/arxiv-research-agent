"""研究轨迹落盘：一次证据研究运行一个文件，天然按运行分桶。

研究轨迹是评测的数据源（PRD 方案 A：检索候选的 query + chunk_id + rank 走事件流），但它
不属于对外契约——``PaperEvidenceResearchResult`` 只暴露研究摘要。因此轨迹由服务层在图跑完
后一次性交给这里落盘，文件名即 research_run_id：同一运行不会被别的运行覆盖，评测运行器可以
直接按 run_id 反查。

写盘失败只告警，绝不影响研究结果——与 ``paper_qa_service._write_qa_trace`` 同策略。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.secret_redaction import redact_sensitive_value

logger = logging.getLogger(__name__)

DEFAULT_RESEARCH_TRACE_DIR = Path("temp/paper-evidence-research-traces")

_UNSAFE_RUN_ID_RE = re.compile(r"[^0-9A-Za-z._-]+")


class ResearchTraceRecorder:
    """把研究轨迹事件写成 ``{trace_dir}/{research_run_id}.json``。"""

    def __init__(self, *, trace_dir: Path | str = DEFAULT_RESEARCH_TRACE_DIR) -> None:
        self.trace_dir = Path(trace_dir)

    def trace_path_for(self, run_id: str) -> Path:
        # 评测记录要能只靠 run_id 反查轨迹文件，所以路径规则必须对外可复算。
        safe_run_id = _UNSAFE_RUN_ID_RE.sub("_", str(run_id or "")).strip("_") or "research"
        return self.trace_dir / f"{safe_run_id}.json"

    def __call__(self, run_id: str, events: list[dict[str, Any]]) -> str | None:
        trace_path = self.trace_path_for(run_id)
        payload = {
            "run_id": str(run_id or ""),
            "written_at": datetime.now(timezone.utc).isoformat(),
            "event_count": len(events or []),
            "events": list(events or []),
        }
        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(
                # 研究事件可能保留 provider 失败原因，完整落盘前仍需移除凭据。
                json.dumps(redact_sensitive_value(payload), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("research trace write failed: run_id=%s error=%s", run_id, exc)
            return None
        return str(trace_path)
