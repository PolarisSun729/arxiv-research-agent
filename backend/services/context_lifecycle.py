from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from services.storage.database_service import DatabaseService
from utils.config import get_agent_runtime_checkpoint_config, get_context_lifecycle_config, get_default_user_id

logger = logging.getLogger(__name__)


class ContextLifecycleService:
    """统一管理上下文数据的生命周期策略、清理入口和健康度诊断。

    本服务不改变原始聊天消息和 session summary 的长期保留语义；它只负责运行时 checkpoint、
    LangGraph checkpoint、debug/trace 这类容易无限增长的数据边界。
    """

    def __init__(
        self,
        *,
        db_service: Optional[DatabaseService] = None,
        lifecycle_config: Optional[Mapping[str, Any]] = None,
        checkpoint_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.db_service = db_service or DatabaseService()
        self.lifecycle_config = dict(lifecycle_config or get_context_lifecycle_config())
        self.checkpoint_config = dict(checkpoint_config or get_agent_runtime_checkpoint_config())

    def retention_policy(self) -> Dict[str, Any]:
        """返回可观测的策略快照，便于 debug 说明每类上下文是否参与 prompt 和清理。"""
        checkpoint_retention_days = max(int(self.checkpoint_config.get("cleanup_retention_days") or 7), 1)
        return {
            "paper_chat_messages": {
                "retention": "long_term",
                "prompt_participation": "never_full_history",
                "read_policy": "frontend_paged_or_model_recent_turn_query",
                "cleanup": "manual_or_session_delete",
            },
            "session_summary": {
                "retention": "long_term",
                "prompt_participation": "summary_section",
                "cleanup": "with_session",
                "bounded": True,
            },
            "recent_turns": {
                "retention": "not_stored_separately",
                "prompt_participation": "recent_turns_section",
                "read_policy": "database_limited_window",
            },
            "agent_runtime_checkpoint": {
                "retention": f"terminal_{checkpoint_retention_days}_days",
                "prompt_participation": "agent_state_summary_only",
                "cleanup": "expire_then_delete_terminal",
            },
            "langgraph_checkpoint": {
                "retention": "aligned_with_agent_runtime_checkpoint",
                "prompt_participation": "never",
                "cleanup": "delete_when_runtime_terminal_retention_elapsed",
            },
            "retrieval_debug": {
                "retention": "assistant_message_snapshot_summarized",
                "prompt_participation": "never",
                "snapshot_mode": self.debug_snapshot_mode,
            },
            "tool_trace": {
                "retention": "bounded_debug_trace",
                "prompt_participation": "never",
                "max_files_per_paper": self.max_trace_files_per_paper,
            },
            "paper_index_artifacts": {
                "retention": "index_lifecycle",
                "prompt_participation": "rag_evidence_only_after_retrieval",
                "cleanup": "qa_index_artifact_status",
            },
        }

    @property
    def debug_snapshot_mode(self) -> str:
        mode = str(self.lifecycle_config.get("debug_snapshot_mode") or "summary").strip().lower()
        return mode if mode in {"full", "summary", "minimal"} else "summary"

    @property
    def max_debug_string_chars(self) -> int:
        return max(int(self.lifecycle_config.get("max_debug_string_chars") or 1200), 120)

    @property
    def max_debug_list_items(self) -> int:
        return max(int(self.lifecycle_config.get("max_debug_list_items") or 8), 1)

    @property
    def max_debug_depth(self) -> int:
        return max(int(self.lifecycle_config.get("max_debug_depth") or 5), 1)

    @property
    def max_trace_files_per_paper(self) -> int:
        return max(int(self.lifecycle_config.get("max_trace_files_per_paper") or 20), 1)

    def run_startup_cleanup(self, *, trace_export_dir: Optional[Any] = None) -> Dict[str, Any]:
        """启动时执行轻量清理；失败只进入结果，不阻断服务启动。"""
        checkpoint_retention_days = max(int(self.checkpoint_config.get("cleanup_retention_days") or 7), 1)
        result: Dict[str, Any] = {
            "policy": self.retention_policy(),
            "expired_runtime_checkpoints": 0,
            "deleted_runtime_checkpoints": 0,
            "deleted_langgraph": {"threads": 0, "checkpoints": 0, "writes": 0},
            "trace_cleanup": {"enabled": False},
            "errors": [],
        }
        try:
            result["expired_runtime_checkpoints"] = self.db_service.expire_agent_runtime_checkpoints()
        except Exception as exc:
            logger.warning("Context lifecycle runtime checkpoint expiration failed: %s", exc)
            result["errors"].append({"stage": "expire_runtime_checkpoints", "error": str(exc)})
        try:
            # 先清理原始 LangGraph checkpoint，再删业务 runtime 记录，避免失去 thread_id 对齐依据。
            result["deleted_langgraph"] = self.db_service.cleanup_langgraph_checkpoints_for_terminal_runtime(
                retention_days=checkpoint_retention_days,
            )
        except Exception as exc:
            logger.warning("Context lifecycle langgraph checkpoint cleanup failed: %s", exc)
            result["errors"].append({"stage": "cleanup_langgraph_checkpoints", "error": str(exc)})
        try:
            result["deleted_runtime_checkpoints"] = self.db_service.cleanup_agent_runtime_checkpoints(
                retention_days=checkpoint_retention_days,
            )
        except Exception as exc:
            logger.warning("Context lifecycle runtime checkpoint cleanup failed: %s", exc)
            result["errors"].append({"stage": "cleanup_runtime_checkpoints", "error": str(exc)})
        if trace_export_dir:
            result["trace_cleanup"] = self.cleanup_retrieval_traces(trace_export_dir)
        return result

    def summarize_startup_cleanup_result(self, result: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        """把启动清理结果压成适合 INFO 的摘要字段，避免整份策略快照把主日志刷成超长单行。"""
        payload = dict(result or {})
        deleted_langgraph = dict(payload.get("deleted_langgraph") or {})
        trace_cleanup = dict(payload.get("trace_cleanup") or {})
        errors = list(payload.get("errors") or [])
        summary: Dict[str, Any] = {
            "expired_runtime_checkpoints": int(payload.get("expired_runtime_checkpoints") or 0),
            "deleted_runtime_checkpoints": int(payload.get("deleted_runtime_checkpoints") or 0),
            "deleted_langgraph_threads": int(deleted_langgraph.get("threads") or 0),
            "deleted_langgraph_checkpoints": int(deleted_langgraph.get("checkpoints") or 0),
            "deleted_langgraph_writes": int(deleted_langgraph.get("writes") or 0),
            "trace_cleanup_enabled": bool(trace_cleanup.get("enabled")),
            "trace_cleanup_deleted_files": int(trace_cleanup.get("deleted_files") or 0),
            "error_count": len(errors),
        }
        if trace_cleanup:
            # trace 目录是否缺失只影响诊断，不需要把整份 trace_cleanup 结构塞进 INFO 日志。
            summary["trace_cleanup_missing"] = bool(trace_cleanup.get("missing"))
            summary["trace_cleanup_failed"] = bool(trace_cleanup.get("error"))
        error_stages = [
            str(item.get("stage")).strip()
            for item in errors
            if isinstance(item, Mapping) and str(item.get("stage") or "").strip()
        ]
        if error_stages:
            # 只保留失败阶段名，既能快速判断卡在哪一段，也避免把完整异常文本重复打到主时间线。
            summary["error_stages"] = error_stages
        return summary

    def cleanup_retrieval_traces(self, trace_export_dir: Any) -> Dict[str, Any]:
        """按论文分桶保留最近 trace 文件，避免开发 trace 目录无限增长。"""
        result = {"enabled": True, "root": str(trace_export_dir), "deleted_files": 0, "kept_per_paper": self.max_trace_files_per_paper}
        try:
            root = Path(str(trace_export_dir)).resolve()
            if not root.exists() or not root.is_dir():
                result["missing"] = True
                return result
            for paper_dir in root.iterdir():
                if not paper_dir.is_dir():
                    continue
                files = [path for path in paper_dir.iterdir() if path.is_file() and path.suffix.lower() in {".json", ".md"}]
                files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
                for path in files[self.max_trace_files_per_paper :]:
                    # 删除前再次确认目标仍在 trace 根目录下，防止异常路径污染清理范围。
                    if root not in path.resolve().parents:
                        continue
                    path.unlink(missing_ok=True)
                    result["deleted_files"] += 1
            return result
        except Exception as exc:
            logger.warning("Context lifecycle trace cleanup failed: root=%s error=%s", trace_export_dir, exc)
            result["error"] = str(exc)
            return result

    def build_paper_qa_health_debug(
        self,
        *,
        user_id: str = "",
        session_id: str = "",
        short_term_debug: Optional[Mapping[str, Any]] = None,
        session_summary: Optional[Mapping[str, Any]] = None,
        prompt_context_debug: Optional[Mapping[str, Any]] = None,
        degraded: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """构造本轮 QA 上下文健康度，明确数据库读取规模和 prompt 使用规模。"""
        normalized_user_id = str(user_id or get_default_user_id()).strip() or get_default_user_id()
        stats = self.db_service.get_context_lifecycle_stats(
            user_id=normalized_user_id,
            paper_session_id=str(session_id or "").strip() or None,
        )
        short_term = dict(short_term_debug or {})
        prompt_debug = dict(prompt_context_debug or {})
        return {
            "policy": self.retention_policy(),
            "paper_chat": stats.get("paper_chat", {}),
            "runtime_context": {
                "db_message_read_count": short_term.get("db_message_read_count"),
                "db_message_read_limit": short_term.get("db_message_read_limit"),
                "total_message_count": short_term.get("total_message_count"),
                "used_turn_count": short_term.get("merged_turn_count") or short_term.get("turn_count"),
                "summary_loaded": bool(session_summary),
                "summary_turn_count": (session_summary or {}).get("summary_turn_count") if isinstance(session_summary, Mapping) else None,
                "filtered_incomplete_turn_count": short_term.get("filtered_incomplete_turn_count"),
            },
            "prompt_budget": {
                "section_order": prompt_debug.get("section_order"),
                "used_chars": prompt_debug.get("used_chars"),
                "total_budget_chars": prompt_debug.get("total_budget_chars"),
                "estimated_tokens": prompt_debug.get("estimated_tokens"),
                "truncated_sections": prompt_debug.get("truncated_sections"),
            },
            "degraded": dict(degraded or {}),
        }

    def build_agent_health_debug(
        self,
        *,
        user_id: str = "",
        session_id: str = "",
        context_merge_debug: Optional[Mapping[str, Any]] = None,
        degraded: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """构造 Agent 上下文健康度，聚焦 checkpoint 是否存在及前端 context 是否被降级。"""
        normalized_user_id = str(user_id or get_default_user_id()).strip() or get_default_user_id()
        normalized_session_id = str(session_id or "").strip()
        stats = self.db_service.get_context_lifecycle_stats(
            user_id=normalized_user_id,
            agent_session_id=normalized_session_id or None,
        )
        return {
            "policy": self.retention_policy(),
            "agent_runtime_checkpoint": stats.get("agent_runtime_checkpoint", {}),
            "langgraph_checkpoint": stats.get("langgraph_checkpoint", {}),
            "context_merge": dict(context_merge_debug or {}),
            "degraded": dict(degraded or {}),
        }

    def prepare_debug_snapshot(self, value: Any) -> Any:
        """按运行模式裁剪 debug 快照；正常聊天正文和 sources 不走这里。"""
        if self.debug_snapshot_mode == "full":
            return value
        if self.debug_snapshot_mode == "minimal":
            return self._minimal_debug(value)
        return self._summarize_debug(value, depth=0)

    def _minimal_debug(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            keep_keys = {
                "original_question",
                "contextualized_question",
                "question_contextualization",
                "memory_modules",
                "memory_context",
                "context_lifecycle",
                "generation",
                "verification",
                "context_pack",
            }
            return {key: self._summarize_debug(item, depth=1) for key, item in dict(value).items() if key in keep_keys}
        return self._summarize_debug(value, depth=0)

    def _summarize_debug(self, value: Any, *, depth: int) -> Any:
        if depth >= self.max_debug_depth:
            return self._shape_marker(value)
        if isinstance(value, str):
            if len(value) <= self.max_debug_string_chars:
                return value
            return {
                "preview": value[: self.max_debug_string_chars].rstrip() + "...",
                "original_chars": len(value),
                "truncated": True,
            }
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, Mapping):
            return {str(key): self._summarize_debug(item, depth=depth + 1) for key, item in dict(value).items()}
        if isinstance(value, (list, tuple)):
            items = list(value)
            summarized = [self._summarize_debug(item, depth=depth + 1) for item in items[: self.max_debug_list_items]]
            if len(items) > self.max_debug_list_items:
                summarized.append({"truncated_items": len(items) - self.max_debug_list_items, "original_count": len(items)})
            return summarized
        return self._summarize_debug(str(value), depth=depth)

    @staticmethod
    def _shape_marker(value: Any) -> Dict[str, Any]:
        if isinstance(value, Mapping):
            return {"type": "object", "keys": list(dict(value).keys())[:12], "truncated": True}
        if isinstance(value, (list, tuple)):
            return {"type": "list", "count": len(value), "truncated": True}
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
        return {"type": type(value).__name__, "chars": len(text), "truncated": True}
