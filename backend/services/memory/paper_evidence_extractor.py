from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.memory.research_profile_generator import ResearchProfileGenerator

logger = logging.getLogger(__name__)

PAPER_EVIDENCE_EXTRACTOR_VERSION = "llm_paper_evidence_v1"
PAPER_EVIDENCE_SCHEMA_VERSION = "paper_evidence_card_v1"

CONCEPT_TYPES = {
    "main_research_area",
    "research_object",
    "method",
    "task",
    "application_domain",
    "technical_concept",
    "evaluation_focus",
    "system_type",
}
LOW_INFORMATION_LABELS = {"paper", "papers", "method", "methods", "model", "models", "framework", "system", "approach", "analysis"}


class PaperEvidenceExtractor:
    """把单篇论文转换成画像构建用 evidence card。

    这里的职责不是总结论文，而是生成可泛化、可校验、可缓存的研究语义证据；
    后续画像聚合只能消费 evidence card，不能再从 title/abstract 直接切 topic。
    """

    def __init__(self, generation_service: Optional[Any] = None):
        self.generation_service = generation_service

    @staticmethod
    def normalize_paper(paper: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        payload = dict(paper or {})
        return {
            "arxiv_id": str(payload.get("arxiv_id") or payload.get("id") or "").strip(),
            "title": str(payload.get("title") or "").strip(),
            "abstract": str(payload.get("abstract") or payload.get("summary") or "").strip(),
            "categories": ResearchProfileGenerator.normalize_preferred_categories(payload.get("categories"), limit=20),
            "published_date": str(payload.get("published_date") or payload.get("published") or "").strip(),
        }

    def extract(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self.normalize_paper(paper)
        if not normalized.get("arxiv_id"):
            return self._error_card(normalized, "missing_arxiv_id", extraction_confidence=0.0)
        if not normalized.get("abstract"):
            # 缺少摘要时不调用 LLM 编造概念，只保留低置信度基础卡供回溯。
            return self._error_card(normalized, "missing_abstract", extraction_confidence=0.15)

        prompt = self._build_prompt(normalized)
        for attempt in range(2):
            try:
                payload = self._call_llm(prompt)
                parsed = self._parse_json_object(payload)
                card = self._validate_and_normalize(parsed, normalized)
                if card.get("schema_valid"):
                    return card
            except Exception as exc:
                logger.warning(
                    "Paper evidence extraction failed arxiv_id=%s attempt=%s error=%s",
                    normalized.get("arxiv_id"),
                    attempt + 1,
                    exc,
                )
                last_error = str(exc)
        return self._error_card(normalized, last_error if "last_error" in locals() else "llm_extraction_failed")

    def _call_llm(self, prompt: str) -> str:
        if self.generation_service is None:
            raise RuntimeError("generation_service_unavailable")
        complete = getattr(self.generation_service, "complete_with_qwen", None)
        if callable(complete):
            return str(
                complete(
                    prompt,
                    task_type="paper_profile_evidence",
                    enable_thinking=False,
                )
                or ""
            )
        generate = getattr(self.generation_service, "generate", None)
        if callable(generate):
            return str(generate(provider="qwen", query=prompt, search_results=[], task_type="paper_profile_evidence") or "")
        raise RuntimeError("generation_service_has_no_supported_method")

    def _build_prompt(self, paper: Dict[str, Any]) -> str:
        categories = ", ".join(paper.get("categories") or [])
        return (
            "你是研究画像证据抽取器。请只输出严格 JSON，不要输出 Markdown。\n"
            "任务：不是总结论文，而是为用户研究画像抽取可泛化概念。\n"
            "禁止：输出完整论文标题作为 topic；输出 arXiv 分类；输出 paper/method/model/framework 等过泛词；"
            "输出语序不自然的 n-gram；输出只对单篇论文有意义的系统名，除非它代表明确技术方向。\n"
            "JSON schema:\n"
            "{\n"
            '  "main_research_area": "string",\n'
            '  "research_objects": ["string"],\n'
            '  "methods": ["string"],\n'
            '  "tasks": ["string"],\n'
            '  "application_domains": ["string"],\n'
            '  "technical_concepts": ["string"],\n'
            '  "evaluation_focus": ["string"],\n'
            '  "system_type": "string",\n'
            '  "candidate_concepts": [\n'
            '    {"label": "string", "type": "technical_concept|method|task|research_object|application_domain|evaluation_focus|system_type|main_research_area", "confidence": 0.0, "evidence_text": "string", "source": "llm", "whether_generalizable": true}\n'
            "  ],\n"
            '  "excluded_concepts": [{"label": "string", "reason": "string"}],\n'
            '  "extraction_confidence": 0.0\n'
            "}\n\n"
            f"arxiv_id: {paper.get('arxiv_id')}\n"
            f"title: {paper.get('title')}\n"
            f"categories: {categories}\n"
            f"published_date: {paper.get('published_date')}\n"
            f"abstract: {paper.get('abstract')[:4000]}\n"
        )

    @staticmethod
    def _parse_json_object(text: str) -> Dict[str, Any]:
        raw = str(text or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
            raw = re.sub(r"```$", "", raw).strip()
        if not raw.startswith("{"):
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                raise ValueError("llm_output_not_json")
            raw = match.group(0)
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("llm_json_not_object")
        return parsed

    def _validate_and_normalize(self, payload: Dict[str, Any], paper: Dict[str, Any]) -> Dict[str, Any]:
        candidate_concepts = self._normalize_candidate_concepts(payload.get("candidate_concepts"), paper)
        if not candidate_concepts:
            raise ValueError("candidate_concepts_empty_after_validation")

        return {
            "schema_version": PAPER_EVIDENCE_SCHEMA_VERSION,
            "arxiv_id": paper.get("arxiv_id"),
            "title": paper.get("title"),
            "abstract": paper.get("abstract"),
            "categories": paper.get("categories") or [],
            "published_date": paper.get("published_date"),
            "main_research_area": self._clean_label(payload.get("main_research_area"), paper),
            "research_objects": self._normalize_string_list(payload.get("research_objects"), paper),
            "methods": self._normalize_string_list(payload.get("methods"), paper),
            "tasks": self._normalize_string_list(payload.get("tasks"), paper),
            "application_domains": self._normalize_string_list(payload.get("application_domains"), paper),
            "technical_concepts": self._normalize_string_list(payload.get("technical_concepts"), paper),
            "evaluation_focus": self._normalize_string_list(payload.get("evaluation_focus"), paper),
            "system_type": self._clean_label(payload.get("system_type"), paper),
            "candidate_concepts": candidate_concepts,
            "excluded_concepts": self._normalize_excluded_concepts(payload.get("excluded_concepts")),
            "extractor_version": PAPER_EVIDENCE_EXTRACTOR_VERSION,
            "extraction_confidence": self._coerce_confidence(payload.get("extraction_confidence"), default=0.65),
            "schema_valid": True,
            "error_message": "",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def _normalize_candidate_concepts(self, values: Any, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        concepts: List[Dict[str, Any]] = []
        source = values if isinstance(values, list) else []
        for item in source:
            if not isinstance(item, dict):
                continue
            label = self._clean_label(item.get("label"), paper)
            if not label:
                continue
            concept_type = str(item.get("type") or "technical_concept").strip()
            if concept_type not in CONCEPT_TYPES:
                concept_type = "technical_concept"
            generalizable = bool(item.get("whether_generalizable", True))
            if not generalizable:
                continue
            concepts.append(
                {
                    "label": label,
                    "type": concept_type,
                    "confidence": self._coerce_confidence(item.get("confidence"), default=0.55),
                    "evidence_text": str(item.get("evidence_text") or "").strip()[:500],
                    "source": str(item.get("source") or "llm").strip() or "llm",
                    "whether_generalizable": True,
                }
            )
        return self._dedupe_concepts(concepts)

    def _normalize_string_list(self, values: Any, paper: Dict[str, Any], limit: int = 12) -> List[str]:
        source = values if isinstance(values, list) else [values] if values else []
        normalized: List[str] = []
        for item in source:
            label = self._clean_label(item, paper)
            if label and label not in normalized:
                normalized.append(label)
            if len(normalized) >= limit:
                break
        return normalized

    @staticmethod
    def _normalize_excluded_concepts(values: Any) -> List[Dict[str, str]]:
        excluded: List[Dict[str, str]] = []
        for item in values if isinstance(values, list) else []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            reason = str(item.get("reason") or "").strip()
            if label:
                excluded.append({"label": label, "reason": reason})
        return excluded[:20]

    def _clean_label(self, value: Any, paper: Dict[str, Any]) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text:
            return ""
        if text == paper.get("title"):
            return ""
        if ResearchProfileGenerator.looks_like_arxiv_category(text) or text.lower().startswith("cs."):
            return ""
        if ResearchProfileGenerator.looks_like_arxiv_id(text):
            return ""
        if ResearchProfileGenerator.looks_like_full_paper_title(text):
            return ""
        if text.lower() in LOW_INFORMATION_LABELS:
            return ""
        return text

    @staticmethod
    def _coerce_confidence(value: Any, default: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = default
        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _dedupe_concepts(concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for concept in sorted(concepts, key=lambda item: float(item.get("confidence") or 0.0), reverse=True):
            key = str(concept.get("label") or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(concept)
        return deduped[:20]

    def _error_card(self, paper: Dict[str, Any], error_message: str, extraction_confidence: float = 0.2) -> Dict[str, Any]:
        return {
            "schema_version": PAPER_EVIDENCE_SCHEMA_VERSION,
            "arxiv_id": paper.get("arxiv_id", ""),
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "categories": paper.get("categories") or [],
            "published_date": paper.get("published_date", ""),
            "main_research_area": "",
            "research_objects": [],
            "methods": [],
            "tasks": [],
            "application_domains": [],
            "technical_concepts": [],
            "evaluation_focus": [],
            "system_type": "",
            "candidate_concepts": [],
            "excluded_concepts": [],
            "extractor_version": PAPER_EVIDENCE_EXTRACTOR_VERSION,
            "extraction_confidence": extraction_confidence,
            "schema_valid": False,
            "error_message": str(error_message or "unknown_error")[:500],
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
