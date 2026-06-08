from __future__ import annotations

import re
from typing import Any, Dict, List


class EvidenceVerifier:
    """对生成后的答案做轻量证据闭环校验。"""

    def verify(
        self,
        *,
        answer: str,
        sources: List[Dict[str, Any]],
        cited_source_ids: List[str] | None = None,
        claims: List[Dict[str, Any]] | None = None,
        generation_insufficient_evidence: bool = False,
    ) -> Dict[str, Any]:
        normalized_answer = str(answer or "").strip()
        normalized_sources = list(sources or [])
        source_ids = [
            str(source.get("source_id", "") or "").strip()
            for source in normalized_sources
            if str(source.get("source_id", "") or "").strip()
        ]
        cited_ids = [str(item).strip() for item in (cited_source_ids or []) if str(item).strip()]
        missing_cited_source_ids = [item for item in cited_ids if item not in set(source_ids)]
        claim_items = list(claims or self.extract_claims(normalized_answer))
        answer_has_specific_claims = bool(claim_items) or self.answer_has_specific_claims(normalized_answer)

        warnings: List[str] = []
        if not normalized_answer:
            warnings.append("answer_empty")
        if not normalized_sources:
            warnings.append("sources_empty")
        if missing_cited_source_ids:
            warnings.append("cited_source_missing")
        if answer_has_specific_claims and not normalized_sources:
            warnings.append("specific_claim_without_source")

        insufficient_evidence = bool(generation_insufficient_evidence)
        if not normalized_answer or not normalized_sources:
            insufficient_evidence = True
        if answer_has_specific_claims and not normalized_sources:
            insufficient_evidence = True

        if insufficient_evidence:
            status = "insufficient_evidence"
        elif warnings:
            status = "warning"
        else:
            status = "passed"

        return {
            "status": status,
            "answer_empty": not bool(normalized_answer),
            "source_count": len(normalized_sources),
            "source_ids": source_ids,
            "cited_source_ids": cited_ids,
            "missing_cited_source_ids": missing_cited_source_ids,
            "claims": claim_items,
            "unsupported_claims": [] if normalized_sources else claim_items,
            "answer_has_specific_claims": answer_has_specific_claims,
            "insufficient_evidence": insufficient_evidence,
            "warnings": warnings,
            "verification_debug": {
                "rule_set": "lightweight_v1",
                "checked_answer_non_empty": True,
                "checked_sources_non_empty": True,
                "checked_cited_source_ids_exist": True,
                "checked_specific_claims_require_sources": True,
            },
        }

    def apply_answer_guardrail(self, answer: str, verification_result: Dict[str, Any]) -> str:
        normalized_answer = str(answer or "").strip()
        if not verification_result.get("insufficient_evidence"):
            return normalized_answer

        # 证据不足时不伪装成确定回答；有原答案则保留但显式标注证据限制，方便用户判断可信度。
        prefix = "当前检索证据不足，无法给出充分支撑的确定性结论。"
        if not normalized_answer:
            return prefix
        if "当前检索证据不足" in normalized_answer or "证据不足" in normalized_answer:
            return normalized_answer
        return f"{prefix}\n\n{normalized_answer}"

    @staticmethod
    def extract_claims(answer: str) -> List[Dict[str, Any]]:
        claims: List[Dict[str, Any]] = []
        for index, sentence in enumerate(re.split(r"(?<=[。！？.!?])\s+", answer or ""), start=1):
            text = sentence.strip()
            if text and EvidenceVerifier.answer_has_specific_claims(text):
                claims.append({"claim_id": f"claim-{index}", "text": text})
        return claims[:12]

    @staticmethod
    def answer_has_specific_claims(answer: str) -> bool:
        patterns = [
            r"\d+(?:\.\d+)?\s*(?:%|percent|points?|epochs?|layers?|samples?|tokens?)",
            r"\b(?:outperform|improve|reduce|increase|achieve|conclude|show|demonstrate)s?\b",
            r"(?:优于|提升|降低|增加|达到|证明|表明|结论|实验|结果|步骤|方法|指标)",
        ]
        return any(re.search(pattern, answer or "", flags=re.IGNORECASE) for pattern in patterns)
