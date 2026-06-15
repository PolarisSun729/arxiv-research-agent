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
        table_numeric_values = self.collect_table_numeric_values(normalized_sources)
        answer_numeric_values = self.extract_answer_numeric_values(normalized_answer)
        unsupported_numeric_values = self.unsupported_answer_numeric_values(
            answer_numeric_values,
            table_numeric_values,
        ) if table_numeric_values else []

        warnings: List[str] = []
        if not normalized_answer:
            warnings.append("answer_empty")
        if not normalized_sources:
            warnings.append("sources_empty")
        if missing_cited_source_ids:
            warnings.append("cited_source_missing")
        if answer_has_specific_claims and not normalized_sources:
            warnings.append("specific_claim_without_source")
        if unsupported_numeric_values:
            warnings.append("answer_numeric_value_not_in_table_evidence")

        insufficient_evidence = bool(generation_insufficient_evidence)
        if not normalized_answer or not normalized_sources:
            insufficient_evidence = True
        if answer_has_specific_claims and not normalized_sources:
            insufficient_evidence = True
        # 如果结构化表格证据已经进入上下文，答案中的具体小数/百分比必须能回溯到 cell 或 computed_value。
        if unsupported_numeric_values:
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
            "unsupported_claims": claim_items if unsupported_numeric_values or not normalized_sources else [],
            "answer_has_specific_claims": answer_has_specific_claims,
            "insufficient_evidence": insufficient_evidence,
            "warnings": warnings,
            "table_cell_evidence_count": self.count_table_cell_evidence(normalized_sources),
            "answer_numeric_values": answer_numeric_values,
            "table_numeric_evidence_values": sorted(table_numeric_values),
            "unsupported_numeric_values": unsupported_numeric_values,
            "verification_debug": {
                "rule_set": "lightweight_v2_table_cell_numeric",
                "checked_answer_non_empty": True,
                "checked_sources_non_empty": True,
                "checked_cited_source_ids_exist": True,
                "checked_specific_claims_require_sources": True,
                "checked_table_cell_evidence": True,
                "checked_answer_numbers_against_table_evidence": bool(table_numeric_values),
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

    @classmethod
    def collect_table_numeric_values(cls, sources: List[Dict[str, Any]]) -> set[str]:
        values: set[str] = set()
        for source in sources:
            if not isinstance(source, dict):
                continue
            evidence = source.get("table_structured_evidence")
            if isinstance(evidence, dict):
                for cell in evidence.get("matched_cells") or []:
                    if isinstance(cell, dict):
                        values.update(cls.numeric_tokens_from_value(cell.get("raw_value")))
                        values.update(cls.numeric_tokens_from_value(cell.get("normalized_value")))
                values.update(cls.numeric_tokens_from_value(evidence.get("computed_value")))
            for citation in source.get("table_cell_citations") or []:
                if isinstance(citation, dict):
                    values.update(cls.numeric_tokens_from_value(citation.get("raw_value")))
                    values.update(cls.numeric_tokens_from_value(citation.get("normalized_value")))
        return values

    @classmethod
    def count_table_cell_evidence(cls, sources: List[Dict[str, Any]]) -> int:
        count = 0
        for source in sources:
            if not isinstance(source, dict):
                continue
            citations = source.get("table_cell_citations")
            if isinstance(citations, list):
                count += len([item for item in citations if isinstance(item, dict)])
                continue
            evidence = source.get("table_structured_evidence")
            if isinstance(evidence, dict):
                count += len([item for item in (evidence.get("matched_cells") or []) if isinstance(item, dict)])
        return count

    @classmethod
    def extract_answer_numeric_values(cls, answer: str) -> List[str]:
        # 只校验百分比和小数，避免把 Table 2、Source 1 这类编号误判成数值主张。
        values: List[str] = []
        for match in re.findall(r"(?<![A-Za-z0-9_-])\d+\.\d+\s*%?|\d+\s*%", answer or ""):
            for token in cls.numeric_tokens_from_value(match):
                if token not in values:
                    values.append(token)
        return values

    @classmethod
    def unsupported_answer_numeric_values(cls, answer_values: List[str], evidence_values: set[str]) -> List[str]:
        unsupported: List[str] = []
        for value in answer_values:
            if cls.numeric_value_supported(value, evidence_values):
                continue
            unsupported.append(value)
        return unsupported

    @staticmethod
    def numeric_tokens_from_value(value: Any) -> set[str]:
        text = str(value if value is not None else "").strip()
        if not text:
            return set()
        tokens: set[str] = set()
        for match in re.findall(r"-?\d+(?:\.\d+)?", text):
            try:
                number = float(match)
            except ValueError:
                continue
            tokens.add(EvidenceVerifier.normalize_numeric_token(number))
        return tokens

    @staticmethod
    def numeric_value_supported(value: str, evidence_values: set[str]) -> bool:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        for evidence in evidence_values:
            try:
                evidence_number = float(evidence)
            except (TypeError, ValueError):
                continue
            if abs(number - evidence_number) <= 1e-6:
                return True
        return False

    @staticmethod
    def normalize_numeric_token(value: float) -> str:
        rounded = round(float(value), 6)
        text = f"{rounded:.6f}".rstrip("0").rstrip(".")
        return text or "0"
