from __future__ import annotations

from hashlib import sha256
from typing import Any, Iterable

from .state import EvidenceCandidate


def candidate_identity(payload: dict[str, Any]) -> str:
    for key in ("source_id", "chunk_id", "original_chunk_id", "parent_chunk_id", "id"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    # 跨轮去重不能依赖本轮排名；缺少索引 ID 时使用内容和位置生成稳定指纹。
    identity = "|".join(
        [
            str(payload.get("content") or "").strip(),
            str(payload.get("section_path") or "").strip(),
            str(payload.get("page_number") or "").strip(),
            str(payload.get("chunk_type") or "text").strip(),
        ]
    )
    return f"candidate-{sha256(identity.encode('utf-8')).hexdigest()[:20]}"


def merge_candidates(
    current: dict[str, EvidenceCandidate],
    incoming: Iterable[dict[str, Any]],
    *,
    target_need_id: str,
    max_items: int,
    max_total_context_chars: int,
) -> tuple[dict[str, EvidenceCandidate], int, int]:
    merged = {key: value.model_copy(deep=True) for key, value in current.items()}
    total_context_chars = sum(len(candidate.content) for candidate in merged.values())
    new_count = 0
    duplicate_count = 0
    for raw_item in incoming:
        payload = dict(raw_item or {})
        candidate_id = candidate_identity(payload)
        existing = merged.get(candidate_id)
        if existing is not None:
            duplicate_count += 1
            existing.appearance_count += 1
            if target_need_id not in existing.matched_need_ids:
                existing.matched_need_ids.append(target_need_id)
            continue
        if len(merged) >= max_items:
            continue
        payload["candidate_id"] = candidate_id
        payload["matched_need_ids"] = [target_need_id]
        candidate = EvidenceCandidate.model_validate(payload)
        if total_context_chars + len(candidate.content) > max_total_context_chars:
            # 候选池是后续草稿和校验的输入上界；超额候选不入池，避免跨轮累计绕过上下文预算。
            continue
        merged[candidate_id] = candidate
        total_context_chars += len(candidate.content)
        new_count += 1
    return merged, new_count, duplicate_count
