from __future__ import annotations

from typing import Any, Dict, List, Tuple

from services.memory.concept_normalizer import ConceptNormalizer
from services.memory.research_profile_generator import ResearchProfileGenerator


PROFILE_REVIEWER_VERSION = "profile_reviewer_v1"


class ProfileReviewer:
    """对 generated profile 草稿做质量闸门，避免低质量主题覆盖 active 画像。"""

    def review(self, draft_profile: Dict[str, Any]) -> Dict[str, Any]:
        revised = self._clone_profile(draft_profile)
        issues: List[Dict[str, Any]] = []
        removed_topics: List[Dict[str, Any]] = []
        merged_topics: List[Dict[str, Any]] = []
        topic_evidence = revised.get("topic_evidence") if isinstance(revised.get("topic_evidence"), dict) else {}

        for field_name, canonical_field in (
            ("positive_topics", "canonical_topics"),
            ("negative_topics", "canonical_negative_topics"),
            ("recent_topics", "canonical_recent_topics"),
        ):
            clean_topics, clean_canonical, field_removed, field_merged = self._clean_topic_field(
                revised.get(field_name) or [],
                revised.get(canonical_field) or [],
                field_name=field_name,
            )
            revised[field_name] = clean_topics
            revised[canonical_field] = clean_canonical
            removed_topics.extend(field_removed)
            merged_topics.extend(field_merged)

        self._remove_positive_negative_conflicts(revised, topic_evidence, issues, removed_topics)
        self._remove_weak_negative_topics(revised, topic_evidence, issues, removed_topics)
        self._remove_read_only_recent_topics(revised, topic_evidence, issues, removed_topics)
        self._record_read_signal_review(revised, issues)
        self._filter_representative_papers(revised, topic_evidence, issues)

        if removed_topics:
            issues.append(
                {
                    "code": "topics_removed_by_quality_review",
                    "severity": "warning",
                    "message": "质量审查移除了标题碎片、过泛词或证据不足的 topic。",
                    "count": len(removed_topics),
                }
            )
        if merged_topics:
            issues.append(
                {
                    "code": "duplicate_topics_merged",
                    "severity": "info",
                    "message": "质量审查合并了重复 topic 投影。",
                    "count": len(merged_topics),
                }
            )

        quality_score = self._quality_score(revised, issues, removed_topics, draft_profile)
        approved = quality_score >= 0.45 and not self._has_severe_issue(issues)
        quality_report = {
            "reviewer_version": PROFILE_REVIEWER_VERSION,
            "approved": approved,
            "quality_score": quality_score,
            "issues": issues,
            "removed_topics": removed_topics,
            "merged_topics": merged_topics,
            "positive_topic_count": len(revised.get("positive_topics") or []),
            "negative_topic_count": len(revised.get("negative_topics") or []),
            "recent_topic_count": len(revised.get("recent_topics") or []),
            "preferred_category_count": len(revised.get("preferred_categories") or []),
            "representative_paper_count": len(revised.get("representative_papers") or []),
        }
        revised["review_status"] = {
            "approved": approved,
            "quality_score": quality_score,
            "reviewer_version": PROFILE_REVIEWER_VERSION,
        }
        revised["quality_report"] = quality_report
        return {
            "approved": approved,
            "issues": issues,
            "removed_topics": removed_topics,
            "merged_topics": merged_topics,
            "revised_profile": revised,
            "quality_score": quality_score,
            "quality_report": quality_report,
        }

    def _clean_topic_field(
        self,
        topics: List[Any],
        canonical_topics: List[Any],
        *,
        field_name: str,
    ) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        removed: List[Dict[str, Any]] = []
        merged: List[Dict[str, Any]] = []
        canonical_by_label = {str(item.get("label") or "").strip(): dict(item) for item in canonical_topics if isinstance(item, dict)}
        clean_topics: List[str] = []
        clean_canonical: List[Dict[str, Any]] = []
        seen_keys: Dict[str, str] = {}
        for raw_topic in topics:
            label = str(raw_topic or "").strip()
            if not label:
                continue
            clean_label = ConceptNormalizer.clean_label(label)
            if not clean_label:
                removed.append({"topic": label, "field": field_name, "reason": "invalid_or_low_information"})
                continue
            key = ConceptNormalizer._topic_key(clean_label)
            if key in seen_keys:
                merged.append({"topic": label, "merged_into": seen_keys[key], "field": field_name})
                continue
            seen_keys[key] = clean_label
            clean_topics.append(clean_label)
            canonical = canonical_by_label.get(label) or {"label": clean_label}
            canonical["label"] = clean_label
            clean_canonical.append(canonical)
        return clean_topics, clean_canonical, removed, merged

    def _remove_positive_negative_conflicts(
        self,
        profile: Dict[str, Any],
        topic_evidence: Dict[str, Any],
        issues: List[Dict[str, Any]],
        removed_topics: List[Dict[str, Any]],
    ) -> None:
        positive_keys = {ConceptNormalizer._topic_key(topic): topic for topic in profile.get("positive_topics") or []}
        negative_keys = {ConceptNormalizer._topic_key(topic): topic for topic in profile.get("negative_topics") or []}
        conflicts = set(positive_keys) & set(negative_keys)
        for key in conflicts:
            positive_topic = positive_keys[key]
            negative_topic = negative_keys[key]
            evidence = topic_evidence.get(positive_topic) or topic_evidence.get(negative_topic) or {}
            if float(evidence.get("positive_score") or 0.0) >= float(evidence.get("negative_score") or 0.0):
                self._remove_topic(profile, "negative", negative_topic)
                removed_topics.append({"topic": negative_topic, "field": "negative_topics", "reason": "conflicts_with_stronger_positive_evidence"})
            else:
                self._remove_topic(profile, "positive", positive_topic)
                removed_topics.append({"topic": positive_topic, "field": "positive_topics", "reason": "conflicts_with_stronger_negative_evidence"})
        if conflicts:
            issues.append({"code": "positive_negative_conflict", "severity": "warning", "message": "同一 topic 同时存在正负证据，已按净证据保留更强一侧。"})

    def _remove_weak_negative_topics(
        self,
        profile: Dict[str, Any],
        topic_evidence: Dict[str, Any],
        issues: List[Dict[str, Any]],
        removed_topics: List[Dict[str, Any]],
    ) -> None:
        for topic in list(profile.get("negative_topics") or []):
            evidence = topic_evidence.get(topic) or {}
            negative_score = float(evidence.get("negative_score") or 0.0)
            evidence_count = int(evidence.get("evidence_count") or 0)
            if ResearchProfileGenerator.looks_like_arxiv_category(topic) or negative_score < 1.8 or evidence_count <= 0:
                # 负向画像会长期影响推荐，证据不足时宁可不写入 active profile。
                self._remove_topic(profile, "negative", topic)
                removed_topics.append({"topic": topic, "field": "negative_topics", "reason": "weak_negative_evidence"})
        if removed_topics:
            weak_count = len([item for item in removed_topics if item.get("reason") == "weak_negative_evidence"])
            if weak_count:
                issues.append({"code": "weak_negative_evidence", "severity": "warning", "message": "证据不足的负向 topic 已被移除。", "count": weak_count})

    def _remove_read_only_recent_topics(
        self,
        profile: Dict[str, Any],
        topic_evidence: Dict[str, Any],
        issues: List[Dict[str, Any]],
        removed_topics: List[Dict[str, Any]],
    ) -> None:
        for topic in list(profile.get("recent_topics") or []):
            evidence = topic_evidence.get(topic) or {}
            actions = {str(item or "").strip().lower() for item in evidence.get("source_actions") or [] if str(item or "").strip()}
            recent_score = float(evidence.get("recent_score") or 0.0)
            if actions and actions <= {"read"}:
                self._remove_topic(profile, "recent", topic)
                removed_topics.append({"topic": topic, "field": "recent_topics", "reason": "read_only_recent_signal"})
            elif recent_score < 0.6:
                self._remove_topic(profile, "recent", topic)
                removed_topics.append({"topic": topic, "field": "recent_topics", "reason": "weak_recent_signal"})
        read_removed = [item for item in removed_topics if item.get("reason") in {"read_only_recent_signal", "weak_recent_signal"}]
        if read_removed:
            issues.append({"code": "recent_topic_pollution", "severity": "warning", "message": "recent_topics 已过滤普通阅读或弱近期行为造成的污染。", "count": len(read_removed)})

    def _record_read_signal_review(self, profile: Dict[str, Any], issues: List[Dict[str, Any]]) -> None:
        report = profile.get("aggregation_report") if isinstance(profile.get("aggregation_report"), dict) else {}
        action_counts = report.get("candidate_count_by_action") if isinstance(report.get("candidate_count_by_action"), dict) else {}
        if not action_counts.get("read"):
            return
        if any(issue.get("code") == "recent_topic_pollution" for issue in issues):
            return
        # read 会被记录为画像证据，但只作为弱信号；这里显式写入审查报告，方便解释为什么它没有进入 recent_topics。
        issues.append(
            {
                "code": "recent_topic_pollution",
                "severity": "info",
                "message": "普通阅读信号已被降权，未单独提升为 recent topic。",
                "count": int(action_counts.get("read") or 0),
            }
        )

    def _filter_representative_papers(
        self,
        profile: Dict[str, Any],
        topic_evidence: Dict[str, Any],
        issues: List[Dict[str, Any]],
    ) -> None:
        source_papers: List[str] = []
        for topic in profile.get("positive_topics") or []:
            source_papers.extend((topic_evidence.get(topic) or {}).get("source_papers") or [])
        allowed = ResearchProfileGenerator.normalize_representative_papers(source_papers, limit=50)
        if not allowed:
            profile["representative_papers"] = ResearchProfileGenerator.normalize_representative_papers(profile.get("representative_papers"), limit=12)
            return
        filtered = [paper for paper in ResearchProfileGenerator.normalize_representative_papers(profile.get("representative_papers"), limit=20) if paper in allowed]
        if len(filtered) < len(profile.get("representative_papers") or []):
            issues.append({"code": "representative_paper_mismatch", "severity": "info", "message": "代表论文已限制为主要正向 topic 的来源论文。"})
        profile["representative_papers"] = filtered[:12]

    @staticmethod
    def _remove_topic(profile: Dict[str, Any], bucket: str, topic: str) -> None:
        field_map = {
            "positive": ("positive_topics", "canonical_topics"),
            "negative": ("negative_topics", "canonical_negative_topics"),
            "recent": ("recent_topics", "canonical_recent_topics"),
        }
        topic_field, canonical_field = field_map[bucket]
        profile[topic_field] = [item for item in profile.get(topic_field) or [] if item != topic]
        profile[canonical_field] = [item for item in profile.get(canonical_field) or [] if not isinstance(item, dict) or item.get("label") != topic]

    @staticmethod
    def _quality_score(
        revised: Dict[str, Any],
        issues: List[Dict[str, Any]],
        removed_topics: List[Dict[str, Any]],
        draft_profile: Dict[str, Any],
    ) -> float:
        useful_signal_count = (
            len(revised.get("positive_topics") or [])
            + len(revised.get("recent_topics") or [])
            + len(revised.get("preferred_categories") or [])
            + len(revised.get("representative_papers") or [])
        )
        score = 0.5 + min(0.35, useful_signal_count * 0.05)
        score -= min(0.3, len([item for item in issues if item.get("severity") == "warning"]) * 0.06)
        original_topic_count = (
            len(draft_profile.get("positive_topics") or [])
            + len(draft_profile.get("negative_topics") or [])
            + len(draft_profile.get("recent_topics") or [])
        )
        if original_topic_count and len(removed_topics) >= original_topic_count:
            # 草稿 topic 全部被审查移除，说明本次构建质量过低，应只保留 snapshot 供排查。
            score -= 0.35
        return round(max(0.0, min(1.0, score)), 4)

    @staticmethod
    def _has_severe_issue(issues: List[Dict[str, Any]]) -> bool:
        return any(item.get("severity") == "error" for item in issues)

    @staticmethod
    def _clone_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
        import json

        return json.loads(json.dumps(profile or {}, ensure_ascii=False))
