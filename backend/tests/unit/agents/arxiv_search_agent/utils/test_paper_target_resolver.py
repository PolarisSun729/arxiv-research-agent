from __future__ import annotations

from backend.tests.lightweight_imports import load_arxiv_utils_module


reference_resolver = load_arxiv_utils_module("paper_reference_resolver")
target_resolver = load_arxiv_utils_module("paper_target_resolver")


def _hint(message: str) -> dict:
    return reference_resolver._resolve_paper_reference(message, context={})


def _paper(arxiv_id: str, title: str) -> dict[str, object]:
    return {"arxiv_id": arxiv_id, "title": title, "authors": ["A"]}


def test_step4_typical_reference_messages_resolve_without_implicit_fallbacks() -> None:
    papers = [
        _paper("2401.00001", "First"),
        _paper("2401.00002", "Second"),
        _paper("2401.00003", "Third"),
    ]
    cases = [
        ("总结第二篇论文", {"last_papers": papers}, "paper_summary", "resolved", "2401.00002", False, "ordinal"),
        ("讲一下这篇论文", {"selected_paper": _paper("2401.00002", "Second")}, "paper_qa", "resolved", "2401.00002", False, "context_paper"),
        ("最后一篇怎么样", {"current_papers": papers}, "paper_qa", "resolved", "2401.00003", False, "last_item"),
        ("收藏第 3 篇", {"last_papers": papers}, "preference_action", "resolved", "2401.00003", True, "ordinal"),
        ("不喜欢这个", {"selected_paper": _paper("2401.00002", "Second")}, "preference_action", "resolved", "2401.00002", True, "context_paper"),
        ("2404.03868 这篇讲什么", {}, "paper_qa", "resolved", "2404.03868", False, "arxiv_id"),
        ("刚才那篇", {"recent_opened_paper": _paper("2401.00003", "Third")}, "paper_qa", "resolved", "2401.00003", False, "context_paper"),
        ("推荐里的第二篇", {"recommendations": papers, "last_papers": [_paper("2501.00001", "Search First"), _paper("2501.00002", "Search Second")]}, "paper_qa", "resolved", "2401.00002", False, "ordinal"),
        ("搜索结果第三篇", {"last_papers": papers, "recommendations": [_paper("2501.00001", "Rec First"), _paper("2501.00002", "Rec Second"), _paper("2501.00003", "Rec Third")]}, "paper_qa", "resolved", "2401.00003", False, "ordinal"),
    ]

    for message, context, action_type, status, arxiv_id, requires_confirmation, reference_type in cases:
        result = target_resolver.resolve_paper_target(
            reference_hint=_hint(message),
            message=message,
            context=context,
            action_type=action_type,
        )

        assert result["status"] == status, message
        assert result["arxiv_id"] == arxiv_id, message
        assert result["requires_confirmation"] is requires_confirmation, message
        assert result["reference_hint"]["reference_type"] == reference_type, message


def test_step4_no_target_expression_does_not_use_selected_or_first_paper() -> None:
    result = target_resolver.resolve_paper_target(
        reference_hint=_hint("总结一下"),
        message="总结一下",
        context={
            "selected_paper": _paper("2401.00001", "Selected"),
            "last_papers": [_paper("2401.00002", "First Result")],
        },
        action_type="paper_summary",
    )

    assert result["status"] == "need_clarification"
    assert result["resolution_reason"] == "reference_unknown"
    assert result["arxiv_id"] is None
    assert result["target"] is None


def test_step4_ordinal_without_search_results_needs_clarification() -> None:
    result = target_resolver.resolve_paper_target(
        reference_hint=_hint("第二篇"),
        message="第二篇",
        context={},
        action_type="paper_qa",
    )

    assert result["status"] == "need_clarification"
    assert result["resolution_reason"] == "ordinal_context_missing"
    assert result["candidates"] == []


def test_arxiv_id_resolves_exact_context_match() -> None:
    result = target_resolver.resolve_paper_target(
        reference_hint=_hint("总结 arXiv 2401.00002"),
        message="总结 arXiv 2401.00002",
        context={"last_papers": [_paper("2401.00001", "First"), _paper("2401.00002", "Second")]},
        action_type="paper_qa",
    )

    assert result["status"] == "resolved"
    assert result["final_target_resolved"] is True
    assert result["arxiv_id"] == "2401.00002"
    assert result["target"]["title"] == "Second"
    assert result["target_resolution"]["reference_hint"]["reference_type"] == "arxiv_id"


def test_ordinal_resolves_from_current_display_list() -> None:
    result = target_resolver.resolve_paper_target(
        reference_hint=_hint("讲一下第二篇论文"),
        message="讲一下第二篇论文",
        context={"current_papers": [_paper("2401.00001", "First"), _paper("2401.00002", "Second")]},
        action_type="paper_qa",
    )

    assert result["status"] == "resolved"
    assert result["arxiv_id"] == "2401.00002"
    assert result["resolution_reason"] == "ordinal_unique_candidate"


def test_last_item_resolves_from_current_list_tail() -> None:
    result = target_resolver.resolve_paper_target(
        reference_hint=_hint("最后一篇讲什么"),
        message="最后一篇讲什么",
        context={"displayed_papers": [_paper("2401.00001", "First"), _paper("2401.00003", "Last")]},
        action_type="paper_qa",
    )

    assert result["status"] == "resolved"
    assert result["arxiv_id"] == "2401.00003"
    assert result["target"]["rank"] == 2


def test_context_paper_uses_selected_or_recent_focus_not_first_list_item() -> None:
    result = target_resolver.resolve_paper_target(
        reference_hint=_hint("这篇论文的方法是什么"),
        message="这篇论文的方法是什么",
        context={
            "last_papers": [_paper("2401.00001", "First"), _paper("2401.00002", "Second")],
            "selected_paper": _paper("2401.00002", "Second"),
        },
        action_type="paper_qa",
    )

    assert result["status"] == "resolved"
    assert result["arxiv_id"] == "2401.00002"
    assert result["resolution_reason"] == "context_focus_unique_candidate"


def test_context_paper_prefers_frontend_visible_paper_over_stale_backend_focus() -> None:
    result = target_resolver.resolve_paper_target(
        reference_hint=_hint("这篇论文的方法是什么"),
        message="这篇论文的方法是什么",
        context={
            "selected_paper": _paper("2401.00001", "Stale backend focus"),
            "frontend_visible_paper": _paper("2401.00002", "Currently visible"),
        },
        action_type="paper_qa",
    )

    assert result["status"] == "resolved"
    assert result["arxiv_id"] == "2401.00002"
    assert result["resolution_reason"] == "frontend_visible_paper_unique_candidate"


def test_all_ordinal_references_use_the_ordered_paper_list() -> None:
    papers = [
        _paper("2401.00001", "First"),
        _paper("2401.00002", "Second"),
        _paper("2401.00003", "Third"),
    ]

    for message, expected_id in (("第一篇论文", "2401.00001"), ("第三篇论文", "2401.00003")):
        result = target_resolver.resolve_paper_target(
            reference_hint=_hint(message),
            message=message,
            context={"last_papers": papers},
            action_type="paper_qa",
        )

        assert result["status"] == "resolved"
        assert result["arxiv_id"] == expected_id
        assert result["reference_hint"]["reference_type"] == "ordinal"


def test_ordinal_without_candidate_list_needs_clarification() -> None:
    result = target_resolver.resolve_paper_target(
        reference_hint=_hint("讲一下第二篇论文"),
        message="讲一下第二篇论文",
        context={},
        action_type="paper_qa",
    )

    assert result["status"] == "need_clarification"
    assert result["resolution_reason"] == "ordinal_context_missing"
    assert result["candidates"] == []
    assert result["arxiv_id"] is None


def test_search_and_recommendation_lists_with_same_ordinal_need_confirmation() -> None:
    result = target_resolver.resolve_paper_target(
        reference_hint=_hint("讲一下第二篇论文"),
        message="讲一下第二篇论文",
        context={
            "last_papers": [_paper("2401.00001", "Search First"), _paper("2401.00002", "Search Second")],
            "recommendations": [_paper("2501.00001", "Rec First"), _paper("2501.00002", "Rec Second")],
        },
        action_type="paper_qa",
    )

    assert result["status"] == "need_confirmation"
    assert result["resolution_reason"] == "ordinal_multiple_candidate_lists"
    assert {item["arxiv_id"] for item in result["candidates"]} == {"2401.00002", "2501.00002"}
    assert result["recommended_candidate"]["arxiv_id"] in {"2401.00002", "2501.00002"}


def test_source_cue_selects_recommendation_list() -> None:
    result = target_resolver.resolve_paper_target(
        reference_hint=_hint("推荐里的第二篇"),
        message="推荐里的第二篇",
        context={
            "last_papers": [_paper("2401.00001", "Search First"), _paper("2401.00002", "Search Second")],
            "recommendations": [_paper("2501.00001", "Rec First"), _paper("2501.00002", "Rec Second")],
        },
        action_type="paper_qa",
    )

    assert result["status"] == "resolved"
    assert result["arxiv_id"] == "2501.00002"


def test_bare_number_returns_candidate_but_requires_confirmation() -> None:
    result = target_resolver.resolve_paper_target(
        reference_hint=_hint("喜欢 2"),
        message="喜欢 2",
        context={"last_papers": [_paper("2401.00001", "First"), _paper("2401.00002", "Second")]},
        action_type="preference_action",
    )

    assert result["status"] == "need_confirmation"
    assert result["resolution_reason"] == "bare_number_requires_confirmation"
    assert result["recommended_candidate"]["arxiv_id"] == "2401.00002"
    assert result["requires_confirmation"] is True


def test_unknown_reference_does_not_fallback_to_selected_paper() -> None:
    result = target_resolver.resolve_paper_target(
        reference_hint=_hint("总结一下"),
        message="总结一下",
        context={"selected_paper": _paper("2401.99999", "Selected")},
        action_type="paper_qa",
    )

    assert result["status"] == "need_clarification"
    assert result["resolution_reason"] == "reference_unknown"
    assert result["arxiv_id"] is None


def test_side_effect_unique_target_is_resolved_but_marked_for_confirmation() -> None:
    result = target_resolver.resolve_paper_target(
        reference_hint=_hint("喜欢这篇论文"),
        message="喜欢这篇论文",
        context={"selected_paper": _paper("2401.00001", "Selected")},
        action_type="preference_action",
    )

    assert result["status"] == "resolved"
    assert result["arxiv_id"] == "2401.00001"
    assert result["risk_level"] == "side_effect"
    assert result["requires_confirmation"] is True
