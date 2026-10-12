from __future__ import annotations

from services.paper_evidence_research.answer_pruning import prune_unsupported_claims

# 复刻 696659ee 第 2 版草稿的形态：无引用的开场概括、编号加粗小标题、项目符号、句末引用与连续引用。
# 小标题不含数字：带数字的标题会被两种提取器都当作主张，而不是留作排版。
_DRAFT = (
    "根据提供的证据，这篇论文提出的方法（CogMem）主要包含两个层面的流程。\n\n"
    "**1. 核心架构：PEC 模式与认知图构建**\n"
    "该方法将记忆建模为以人为中心的图 $G = (V, E)$ [source:11]。该图包含五种节点类型 [source:11]。\n\n"
    "**2. 记忆生成与演化**\n"
    "*   **记忆化（Memorize）：** 系统将原始对话转化为图节点。[source:54]\n"
    "*   **巩固（Consolidate）：** 离线机制合并重复事实。[source:5]\n\n"
    "总结来说，流程是从原始对话构建认知图并在图上推理 [source:54][source:13]。"
)

_SUPPORTED = [
    "该方法将记忆建模为以人为中心的图 $G = (V, E)$。",
    "该图包含五种节点类型。",
    "记忆化（Memorize）： 系统将原始对话转化为图节点。",
    "总结来说，流程是从原始对话构建认知图并在图上推理。",
]


def _prune(removed: list[str], supported: list[str] = _SUPPORTED, allowed: set[str] | None = None) -> str | None:
    return prune_unsupported_claims(
        _DRAFT,
        removed_claim_texts=removed,
        supported_claim_texts=supported,
        allowed_citation_ids=allowed if allowed is not None else {"11", "54", "13"},
    )


def test_removes_unsupported_claims_in_place_and_keeps_markdown_structure() -> None:
    pruned = _prune([
        "根据提供的证据，这篇论文提出的方法（CogMem）主要包含两个层面的流程。",
        "巩固（Consolidate）： 离线机制合并重复事实。",
    ])

    assert pruned == (
        "**1. 核心架构：PEC 模式与认知图构建**\n"
        "该方法将记忆建模为以人为中心的图 $G = (V, E)$ [source:11]。该图包含五种节点类型 [source:11]。\n\n"
        "**2. 记忆生成与演化**\n"
        "*   **记忆化（Memorize）：** 系统将原始对话转化为图节点。[source:54]\n\n"
        "总结来说，流程是从原始对话构建认知图并在图上推理 [source:54][source:13]。"
    )


def test_markers_outside_the_final_citation_list_are_dropped() -> None:
    pruned = _prune(
        [
            "根据提供的证据，这篇论文提出的方法（CogMem）主要包含两个层面的流程。",
            "巩固（Consolidate）： 离线机制合并重复事实。",
        ],
        allowed={"11", "54"},
    )

    assert pruned is not None
    assert "[source:13]" not in pruned and "[source:5]" not in pruned
    assert "推理 [source:54]。" in pruned


def test_unlocatable_claim_falls_back() -> None:
    assert _prune(["草稿里没有这句话"]) is None


def test_unverified_text_left_in_draft_falls_back() -> None:
    # 提取器漏掉了"巩固"那句：它既不受支持也没被删除，不能原位保留在正文里。
    assert _prune(["根据提供的证据，这篇论文提出的方法（CogMem）主要包含两个层面的流程。"]) is None


def test_leftover_heading_with_numbers_is_not_trusted_as_structure() -> None:
    draft = "**PEC 2 F 模式提升 30% 准确率**\n该方法将记忆建模为图 [source:11]。"
    pruned = prune_unsupported_claims(
        draft,
        removed_claim_texts=[],
        supported_claim_texts=["该方法将记忆建模为图。"],
        allowed_citation_ids={"11"},
    )
    assert pruned is None


def test_removing_claim_that_starts_inside_bold_leaves_no_dangling_markers() -> None:
    draft = "- **记忆化：** 系统将对话转为节点 [source:11]。\n- **巩固：** 离线合并事实 [source:5]。"
    pruned = prune_unsupported_claims(
        draft,
        removed_claim_texts=["巩固： 离线合并事实。"],
        supported_claim_texts=["记忆化： 系统将对话转为节点。"],
        allowed_citation_ids={"11"},
    )
    assert pruned == "- **记忆化：** 系统将对话转为节点 [source:11]。"
