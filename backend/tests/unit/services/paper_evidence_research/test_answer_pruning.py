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


# 复刻 bbd87754 第 1 版草稿：父级列表项是无引用标签，子项缩进 4 格；LLM 摘录带着行首的 "* " 和 "1."。
_NESTED_DRAFT = (
    "### 记忆生成\n"
    "*   **a) 记忆化 (Memorize)**：\n"
    "    *   系统处理原始对话，将其转换为初始的认知图。\n"
    "    *   它将情节痕迹转化为图中的节点 [source:54]。\n\n"
    "*   **c) 推理 (Reasoning)**：\n"
    "    *   **工具集操作**：\n"
    "        1.  **Anchoring（锚定）**：将查询映射到入口节点 [source:54]。\n"
    "        2.  **Traversal（遍历）**：在图中进行多跳导航 [source:54]。"
)


def test_supported_claims_with_leading_list_markers_still_count_as_covered() -> None:
    # bbd87754：残留检查逐行比对时先去掉了行首 "* "，带 "* " 的主张一条都对不上，整篇被误判为漏句。
    pruned = prune_unsupported_claims(
        _NESTED_DRAFT,
        removed_claim_texts=["* 系统处理原始对话，将其转换为初始的认知图。"],
        supported_claim_texts=[
            "* 它将情节痕迹转化为图中的节点。",
            "1. Anchoring（锚定）：将查询映射到入口节点。",
            "2. Traversal（遍历）：在图中进行多跳导航。",
        ],
        allowed_citation_ids={"54"},
    )
    assert pruned is not None
    assert "系统处理原始对话" not in pruned


def test_claim_spanning_several_list_items_counts_as_covered() -> None:
    draft = "*   **巩固**：\n    *   离线合并重复事实。\n    *   调和冲突主张 [source:54]。\n\n总结：流程有两层。"
    pruned = prune_unsupported_claims(
        draft,
        removed_claim_texts=["总结：流程有两层。"],
        supported_claim_texts=["* 巩固： * 离线合并重复事实。 * 调和冲突主张。"],
        allowed_citation_ids={"54"},
    )
    assert pruned == "*   **巩固**：\n    *   离线合并重复事实。\n    *   调和冲突主张 [source:54]。"


def test_children_of_removed_list_items_are_dedented_instead_of_becoming_code_blocks() -> None:
    pruned = prune_unsupported_claims(
        _NESTED_DRAFT,
        removed_claim_texts=[
            "a) 记忆化 (Memorize)：",
            "系统处理原始对话，将其转换为初始的认知图。",
            "工具集操作：",
        ],
        supported_claim_texts=[
            "它将情节痕迹转化为图中的节点。",
            "Anchoring（锚定）：将查询映射到入口节点。",
            "Traversal（遍历）：在图中进行多跳导航。",
        ],
        allowed_citation_ids={"54"},
    )
    # 原父项被删，缩进 4 格的子项放在段首会被渲染成代码块；挪到最近保留的上级内容列。
    assert pruned == (
        "### 记忆生成\n"
        "*   它将情节痕迹转化为图中的节点 [source:54]。\n\n"
        "*   **c) 推理 (Reasoning)**：\n"
        "    1.  **Anchoring（锚定）**：将查询映射到入口节点 [source:54]。\n"
        "    2.  **Traversal（遍历）**：在图中进行多跳导航 [source:54]。"
    )
