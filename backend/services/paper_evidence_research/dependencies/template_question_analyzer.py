"""按意图模板生成证据需求账本的兜底分析器。

它只会提出"套路需求"，永远覆盖不了问题的长尾；价值在于保证 LLM 分析器失败时研究图
仍能以合法账本启动，并天然充当评测体系里的最弱基线。关键词表与
services/intent/intent_service.py 的 MAIN_INTENTS 语族对齐，但刻意保持本地自包含，
避免研究引擎与意图服务产生运行时耦合。
"""

from __future__ import annotations

from typing import Any

# 关键词按意图分桶；匹配时按出现次数计分，同分时先到先得，保证确定性。
_INTENT_KEYWORDS: dict[str, list[str]] = {
    "method_flow": [
        "method", "approach", "framework", "architecture", "algorithm",
        "pipeline", "workflow", "方法", "流程", "框架", "算法", "架构",
    ],
    "experiment_setup": [
        "experiment", "evaluation", "dataset", "benchmark", "metric",
        "hyperparameter", "setup", "实验", "数据集", "评估", "基准", "指标", "超参",
    ],
    "result_analysis": [
        "result", "performance", "ablation", "score", "effect", "improve",
        "结果", "性能", "消融", "效果", "提升", "分数",
    ],
    "comparison": [
        "compare", "comparison", "versus", "baseline", "outperform",
        "比较", "对比", "优于", "基线", "差异",
    ],
    "limitation": [
        "limitation", "weakness", "shortcoming", "failure", "fail",
        "局限", "缺点", "失败", "不足", "缺陷",
    ],
    "contribution": [
        "contribution", "novelty", "innovation", "创新", "贡献", "首创",
    ],
    "paper_overview": [
        "overview", "summary", "summarize", "what is this paper about",
        "概述", "总结", "梗概", "讲了什么",
    ],
}

# 每个意图给出固定的核心/支撑需求模板；{question} 占位符会被原问题替换。
_NEED_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "method_flow": [
        {"description": "论文提出的核心方法流程与关键模块", "importance": "core"},
        {"description": "方法各模块的输入输出与衔接方式", "importance": "supporting"},
    ],
    "experiment_setup": [
        {"description": "实验使用的数据集与评估指标设置", "importance": "core"},
        {"description": "训练细节与关键超参数配置", "importance": "supporting"},
    ],
    "result_analysis": [
        {"description": "主要实验结果与关键数字证据", "importance": "core"},
        {"description": "结果差异背后的原因分析", "importance": "supporting"},
    ],
    "comparison": [
        {"description": "与基线方法的直接对比结论", "importance": "core"},
        {"description": "支撑对比结论的具体数值或表格证据", "importance": "supporting"},
    ],
    "limitation": [
        {"description": "论文自述的局限性与失效场景", "importance": "core"},
        {"description": "论文提出的未来工作方向", "importance": "supporting"},
    ],
    "contribution": [
        {"description": "论文声称的核心贡献及其支撑证据", "importance": "core"},
    ],
    "paper_overview": [
        {"description": "论文要解决的问题与核心思路", "importance": "core"},
    ],
    "other": [
        {"description": "回答问题「{question}」所需的核心证据与关键事实", "importance": "core"},
        {"description": "与问题相关的背景定义与术语解释", "importance": "supporting"},
    ],
}


def classify_question_intent(question: str) -> str:
    lowered = (question or "").lower()
    best_intent = "other"
    best_hits = 0
    for intent, keywords in _INTENT_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in lowered)
        if hits > best_hits:
            best_intent = intent
            best_hits = hits
    return best_intent


def _render_description(description: str, question: str) -> str:
    # 用 replace 而不是 str.format：问题文本里可能出现花括号，format 会直接抛错。
    return description.replace("{question}", question)


class TemplateQuestionAnalyzer:
    """规则兜底问题分析器：意图模板 -> 固定 needs，保证账本永远合法可启动。"""

    def analyze(self, request: Any) -> dict[str, Any]:
        question = str(getattr(request, "original_question", "") or "").strip()
        intent = classify_question_intent(question)
        evidence_needs = [
            {
                "need_id": f"need-{intent}-{index}",
                "description": _render_description(template["description"], question),
                "importance": template["importance"],
                "directly_required_by_question": template["importance"] == "core",
            }
            for index, template in enumerate(_NEED_TEMPLATES.get(intent, _NEED_TEMPLATES["other"]))
        ]
        return {
            "research_question": question,
            "evidence_needs": evidence_needs,
            "analyzer_source": "template",
            "intent": intent,
        }
