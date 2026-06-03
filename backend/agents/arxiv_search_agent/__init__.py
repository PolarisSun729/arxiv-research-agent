from __future__ import annotations

# 这个文件是 arxiv_search_agent 包的统一导出入口。
#
# 它本身不承载复杂业务逻辑，主要职责是把这个包里最重要的：
# - 节点函数，
# - 图构建函数，
# - schema / state 模型，
# - service 运行入口，
# 统一整理后暴露给外部模块使用。
#
# 这样上层调用方只需要 import 这个包，就能拿到核心公共能力，
# 而不必分别记住每个对象具体定义在哪个文件中。

from .nodes import (
    apply_preference_action,
    build_search_tool_args,
    check_search_result,
    invoke_search_tool,
    parse_search_request,
    personalized_rank_and_annotate_papers,
    synthesize_response,
)
from .graph import build_arxiv_search_graph, route_after_parse
from .graph import export_arxiv_search_graph_mermaid
from .schemas import (
    AgentStep,
    AgentStreamEvent,
    AgentToolCall,
    ArxivSearchGraphResponse,
    ArxivSearchRequest,
    ArxivSearchResponse,
    ArxivSearchSpec,
)
from .service import run_arxiv_search_agent, stream_arxiv_search_agent
from .state import AgentState

# __all__ 明确声明这个包对外的公共 API 边界：
# 1. 状态与 schema 模型；
# 2. LangGraph 相关构建与导出函数；
# 3. 运行时节点函数；
# 4. 同步/流式执行入口。
__all__ = [
    "AgentState",
    "AgentStep",
    "AgentStreamEvent",
    "AgentToolCall",
    "ArxivSearchGraphResponse",
    "ArxivSearchRequest",
    "ArxivSearchResponse",
    "ArxivSearchSpec",
    "build_arxiv_search_graph",
    "apply_preference_action",
    "build_search_tool_args",
    "check_search_result",
    "invoke_search_tool",
    "parse_search_request",
    "personalized_rank_and_annotate_papers",
    "export_arxiv_search_graph_mermaid",
    "route_after_parse",
    "run_arxiv_search_agent",
    "stream_arxiv_search_agent",
    "synthesize_response",
]
