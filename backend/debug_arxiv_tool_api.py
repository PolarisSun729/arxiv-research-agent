from typing import Any, Dict

from fastapi import Body, FastAPI

from tools.arxiv_tools import search_arxiv_raw, search_arxiv_structured

app = FastAPI(title="arXiv Tool Debug API")


@app.post("/tools/arxiv/raw")
async def call_arxiv_raw(payload: Dict[str, Any] = Body(...)):
    """
    模拟 Agent 调用 search_arxiv_raw 工具。
    """
    return search_arxiv_raw(**payload)


@app.post("/tools/arxiv/structured")
async def call_arxiv_structured(payload: Dict[str, Any] = Body(...)):
    """
    模拟 Agent 调用 search_arxiv_structured 工具。
    """
    return search_arxiv_structured(**payload)


@app.post("/tools/call")
async def call_tool(payload: Dict[str, Any] = Body(...)):
    """
    更接近 Agent 的统一工具调用格式：
    {
      "tool_name": "search_arxiv_structured",
      "arguments": {...}
    }
    """
    tool_name = payload.get("tool_name")
    arguments = payload.get("arguments") or {}

    if tool_name == "search_arxiv_raw":
        return search_arxiv_raw(**arguments)

    if tool_name == "search_arxiv_structured":
        return search_arxiv_structured(**arguments)

    return {
        "ok": False,
        "tool_name": tool_name,
        "summary": "未知工具",
        "data": None,
        "trace": {
            "raw_inputs": payload,
        },
        "error": {
            "code": "unknown_tool",
            "message": f"Unknown tool: {tool_name}",
        },
    }