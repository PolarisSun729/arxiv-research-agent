"""管理端论文索引查看路由（仅管理员）。

只读展示已建索引论文的切分与解析产物，用来检查 chunk 切分效果、对照切分参数：
- GET /admin/paper-indexes                               分页列出建过索引的论文；
- GET /admin/paper-indexes/{arxiv_id}                    版本列表 + 所选版本的切分参数、统计和全部 chunk；
- GET /admin/paper-indexes/{arxiv_id}/parsed             所选版本的解析结果（逐页文本与 Docling Markdown）；
- GET /admin/paper-indexes/{arxiv_id}/assets/{chunk_id}  所选版本中图片 chunk 的原图。

版本用 build_id 查询参数指定，缺省为当前生效版本；调用方不能传任何文件路径。
接口用同步 def：chunk 文件可达数 MB，交给线程池读取，不阻塞事件循环。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from dependencies import get_paper_qa_index_store
from services.paper_qa import index_inspection as inspection

router = APIRouter(prefix="/admin/paper-indexes", tags=["admin-paper-indexes"])


def _select(store: Any, arxiv_id: str, build_id: str) -> Tuple[Dict[str, Any], list, Optional[Mapping[str, Any]], Any]:
    """读取索引主表和版本列表，选出要看的版本，并给出它的 chunk_file。"""
    index_record = store.get_paper_qa_index(arxiv_id)
    if not index_record:
        raise HTTPException(status_code=404, detail="paper index not found")
    builds = store.list_paper_qa_index_builds(arxiv_id)
    selected = inspection.select_build(index_record, builds, build_id)
    if build_id and selected is None:
        raise HTTPException(status_code=404, detail="index build not found")
    # 没有版本记录的旧索引只在主表上保存 chunk_file。
    chunk_file = (selected or index_record).get("chunk_file")
    return index_record, builds, selected, chunk_file


def _load_artifact(chunk_file: Any) -> Dict[str, Any]:
    try:
        return inspection.load_chunk_artifact(chunk_file)
    except inspection.ChunkArtifactUnavailable:
        raise HTTPException(status_code=404, detail="chunk artifact not found")


@router.get("")
def list_paper_indexes(
    q: str = Query("", max_length=200),
    status: str = Query("", max_length=32),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    store=Depends(get_paper_qa_index_store),
):
    """按最近更新倒序列出索引主表；q 匹配 arxiv_id 或标题，status 精确过滤。"""
    items, total = store.list_paper_qa_indexes(query=q, status=status, offset=offset, limit=limit)
    return {"items": items, "total": total}


@router.get("/{arxiv_id}")
def get_paper_index(
    arxiv_id: str,
    build_id: str = Query("", max_length=128),
    store=Depends(get_paper_qa_index_store),
):
    """索引状态、版本列表，以及所选版本的切分概览、统计和全部 chunk。"""
    index_record, builds, selected, chunk_file = _select(store, arxiv_id, build_id)
    return {
        "arxiv_id": arxiv_id,
        "status": index_record.get("status"),
        "active_build_id": index_record.get("active_build_id"),
        "selected_build_id": (selected or {}).get("build_id") or "",
        "builds": [inspection.project_build(build) for build in builds],
        "artifact": inspection.describe_chunk_artifact(chunk_file),
    }


@router.get("/{arxiv_id}/parsed")
def get_parsed_document(
    arxiv_id: str,
    build_id: str = Query("", max_length=128),
    store=Depends(get_paper_qa_index_store),
):
    """所选版本的解析结果；和 chunk 分开取，避免详情接口每次都带上整篇文本。"""
    *_, chunk_file = _select(store, arxiv_id, build_id)
    return inspection.project_parsed_document(_load_artifact(chunk_file))


@router.get("/{arxiv_id}/assets/{chunk_id}")
def get_figure_asset(
    arxiv_id: str,
    chunk_id: int,
    build_id: str = Query("", max_length=128),
    store=Depends(get_paper_qa_index_store),
):
    """按 chunk_id 返回图片 chunk 的原图；非图片 chunk 或原图已清理都返回 404。"""
    *_, chunk_file = _select(store, arxiv_id, build_id)
    try:
        return FileResponse(inspection.resolve_figure_asset(_load_artifact(chunk_file), chunk_id))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="figure asset not found")
