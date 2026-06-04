from __future__ import annotations

"""用户画像与偏好相关路由。

该模块负责用户偏好、喜欢/不喜欢论文、行为埋点、研究画像、兴趣向量
以及个性化推荐等能力的接口编排，本身主要做参数接收、鉴权范围控制和服务转发。
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from dependencies import get_database_service, get_memory_service, get_recommendation_service
from utils.config import get_default_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/user", tags=["user"])


class PaperActionRequest(BaseModel):
    """记录用户对论文执行的某类动作。

    这里的 action_type 可以承载更广义的行为语义，
    例如浏览、收藏、加入对话、导出等，而不限于点赞/点踩。
    """
    user_id: str = Field(default_factory=get_default_user_id)
    arxiv_id: str
    action_type: str
    paper: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class ResearchProfileRequest(BaseModel):
    """研究画像更新请求。

    该模型描述的是用户长期偏好信号，通常由人工编辑或系统总结后写入，
    用于后续推荐、问答风格调优和研究方向理解。
    """
    user_id: str = Field(default_factory=get_default_user_id)
    positive_topics: Optional[list[str]] = None
    negative_topics: Optional[list[str]] = None
    recent_topics: Optional[list[str]] = None
    preferred_categories: Optional[list[str]] = None
    preferred_answer_style: Optional[str] = None
    common_question_types: Optional[list[str]] = None
    representative_papers: Optional[list[str]] = None


@router.post("/preferences")
async def upsert_user_preferences(
    user_id: str = Body(default_factory=get_default_user_id),
    db_service=Depends(get_database_service),
):
    """读取指定用户的偏好设置。

    虽然路由名叫 upsert，但当前实现更像“按 user_id 读取偏好”，
    便于前端初始化用户配置面板时直接获取现有数据。
    """
    try:
        preferences = db_service.get_user_preferences(user_id=user_id)
        return {"status": "success", "message": "User preferences retrieved", "preferences": preferences}
    except Exception as exc:
        logger.error("Error getting user preferences: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/preferences/{user_id}")
async def get_user_preferences(user_id: str, db_service=Depends(get_database_service)):
    """按路径参数获取用户偏好。"""
    try:
        preferences = db_service.get_user_preferences(user_id=user_id)
        return preferences
    except Exception as exc:
        logger.error("Error getting user preferences: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/like-paper")
async def like_paper(
    arxiv_id: str = Body(...),
    user_id: str = Body(default_factory=get_default_user_id),
    paper: Optional[Dict[str, Any]] = Body(None),
    recommendation_service=Depends(get_recommendation_service),
):
    """记录用户“喜欢论文”的显式正反馈。"""
    try:
        return recommendation_service.record_user_paper_preference(
            user_id=user_id,
            arxiv_id=arxiv_id,
            liked=True,
            # 额外传入 paper 快照，避免服务层必须再次查库或回源才能补齐上下文。
            paper_payload=paper,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error liking paper: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/dislike-paper")
async def dislike_paper(
    arxiv_id: str = Body(...),
    user_id: str = Body(default_factory=get_default_user_id),
    paper: Optional[Dict[str, Any]] = Body(None),
    recommendation_service=Depends(get_recommendation_service),
):
    """记录用户“不喜欢论文”的显式负反馈。"""
    try:
        return recommendation_service.record_user_paper_preference(
            user_id=user_id,
            arxiv_id=arxiv_id,
            liked=False,
            paper_payload=paper,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error disliking paper: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/paper-action")
async def record_paper_action(
    payload: PaperActionRequest,
    recommendation_service=Depends(get_recommendation_service),
):
    """记录用户对论文执行的通用行为事件。"""
    try:
        return recommendation_service.record_user_paper_action(
            user_id=payload.user_id,
            arxiv_id=payload.arxiv_id,
            action_type=payload.action_type,
            paper_payload=payload.paper,
            metadata=payload.metadata,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error recording paper action: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/paper-action")
async def remove_paper_action(
    arxiv_id: str = Body(...),
    action_type: str = Body(...),
    user_id: str = Body(default_factory=get_default_user_id),
    db_service=Depends(get_database_service),
):
    """删除某条已记录的用户论文行为。"""
    try:
        success = db_service.remove_user_paper_action(user_id=user_id, arxiv_id=arxiv_id, action_type=action_type)
        if success:
            return {"status": "success", "message": "Paper action removed"}
        raise HTTPException(status_code=404, detail="Paper action not found")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error removing paper action: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/paper-actions/{user_id}")
async def get_user_paper_actions(
    user_id: str,
    action_type: Optional[str] = None,
    db_service=Depends(get_database_service),
):
    """获取用户的论文行为明细及聚合映射。"""
    try:
        actions = db_service.get_user_paper_actions(user_id=user_id, action_type=action_type)
        return {
            "status": "success",
            "user_id": user_id,
            "action_type": action_type,
            "actions": actions,
            # action_map 常用于前端快速判断某篇论文当前是否已被执行过某种动作。
            "action_map": db_service.get_user_paper_action_map(user_id=user_id),
        }
    except Exception as exc:
        logger.error("Error getting paper actions: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/research-profile/{user_id}")
async def get_user_research_profile(user_id: str, memory_service=Depends(get_memory_service)):
    """读取用户研究画像。"""
    try:
        return memory_service.load_user_profile(user_id=user_id)
    except Exception as exc:
        logger.error("Error getting research profile: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.put("/research-profile")
async def upsert_user_research_profile(
    payload: ResearchProfileRequest,
    memory_service=Depends(get_memory_service),
):
    """以“整体更新/补全”的语义写入研究画像。"""
    try:
        # user_id 单独作为主键传入，其他非空字段才会参与画像更新，避免把未传值误写成 null。
        profile_payload = payload.model_dump(exclude={"user_id"}, exclude_none=True)
        profile = memory_service.patch_user_profile(user_id=payload.user_id, patch=profile_payload, source="manual_upsert")
        return {"status": "success", "profile": profile}
    except Exception as exc:
        logger.error("Error upserting research profile: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.patch("/research-profile")
async def patch_user_research_profile(
    payload: ResearchProfileRequest,
    memory_service=Depends(get_memory_service),
):
    """以“局部补丁”的语义更新研究画像。"""
    try:
        profile_payload = payload.model_dump(exclude={"user_id"}, exclude_none=True)
        profile = memory_service.patch_user_profile(user_id=payload.user_id, patch=profile_payload, source="manual")
        return {"status": "success", "profile": profile}
    except Exception as exc:
        logger.error("Error patching research profile: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/like-paper")
async def remove_like(
    arxiv_id: str = Body(...),
    user_id: str = Body(default_factory=get_default_user_id),
    db_service=Depends(get_database_service),
):
    """撤销用户对论文的点赞记录。"""
    try:
        success = db_service.remove_liked_paper(user_id=user_id, arxiv_id=arxiv_id)
        if success:
            return {"status": "success", "message": "Paper removed from liked list"}
        raise HTTPException(status_code=500, detail="Failed to remove from liked list")
    except Exception as exc:
        logger.error("Error removing liked paper: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/dislike-paper")
async def remove_dislike(
    arxiv_id: str = Body(...),
    user_id: str = Body(default_factory=get_default_user_id),
    db_service=Depends(get_database_service),
):
    """撤销用户对论文的点踩记录。"""
    try:
        success = db_service.remove_disliked_paper(user_id=user_id, arxiv_id=arxiv_id)
        if success:
            return {"status": "success", "message": "Paper removed from disliked list"}
        raise HTTPException(status_code=500, detail="Failed to remove from disliked list")
    except Exception as exc:
        logger.error("Error removing disliked paper: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/generate-interest-vector")
async def generate_user_interest_vector(
    user_id: str = Body(default_factory=get_default_user_id),
    recommendation_service=Depends(get_recommendation_service),
):
    """根据用户行为与偏好数据重新生成兴趣向量。"""
    try:
        return recommendation_service.generate_user_interest_vector(user_id=user_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error generating interest vector: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/interest-vector")
async def get_user_interest_vector(
    user_id: str = Query(default_factory=get_default_user_id),
    db_service=Depends(get_database_service),
):
    """获取用户当前已保存的兴趣向量。"""
    try:
        result = db_service.get_user_interest_vector(user_id=user_id)
        if result:
            return result
        raise HTTPException(status_code=404, detail="User interest vector not found")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error getting interest vector: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/recommend-papers")
async def recommend_papers(
    user_id: str = Body(default_factory=get_default_user_id),
    top_n: int = Body(10),
    max_age_months: int = Body(6),
    recommendation_service=Depends(get_recommendation_service),
):
    """基于用户兴趣向量和近期论文池生成个性化推荐。"""
    try:
        return recommendation_service.recommend_papers(user_id=user_id, top_n=top_n, max_age_months=max_age_months)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error generating recommendations: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
