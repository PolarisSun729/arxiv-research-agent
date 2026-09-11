from __future__ import annotations

"""用户画像与偏好相关路由。

该模块负责用户偏好、喜欢/不喜欢论文、行为埋点、研究画像、兴趣向量
以及个性化推荐等能力的接口编排，本身主要做参数接收、鉴权范围控制和服务转发。
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from dependencies import (
    get_interest_vector_store,
    get_memory_service,
    get_recommendation_service,
    get_user_preference_store,
)
from services.user_behavior_policy import validate_weak_paper_action_type
from utils.config import get_default_user_id
from auth.context import bind_user_id, current_auth
from auth.errors import AuthError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/user", tags=["user"])


class PaperActionRequest(BaseModel):
    """记录用户对论文执行的弱行为。

    点赞/点踩是强偏好，必须走 like-paper/dislike-paper；这里提前拒绝等价 action_type，
    避免 paper-action 和偏好表同时成为同一状态的权威来源。
    """
    user_id: str = Field(default_factory=get_default_user_id)
    arxiv_id: str
    action_type: str
    paper: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    @field_validator("action_type", mode="before")
    @classmethod
    def _validate_action_type(cls, value: Any) -> str:
        try:
            return validate_weak_paper_action_type(value)
        except ValueError as exc:
            if str(exc) == "explicit_preference_action_not_allowed":
                raise ValueError("paper-action 不接受 like/dislike；请使用 /user/like-paper 或 /user/dislike-paper") from exc
            raise ValueError("unsupported paper action type") from exc


class ResearchProfileRequest(BaseModel):
    """研究画像更新请求。

    该模型描述的是用户长期偏好信号，通常由人工编辑或系统总结后写入，
    用于后续推荐、问答风格调优和研究方向理解。
    """
    user_id: str = Field(default_factory=get_default_user_id)
    positive_topics: Optional[list[str]] = None
    negative_topics: Optional[list[str]] = None
    recent_topics: Optional[list[str]] = None
    pinned_topics: Optional[list[str]] = None
    hidden_topics: Optional[list[str]] = None
    preferred_categories: Optional[list[str]] = None
    preferred_answer_style: Optional[str] = None
    common_question_types: Optional[list[str]] = None
    representative_papers: Optional[list[str]] = None


class InterestVectorRequest(BaseModel):
    """前端使用对象形式传参，历史调用的单字符串形式在路由边界继续兼容。"""

    user_id: Optional[str] = None


class RebuildResearchProfileRequest(BaseModel):
    """研究画像重建请求；前端默认使用快速增量模式，全量/修复模式留给显式维护入口。"""
    user_id: str = Field(default_factory=get_default_user_id)
    async_build: bool = True
    build_mode: str = "incremental"
    max_papers: Optional[int] = None


class ActivateProfileSnapshotRequest(BaseModel):
    """切换 active snapshot 的请求。"""
    user_id: str = Field(default_factory=get_default_user_id)
    snapshot_id: str


@router.get("/preferences/{user_id}")
async def get_user_preferences(user_id: str, user_preference_store=Depends(get_user_preference_store)):
    """按路径参数获取用户偏好。"""
    try:
        preferences = user_preference_store.get_user_preferences(user_id=user_id)
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
    """记录用户对论文执行的弱行为事件。"""
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
    user_preference_store=Depends(get_user_preference_store),
):
    """删除某条已记录的弱论文行为。"""
    try:
        try:
            normalized_action = validate_weak_paper_action_type(action_type)
        except ValueError as exc:
            if str(exc) == "explicit_preference_action_not_allowed":
                # 强偏好撤销必须走专用接口，避免调用方误以为 paper-action 能删除权威点赞状态。
                raise HTTPException(status_code=400, detail="paper-action 不接受 like/dislike；请使用 DELETE /user/like-paper 或 DELETE /user/dislike-paper") from exc
            raise HTTPException(status_code=400, detail="unsupported paper action type") from exc
        success = user_preference_store.remove_user_paper_action(
            user_id=user_id,
            arxiv_id=arxiv_id,
            action_type=normalized_action,
        )
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
    user_preference_store=Depends(get_user_preference_store),
):
    """获取用户的论文行为明细及聚合映射。"""
    try:
        actions = user_preference_store.get_user_paper_actions(user_id=user_id, action_type=action_type)
        return {
            "status": "success",
            "user_id": user_id,
            "action_type": action_type,
            "actions": actions,
            # action_map 常用于前端快速判断某篇论文当前是否已被执行过某种动作。
            "action_map": user_preference_store.get_user_paper_action_map(user_id=user_id),
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


@router.get("/research-profile/{user_id}/detail")
async def get_user_research_profile_detail(user_id: str, memory_service=Depends(get_memory_service)):
    """读取画像详情：manual/generated/effective、质量报告、构建任务和快照摘要。"""
    try:
        return {"status": "success", "detail": memory_service.load_user_profile_detail(user_id=user_id)}
    except Exception as exc:
        logger.error("Error getting research profile detail: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/research-profile/{user_id}/topic-evidence")
async def get_user_research_profile_topic_evidence(
    user_id: str,
    topic: str = Query(...),
    memory_service=Depends(get_memory_service),
):
    """读取单个 topic 的来源证据，供前端解释画像项。"""
    try:
        return {"status": "success", **memory_service.get_profile_topic_evidence(user_id=user_id, topic=topic)}
    except Exception as exc:
        logger.error("Error getting profile topic evidence: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/research-profile/build-jobs/{job_id}")
async def get_user_research_profile_build_job(job_id: str, memory_service=Depends(get_memory_service)):
    """查询画像构建任务状态，前端可据此轮询。"""
    try:
        job = memory_service.get_profile_build_job(job_id)
        context = current_auth.get()
        if context and job and job.get("user_id") != context.user_id:
            raise AuthError("private_resource_not_found")
        if not job:
            raise HTTPException(status_code=404, detail="profile build job not found")
        return {"status": "success", "job": job}
    except (HTTPException, AuthError):
        raise
    except Exception as exc:
        logger.error("Error getting profile build job: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/research-profile/{user_id}/build-jobs")
async def list_user_research_profile_build_jobs(
    user_id: str,
    limit: int = Query(20, ge=1, le=100),
    memory_service=Depends(get_memory_service),
):
    """列出用户画像构建任务历史。"""
    try:
        return {"status": "success", "items": memory_service.list_profile_build_jobs(user_id=user_id, limit=limit)}
    except Exception as exc:
        logger.error("Error listing profile build jobs: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/research-profile/snapshots/activate")
async def activate_user_research_profile_snapshot(
    payload: ActivateProfileSnapshotRequest,
    memory_service=Depends(get_memory_service),
):
    """切换 active snapshot，用于低质量构建后的人工回滚或确认。"""
    try:
        profile = memory_service.activate_profile_snapshot(user_id=payload.user_id, snapshot_id=payload.snapshot_id)
        return {"status": "success", "profile": profile}
    except ValueError as exc:
        if str(exc) == "profile_snapshot_not_found":
            raise HTTPException(status_code=404, detail=str(exc))
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Error activating profile snapshot: %s", str(exc))
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


@router.post("/research-profile/rebuild")
async def rebuild_user_research_profile(
    payload: RebuildResearchProfileRequest,
    background_tasks: BackgroundTasks,
    memory_service=Depends(get_memory_service),
):
    """根据用户行为证据重建画像；默认返回 build job，避免慢速构建阻塞前端。"""
    try:
        if not payload.async_build:
            # 同步模式仅保留给测试和本地维护；前端默认使用异步 job，避免慢速 LLM 构建阻塞请求。
            profile = memory_service.rebuild_user_research_profile(
                user_id=payload.user_id,
                build_mode=payload.build_mode,
                max_papers=payload.max_papers,
            )
            return {"status": "success", "profile": profile}
        job = memory_service.create_profile_rebuild_job(
            user_id=payload.user_id,
            build_config={"build_mode": payload.build_mode, "max_papers": payload.max_papers},
        )
        background_tasks.add_task(
            memory_service.run_profile_rebuild_job,
            payload.user_id,
            job.get("job_id"),
            payload.build_mode,
            payload.max_papers,
        )
        return {
            "status": "accepted",
            "job": job,
            "profile": memory_service.load_user_profile(payload.user_id),
        }
    except Exception as exc:
        logger.error("Error rebuilding research profile: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/like-paper")
async def remove_like(
    arxiv_id: str = Body(...),
    user_id: str = Body(default_factory=get_default_user_id),
    user_preference_store=Depends(get_user_preference_store),
):
    """撤销用户对论文的点赞记录。"""
    try:
        success = user_preference_store.remove_liked_paper(user_id=user_id, arxiv_id=arxiv_id)
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
    user_preference_store=Depends(get_user_preference_store),
):
    """撤销用户对论文的点踩记录。"""
    try:
        success = user_preference_store.remove_disliked_paper(user_id=user_id, arxiv_id=arxiv_id)
        if success:
            return {"status": "success", "message": "Paper removed from disliked list"}
        raise HTTPException(status_code=500, detail="Failed to remove from disliked list")
    except Exception as exc:
        logger.error("Error removing disliked paper: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/generate-interest-vector")
async def generate_user_interest_vector(
    user_id: str | InterestVectorRequest = Body(default_factory=get_default_user_id),
    recommendation_service=Depends(get_recommendation_service),
):
    """根据用户行为与偏好数据重新生成兴趣向量。"""
    # 两种传参形式都绑定可信身份，不能因兼容标量请求而落回其他用户的数据空间。
    resolved_user_id = bind_user_id(user_id.user_id if isinstance(user_id, InterestVectorRequest) else user_id,
                                    fallback=get_default_user_id())
    try:
        return recommendation_service.generate_user_interest_vector(user_id=resolved_user_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error generating interest vector: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/interest-vector")
async def get_user_interest_vector(
    user_id: str = Query(default_factory=get_default_user_id),
    interest_vector_store=Depends(get_interest_vector_store),
):
    """获取用户当前已保存的兴趣向量。"""
    try:
        result = interest_vector_store.get_user_interest_vector(user_id=user_id)
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
