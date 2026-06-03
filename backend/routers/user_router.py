from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from dependencies import get_database_service, get_memory_service, get_recommendation_service
from utils.config import get_default_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/user", tags=["user"])


class PaperActionRequest(BaseModel):
    user_id: str = Field(default_factory=get_default_user_id)
    arxiv_id: str
    action_type: str
    paper: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class ResearchProfileRequest(BaseModel):
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
    try:
        preferences = db_service.get_user_preferences(user_id=user_id)
        return {"status": "success", "message": "User preferences retrieved", "preferences": preferences}
    except Exception as exc:
        logger.error("Error getting user preferences: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/preferences/{user_id}")
async def get_user_preferences(user_id: str, db_service=Depends(get_database_service)):
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
    try:
        return recommendation_service.record_user_paper_preference(
            user_id=user_id,
            arxiv_id=arxiv_id,
            liked=True,
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
    try:
        actions = db_service.get_user_paper_actions(user_id=user_id, action_type=action_type)
        return {
            "status": "success",
            "user_id": user_id,
            "action_type": action_type,
            "actions": actions,
            "action_map": db_service.get_user_paper_action_map(user_id=user_id),
        }
    except Exception as exc:
        logger.error("Error getting paper actions: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/research-profile/{user_id}")
async def get_user_research_profile(user_id: str, memory_service=Depends(get_memory_service)):
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
    try:
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
    try:
        return recommendation_service.recommend_papers(user_id=user_id, top_n=top_n, max_age_months=max_age_months)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error generating recommendations: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
