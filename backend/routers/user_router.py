from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, HTTPException

from dependencies import get_database_service, get_recommendation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/user", tags=["user"])


@router.post("/preferences")
async def upsert_user_preferences(
    user_id: str = Body("local_user"),
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
    user_id: str = Body("local_user"),
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
    user_id: str = Body("local_user"),
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


@router.delete("/like-paper")
async def remove_like(
    arxiv_id: str = Body(...),
    user_id: str = Body("local_user"),
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
    user_id: str = Body("local_user"),
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
    user_id: str = Body("local_user"),
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
    user_id: str = "local_user",
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
    user_id: str = Body("local_user"),
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

