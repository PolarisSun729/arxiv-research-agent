from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, HTTPException

from dependencies import (
    get_database_service,
    get_embedding_service,
    get_current_embedding_config,
    get_recommendation_service,
    get_vector_store_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["paper"])


@router.post("/paper")
async def add_paper(
    arxiv_id: str = Body(...),
    title: str = Body(...),
    authors: str = Body(...),
    abstract: str = Body(...),
    categories: str = Body(...),
    published_date: str = Body(...),
    url: str = Body(...),
    collection_name: str = Body("arxiv_abstracts"),
    db_service=Depends(get_database_service),
    embedding_service=Depends(get_embedding_service),
):
    try:
        logger.info("Adding paper with embedding: %s", arxiv_id)

        embedding_config = get_current_embedding_config()
        logger.info(
            "Creating embedding for abstract using %s / %s",
            embedding_config.provider,
            embedding_config.model_name,
        )
        text_to_embed = embedding_service.build_paper_embedding_text(title, abstract)
        embedding = embedding_service.create_single_embedding(
            text_to_embed,
            provider=embedding_config.provider,
            model=embedding_config.model_name,
            api_key=embedding_config.api_key,
            base_url=embedding_config.base_url,
            dimension=embedding_config.dimension,
        )

        logger.info("Embedding created, dimension: %s", len(embedding))

        metadata = {
            "content": abstract,
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "categories": categories,
            "published_date": published_date,
            "url": url,
            "embedding_model": embedding_config.model_name,
        }

        logger.info("Inserting embedding to collection: %s", collection_name)
        vector_store_service = get_vector_store_service()
        embedding_id = vector_store_service.insert_single_embedding(collection_name, embedding, metadata)

        logger.info("Embedding inserted with ID: %s", embedding_id)
        logger.info("Adding paper to database: %s, embedding_id: %s", metadata, embedding_id)

        success = db_service.add_paper(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "categories": categories,
                "published_date": published_date,
                "url": url,
                "embedding_id": str(embedding_id),
                "embedding_model": embedding_config.model_name,
            }
        )

        if success:
            logger.info("Paper %s added successfully with embedding", arxiv_id)
            return {
                "status": "success",
                "message": "Paper added with embedding",
                "arxiv_id": arxiv_id,
                "embedding_id": embedding_id,
                "embedding_model": embedding_config.model_name,
                "vector_dimension": len(embedding),
                "collection_name": collection_name,
            }
        raise HTTPException(status_code=500, detail="Failed to add paper")
    except Exception as exc:
        logger.error("Error adding paper: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/paper/{arxiv_id}")
async def get_paper(
    arxiv_id: str,
    db_service=Depends(get_database_service),
    recommendation_service=Depends(get_recommendation_service),
):
    try:
        paper = db_service.get_paper(arxiv_id)
        if paper:
            return paper
        source_paper = recommendation_service._fetch_paper_from_arxiv_with_rate_limit(arxiv_id)
        if source_paper:
            return recommendation_service._materialize_paper_from_source(source_paper, arxiv_id)
        raise HTTPException(status_code=404, detail="Paper not found")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error getting paper: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/paper/{arxiv_id}")
async def delete_paper(arxiv_id: str, db_service=Depends(get_database_service)):
    try:
        success = db_service.delete_paper(arxiv_id)
        if success:
            return {"status": "success", "message": "Paper deleted"}
        raise HTTPException(status_code=404, detail="Paper not found")
    except Exception as exc:
        logger.error("Error deleting paper: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/papers")
async def get_all_papers(db_service=Depends(get_database_service)):
    try:
        papers = db_service.get_all_papers()
        return {"papers": papers}
    except Exception as exc:
        logger.error("Error getting all papers: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/papers/category/{category}")
async def search_papers_by_category(category: str, db_service=Depends(get_database_service)):
    try:
        papers = db_service.search_papers_by_category(category)
        return {"papers": papers}
    except Exception as exc:
        logger.error("Error searching papers by category: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

