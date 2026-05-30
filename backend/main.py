from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.arxiv_router import router as arxiv_router
from routers.agent_router import router as agent_router
from routers.chunk_router import router as chunk_router
from routers.paper_router import router as paper_router
from routers.qa_router import router as qa_router
from routers.user_router import router as user_router

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(arxiv_router)
app.include_router(agent_router)
app.include_router(user_router)
app.include_router(paper_router)
app.include_router(qa_router)
app.include_router(chunk_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="debug",
    )
