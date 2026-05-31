from __future__ import annotations

import argparse
import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dependencies import SERVICE_LOAD_MODE, normalize_service_load_mode, warm_up_services
from routers.agent_router import router as agent_router
from routers.arxiv_router import router as arxiv_router
from routers.chunk_router import router as chunk_router
from routers.paper_router import router as paper_router
from routers.qa_router import router as qa_router
from routers.user_router import router as user_router

logging.basicConfig(level=logging.DEBUG)


def create_app(load_mode: str | None = None) -> FastAPI:
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 缁熶竴鎶婂悗绔叕寮€鎺ュ彛鎸傚湪 /api 涓嬶紝鍓嶇鍙渶瑕佷繚鐣欎竴涓ǔ瀹氱殑鍩虹鍓嶇紑銆?
    app.include_router(arxiv_router, prefix="/api")
    app.include_router(agent_router, prefix="/api")
    app.include_router(user_router, prefix="/api")
    app.include_router(paper_router, prefix="/api")
    app.include_router(qa_router, prefix="/api")
    app.include_router(chunk_router, prefix="/api")

    resolved_load_mode = normalize_service_load_mode(load_mode)

    @app.on_event("startup")
    async def _warm_up_services() -> None:
        logging.getLogger(__name__).info("Backend service load mode: %s", resolved_load_mode)
        if resolved_load_mode != "preload":
            return

        warm_up_services(resolved_load_mode)

        from agents.arxiv_search_agent.schemas import get_default_agent_arxiv_categories, get_valid_arxiv_categories
        from tools.tool_registry import get_tool_registry

        get_tool_registry()
        get_valid_arxiv_categories()
        get_default_agent_arxiv_categories()

    return app


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--load-mode",
        choices=["lazy", "preload"],
        default=SERVICE_LOAD_MODE,
        help="Backend startup load mode: lazy or preload",
    )
    args, _ = parser.parse_known_args(sys.argv[1:])

    import uvicorn

    uvicorn.run(
        create_app(load_mode=args.load_mode),
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="debug",
    )
