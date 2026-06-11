from __future__ import annotations

import argparse
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.errors import AppError, ErrorCode, http_exception_to_app_error, make_error_payload
from dependencies import SERVICE_LOAD_MODE, normalize_service_load_mode, warm_up_services
from routers.agent_router import router as agent_router
from routers.arxiv_router import router as arxiv_router
from routers.paper_router import router as paper_router
from routers.qa_router import router as qa_router
from routers.user_router import router as user_router
from utils.config import get_debug_routes_runtime_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def create_app(load_mode: str | None = None, *, enable_debug_routes: bool | None = None) -> FastAPI:
    resolved_load_mode = normalize_service_load_mode(load_mode)
    debug_route_config = get_debug_routes_runtime_config()
    resolved_enable_debug_routes = (
        bool(debug_route_config.get("enable_debug_routes"))
        if enable_debug_routes is None
        else bool(enable_debug_routes)
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 使用 lifespan 替代 on_event，避免 FastAPI 的弃用警告。
        # 同时保持预加载行为不变。
        logging.getLogger(__name__).info("Backend service load mode: %s", resolved_load_mode)
        try:
            # 上下文生命周期清理只处理 checkpoint/debug/trace 边界，不触碰原始聊天历史和 session summary。
            from services.context_lifecycle import ContextLifecycleService
            from utils.config import get_enhanced_retrieval_runtime_config

            cleanup_result = ContextLifecycleService().run_startup_cleanup(
                trace_export_dir=get_enhanced_retrieval_runtime_config().get("trace_export_dir"),
            )
            logging.getLogger(__name__).info("Context lifecycle startup cleanup: %s", cleanup_result)
        except Exception as exc:
            logging.getLogger(__name__).warning("Context lifecycle startup cleanup skipped: %s", exc)
        if resolved_load_mode == "preload":
            warm_up_services(resolved_load_mode)

            from agents.arxiv_search_agent.schemas import get_default_agent_arxiv_categories, get_valid_arxiv_categories
            from tools.tool_registry import get_tool_registry

            get_tool_registry()
            get_valid_arxiv_categories()
            get_default_agent_arxiv_categories()

        yield

    app = FastAPI(lifespan=lifespan)

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        # 全局出口只返回稳定错误契约；完整异常上下文留在日志里，避免前端收到底层堆栈。
        logging.getLogger(__name__).warning(
            "api_error code=%s recoverable=%s path=%s context=%s detail=%s",
            exc.code,
            exc.recoverable,
            request.url.path,
            exc.context,
            exc.detail,
        )
        return exc.to_response()

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
        app_error = http_exception_to_app_error(exc)
        logging.getLogger(__name__).warning(
            "http_error mapped code=%s status=%s path=%s detail=%s",
            app_error.code,
            exc.status_code,
            request.url.path,
            app_error.detail,
        )
        return app_error.to_response()

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        # 请求体验上只提示参数错误，细节中保留字段级摘要，便于开发调试但不暴露内部实现。
        payload = make_error_payload(
            code=ErrorCode.REQUEST_VALIDATION_ERROR,
            detail=str(exc.errors()),
            recoverable=True,
        )
        logging.getLogger(__name__).warning(
            "request_validation_error path=%s detail=%s",
            request.url.path,
            payload.get("detail"),
        )
        return JSONResponse(status_code=422, content=payload)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 保持现有的路由注册顺序不变。
    app.include_router(arxiv_router, prefix="/api")
    app.include_router(agent_router, prefix="/api")
    app.include_router(user_router, prefix="/api")
    app.include_router(paper_router, prefix="/api")
    app.include_router(qa_router, prefix="/api")
    if resolved_enable_debug_routes:
        # chunk debug 会暴露本地解析产物，只有显式开启调试路由时才挂载到内部 debug 前缀。
        from routers.chunk_router import router as chunk_debug_router

        app.include_router(chunk_debug_router, prefix="/api")
        logging.getLogger(__name__).info("Debug routes enabled: /api/debug/chunks")
    else:
        logging.getLogger(__name__).info("Debug routes disabled")

    return app


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--load-mode",
        choices=["lazy", "preload"],
        default=SERVICE_LOAD_MODE,
        help="后端启动加载模式：lazy 或 preload",
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
