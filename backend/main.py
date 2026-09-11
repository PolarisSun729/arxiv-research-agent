from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from auth.api_key_middleware import ApiKeyMiddleware, ApiKeySettings, SECURITY_HEADERS, get_allowed_origins, verify_api_key
from auth.jwt_handler import JwtAuthenticator
from auth.jwt_middleware import JwtAuthMiddleware, UserQuotaMiddleware, require_jwt_identity
from auth.settings import JwtSettings, auth_mode
from auth.store import AuthStore
from core.errors import AppError, ErrorCode, http_exception_to_app_error, make_error_payload
from core.responses import RedactedJSONResponse
from dependencies import (
    SERVICE_LOAD_MODE,
    get_agent_runtime_checkpoint_store,
    get_agent_resume_run_manager,
    get_index_job_manager,
    normalize_service_load_mode,
    warm_up_services,
)
from routers.agent_router import router as agent_router
from routers.arxiv_router import router as arxiv_router
from routers.paper_router import router as paper_router
from routers.qa_router import router as qa_router
from routers.user_router import router as user_router
from utils.config import get_debug_routes_runtime_config
from utils.logging_utils import configure_backend_logging, info_event
from middleware.audit_log import AuditLogMiddleware, AuditSink
from middleware.ip_filter import IPFilterMiddleware, IPFilterSettings
from middleware.rate_limit import RateLimitController, RateLimitMiddleware, RateLimitSettings

LOGGING_RUNTIME_CONFIG = configure_backend_logging()
logger = logging.getLogger(__name__)


def _uvicorn_log_level(level_name: str | None) -> str:
    normalized = str(level_name or "INFO").lower()
    return normalized if normalized in {"critical", "error", "warning", "info", "debug", "trace"} else "info"


def create_app(load_mode: str | None = None, *, enable_debug_routes: bool | None = None) -> FastAPI:
    # 模式在启动时固定，JWT 不会因为配置了旧 API Key 就跳过账号权限。
    mode = auth_mode()
    security_settings = ApiKeySettings.from_environment() if mode == "api_key" else ApiKeySettings((), get_allowed_origins())
    authenticator = None
    if mode == "jwt":
        jwt_settings = JwtSettings.from_environment()
        authenticator = JwtAuthenticator(jwt_settings, AuthStore(jwt_settings.database_path))
    ip_settings = IPFilterSettings.from_environment()
    rate_controller = RateLimitController(RateLimitSettings.from_environment())
    audit_sink = AuditSink()
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
        info_event(logger, "backend.startup", load_mode=resolved_load_mode)
        try:
            # 上下文生命周期清理只处理 checkpoint/debug/trace 边界，不触碰原始聊天历史和 session summary。
            from services.context_lifecycle import ContextLifecycleService
            from utils.config import get_enhanced_retrieval_runtime_config

            context_lifecycle_service = ContextLifecycleService(
                agent_runtime_checkpoint_store=get_agent_runtime_checkpoint_store(),
            )
            cleanup_result = context_lifecycle_service.run_startup_cleanup(
                trace_export_dir=get_enhanced_retrieval_runtime_config().get("trace_export_dir"),
            )
            # 启动主日志只保留清理摘要，避免把整份 retention policy 打成超长单行后在控制台视觉换行。
            info_event(
                logger,
                "backend.startup_cleanup_done",
                **context_lifecycle_service.summarize_startup_cleanup_result(cleanup_result),
            )
        except Exception as exc:
            logger.warning("Context lifecycle startup cleanup skipped: %s", exc)
        if resolved_load_mode == "preload":
            warm_up_services(resolved_load_mode)

            from agents.arxiv_search_agent.schemas import get_default_agent_arxiv_categories, get_valid_arxiv_categories
            from tools.tool_registry import get_tool_registry

            get_tool_registry()
            get_valid_arxiv_categories()
            get_default_agent_arxiv_categories()

        index_job_manager = None
        try:
            # worker 的待办与 lease 都在 SQLite 中；启动时无条件拉起，才能接管上个进程遗留的 pending/retrying job。
            index_job_manager = get_index_job_manager()
            index_job_manager.start()
            info_event(logger, "qa_index_worker.started", worker_id=index_job_manager.worker_id)
            # pending resume run 可以安全重领；上个进程已 running 的 run 标记 indeterminate，禁止重放 checkpoint。
            get_agent_resume_run_manager().recover_incomplete_runs()
        except Exception as exc:
            # 后台能力启动失败必须显式记录，Agent 后续提交会返回失败，不能静默回退到同步建索引。
            logger.exception("QA index lease worker failed to start: %s", exc)

        try:
            yield
        finally:
            if index_job_manager is not None:
                index_job_manager.stop()
                info_event(logger, "qa_index_worker.stopped", worker_id=index_job_manager.worker_id)

    app = FastAPI(lifespan=lifespan, default_response_class=RedactedJSONResponse, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.api_key_settings = security_settings
    app.state.auth_mode = mode
    app.state.authenticator = authenticator
    app.state.limiter = rate_controller.limiter
    app.state.rate_limit_controller = rate_controller

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        request.scope["security_error_code"] = exc.code
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
        request.scope["security_error_code"] = app_error.code
        logging.getLogger(__name__).warning(
            "http_error mapped code=%s status=%s path=%s detail=%s",
            app_error.code,
            exc.status_code,
            request.url.path,
            exc.detail,
        )
        # 兼容统一错误契约，同时保留认证 challenge 等 HTTP 语义。
        return JSONResponse(status_code=exc.status_code, content=app_error.to_payload(), headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        request.scope["security_error_code"] = ErrorCode.REQUEST_VALIDATION_ERROR
        # Pydantic 的 input/ctx/msg 可能回显整个请求体；仅返回字段位置和校验类型。
        payload = make_error_payload(
            code=ErrorCode.REQUEST_VALIDATION_ERROR,
            detail=[{"loc": error.get("loc"), "type": error.get("type")} for error in exc.errors()],
            recoverable=True,
        )
        logging.getLogger(__name__).warning(
            "request_validation_error path=%s detail=%s",
            request.url.path,
            payload.get("detail"),
        )
        return JSONResponse(status_code=422, content=payload)

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        # 未被业务层识别的异常只进脱敏日志，响应不能携带 provider 原文或服务器堆栈。
        logger.exception("unhandled_api_error path=%s", request.url.path, exc_info=exc)
        headers = dict(SECURITY_HEADERS)
        if request.scope.get("security_request_id"):
            headers["X-Request-ID"] = request.scope["security_request_id"]
        # ServerErrorMiddleware 在 CORS 外层调用此处理器，因此兜底响应要复用同一份 Origin 允许列表。
        origin = request.headers.get("origin")
        if origin in security_settings.allowed_origins:
            headers.update({"Access-Control-Allow-Origin": origin, "Vary": "Origin"})
        return JSONResponse(status_code=500, content=make_error_payload(code=ErrorCode.UNKNOWN_ERROR), headers=headers)

    # 注册顺序与执行顺序相反：IP 防护 → 认证/角色/身份 → 用户速率 → 用户日配额 → 业务。
    if authenticator:
        app.add_middleware(UserQuotaMiddleware)
    app.add_middleware(RateLimitMiddleware, controller=rate_controller)
    if authenticator:
        app.add_middleware(JwtAuthMiddleware, authenticator=authenticator)
    else:
        app.add_middleware(ApiKeyMiddleware, settings=security_settings)
    app.add_middleware(RateLimitMiddleware, controller=rate_controller, before_auth=True)
    app.add_middleware(IPFilterMiddleware, settings=ip_settings)
    # CORS 位于认证外层，预检无须密钥，401/403 也能被允许的前端正常读取。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(security_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key", "Accept"],
        expose_headers=["Retry-After", "X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset", "X-RateLimit-Scope", "X-DailyQuota-Limit", "X-DailyQuota-Remaining", "X-DailyQuota-Reset"],
    )
    # 审计位于 CORS 外层，覆盖预检、认证拒绝、限流拒绝和业务异常，并观察完整 SSE 生命周期。
    app.add_middleware(AuditLogMiddleware, sink=audit_sink, routes=app.routes, ip_settings=ip_settings)

    # JWT 入口有默认拒绝的权限清单；API Key 模式保留旧的可信单团队语义。
    api_dependencies = [Depends(require_jwt_identity if authenticator else verify_api_key)]
    if authenticator:
        from routers.auth_router import router as auth_router
        app.include_router(auth_router, prefix="/api")
    app.include_router(arxiv_router, prefix="/api", dependencies=api_dependencies)
    app.include_router(agent_router, prefix="/api", dependencies=api_dependencies)
    app.include_router(user_router, prefix="/api", dependencies=api_dependencies)
    app.include_router(paper_router, prefix="/api", dependencies=api_dependencies)
    app.include_router(qa_router, prefix="/api", dependencies=api_dependencies)
    if resolved_enable_debug_routes:
        # chunk debug 会暴露本地解析产物，只有显式开启调试路由时才挂载到内部 debug 前缀。
        from routers.chunk_router import router as chunk_debug_router

        app.include_router(chunk_debug_router, prefix="/api", dependencies=api_dependencies)
        info_event(logger, "backend.debug_routes", enabled=True, route="/api/debug/chunks")
    else:
        info_event(logger, "backend.debug_routes", enabled=False)

    @app.get("/api/auth/check", dependencies=api_dependencies)
    async def check_access_key():
        """登录页只验证访问资格，不初始化模型或读取用户数据。"""
        return {"status": "authenticated"}

    @app.get("/api/auth/config", include_in_schema=False)
    async def auth_config():
        """登录页只读取公开的模式与注册开关，不返回任何密钥或内部配置。"""
        return {"mode": mode, "registration_enabled": bool(authenticator and authenticator.settings.registration_enabled)}

    @app.get("/health", include_in_schema=False)
    async def health_check():
        """公开存活探针，只返回固定服务标识，不包含配置、路径或外部依赖详情。"""
        return {"status": "healthy", "service": "arxiv-research-backend", "version": "1.0.0"}

    info_event(logger, "backend.security", auth_mode=mode, key_count=len(security_settings.key_hashes), allowed_origins=security_settings.allowed_origins)
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
        # 公网入口交给 HTTPS 反向代理；容器部署需要外部监听时必须显式配置。
        host=os.getenv("BACKEND_HOST", "127.0.0.1"),
        # 原始 TCP 对端必须保留，代理头只由 IPFilterMiddleware 按 TRUSTED_PROXY_IPS 解释。
        proxy_headers=False,
        port=8001,
        reload=False,
        # uvicorn 自身跟随后端日志级别；access log 单独可控，避免 HTTP 访问行淹没业务事件。
        log_level=_uvicorn_log_level(str(LOGGING_RUNTIME_CONFIG.get("level") or "INFO")),
        access_log=bool(LOGGING_RUNTIME_CONFIG.get("access_log", False)),
    )
