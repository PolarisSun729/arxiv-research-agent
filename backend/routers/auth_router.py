"""注册、登录与账号管理路由（挂载在 /api/auth 下，仅在 JWT 认证模式下注册）。

接口分三类：
- 匿名可访问：/register（需开启注册开关）、/login；
- 登录用户自身：/me、/logout、/password；
- 管理员专用：/users 系列，用于创建、查询、启停账号以及调整每日配额。

只有登录专用响应可以发送新签发的 access_token；其余响应都经过全局 JSON 脱敏器。
各接口把被操作的用户 ID 写入 request.scope["security_target_user_id"]，供审计中间件记录。
"""

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from auth.context import current_auth
from auth.errors import AuthError
from auth.passwords import hash_password, verify_password
from auth.schemas import AdminUserCreate, AdminUserUpdate, PasswordChange, QuotaUpdate, UserCreate, UserLogin
from middleware.common import policy_error


router = APIRouter(prefix="/auth", tags=["authentication"])


def _actor():
    """取出 JWT 中间件写入的当前认证上下文；未登录时抛出 missing_token（401）。"""
    context = current_auth.get()
    if context is None:
        raise AuthError("missing_token")
    return context


def _user_view(store, user) -> dict:
    """用户的公开视图（不含密码哈希等敏感字段），并附带各类配额的当日用量。"""
    return {**user.public_view(), "quotas": store.quotas(user.user_id)}


# ---------------------------------------------------------------------------
# 匿名接口
# ---------------------------------------------------------------------------


@router.post("/register", status_code=201)
def register(payload: UserCreate, request: Request):
    """自助注册；新账号固定为 guest 角色，需管理员再提升权限。"""
    authenticator = request.app.state.authenticator
    if not authenticator.settings.registration_enabled:
        raise AuthError("registration_disabled")
    user = authenticator.store.create_user(username=payload.username, email=str(payload.email),
                                           hashed_password=hash_password(payload.password.get_secret_value()), role="guest")
    request.scope["security_target_user_id"] = user.user_id
    return user.public_view()


@router.post("/login", response_class=JSONResponse)
def login(payload: UserLogin, request: Request):
    """用户名密码登录，成功后返回新签发的 access_token。"""
    # IP 维度的限流已在请求体解析前完成；这里再按归一化账号共享一份预算，限制跨 IP 的密码猜测。
    try:
        decision = request.app.state.rate_limit_controller.login_attempt(payload.username)
    except Exception:
        # 限流存储不可用时拒绝登录（fail-closed），而不是放行无限次尝试。
        raise AuthError("security_storage_unavailable", retry_after=5) from None
    if decision.code:
        return policy_error(request.scope, decision.code, 429, retry_after=decision.retry_after, headers=decision.headers)
    result = request.app.state.authenticator.login(payload.username, payload.password.get_secret_value())
    # 登录成功后把用户对象交给审计中间件，用于记录"谁登录了"。
    user = request.app.state.authenticator.store.get_by_username(payload.username)
    request.scope["security_login_user"] = user
    # 默认 JSON 脱敏器会隐藏 access_token；仅此白名单出口返回令牌，审计从不记录响应正文。
    # no-store / no-cache 防止令牌被浏览器或中间代理缓存。
    return JSONResponse(result, headers={"Cache-Control": "no-store", "Pragma": "no-cache"})


# ---------------------------------------------------------------------------
# 当前登录用户
# ---------------------------------------------------------------------------


@router.get("/me")
def me():
    """返回当前登录用户的资料与配额用量。"""
    context = _actor()
    return _user_view(context.store, context.session.user)


@router.post("/logout")
def logout():
    """吊销当前令牌对应的会话（按 jti），令牌即使未过期也将立即失效。"""
    context = _actor()
    context.store.revoke_session(user_id=context.user_id, jti=context.session.jti)
    return {"status": "logged_out"}


@router.post("/password")
def change_password(payload: PasswordChange):
    """修改密码；需校验原密码。修改后该用户所有已签发的令牌（包括当前这个）都会失效，前端需重新登录。"""
    context = _actor()
    if not verify_password(payload.current_password.get_secret_value(), context.session.user.hashed_password):
        raise AuthError("invalid_credentials")
    context.store.change_password(context.session.user, hashed_password=hash_password(payload.new_password.get_secret_value()))
    return {"status": "password_changed", "reauthenticate": True}


# ---------------------------------------------------------------------------
# 管理员接口
# ---------------------------------------------------------------------------


@router.get("/users")
def list_users(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100)):
    """分页列出全部用户。"""
    context = _actor()
    context.require_roles({"admin"})
    return context.store.list_users(offset=offset, limit=limit)


@router.post("/users", status_code=201)
def create_user(payload: AdminUserCreate, request: Request):
    """管理员直接创建账号，可指定任意角色。"""
    context = _actor()
    context.require_roles({"admin"})
    user = context.store.create_user(username=payload.username, email=str(payload.email), role=payload.role,
                                    hashed_password=hash_password(payload.password.get_secret_value()))
    request.scope["security_target_user_id"] = user.user_id
    return _user_view(context.store, user)


@router.get("/users/{target_user_id}")
def get_user(target_user_id: str, request: Request):
    """查询单个用户的资料与配额用量。"""
    context = _actor()
    context.require_roles({"admin"})
    user = context.store.get_user(target_user_id)
    if user is None:
        raise AuthError("user_not_found")
    request.scope["security_target_user_id"] = user.user_id
    return _user_view(context.store, user)


@router.patch("/users/{target_user_id}")
def update_user(target_user_id: str, payload: AdminUserUpdate, request: Request):
    """修改用户角色或启用状态；两个字段都未提供时视为无效请求。"""
    context = _actor()
    context.require_roles({"admin"})
    if payload.role is None and payload.is_active is None:
        raise AuthError("request_validation_error")
    user = context.store.update_user(target_user_id, role=payload.role, is_active=payload.is_active)
    request.scope["security_target_user_id"] = user.user_id
    return _user_view(context.store, user)


@router.put("/users/{target_user_id}/quotas")
def update_quota(target_user_id: str, payload: QuotaUpdate, request: Request):
    """设置某个用户某类配额（papers / qa_queries / agent_runs）的每日上限，-1 表示不限，不影响当日已用量。"""
    context = _actor()
    context.require_roles({"admin"})
    context.store.set_quota(target_user_id, quota_type=payload.quota_type, daily_limit=payload.daily_limit)
    request.scope["security_target_user_id"] = target_user_id
    return {"quotas": context.store.quotas(target_user_id)}
