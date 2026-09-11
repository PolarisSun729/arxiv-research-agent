"""注册、登录、账号管理；只有登录专用响应可以发送新签发的 access_token。"""

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from auth.context import current_auth
from auth.errors import AuthError
from auth.passwords import hash_password, verify_password
from auth.schemas import AdminUserCreate, AdminUserUpdate, PasswordChange, QuotaUpdate, UserCreate, UserLogin


router = APIRouter(prefix="/auth", tags=["authentication"])


def _actor(request: Request):
    context = current_auth.get()
    if context is None:
        raise AuthError("missing_token")
    return context


def _user_view(store, user) -> dict:
    return {**user.public_view(), "quotas": store.quotas(user.user_id)}


@router.post("/register", status_code=201)
def register(payload: UserCreate, request: Request):
    authenticator = request.app.state.authenticator
    if not authenticator.settings.registration_enabled:
        raise AuthError("registration_disabled")
    user = authenticator.store.create_user(username=payload.username, email=str(payload.email),
                                           hashed_password=hash_password(payload.password.get_secret_value()), role="guest")
    request.scope["security_target_user_id"] = user.user_id
    return user.public_view()


@router.post("/login", response_class=JSONResponse)
def login(payload: UserLogin, request: Request):
    # IP 限流已在解析前完成；再按归一化账号共享预算，限制跨 IP 的密码猜测。
    try:
        decision = request.app.state.rate_limit_controller.login_attempt(payload.username)
    except Exception:
        raise AuthError("security_storage_unavailable", retry_after=5) from None
    if decision.code:
        from middleware.common import policy_error
        return policy_error(request.scope, decision.code, 429, retry_after=decision.retry_after, headers=decision.headers)
    result = request.app.state.authenticator.login(payload.username, payload.password.get_secret_value())
    user = request.app.state.authenticator.store.get_by_username(payload.username)
    request.scope["security_login_user"] = user
    # 默认 JSON 脱敏器会隐藏 access_token；仅此白名单出口返回令牌，审计从不记录响应正文。
    return JSONResponse(result, headers={"Cache-Control": "no-store", "Pragma": "no-cache"})


@router.get("/me")
def me(request: Request):
    context = _actor(request)
    return _user_view(context.store, context.session.user)


@router.post("/logout")
def logout(request: Request):
    context = _actor(request)
    context.store.revoke_session(user_id=context.user_id, jti=context.session.jti)
    return {"status": "logged_out"}


@router.post("/password")
def change_password(payload: PasswordChange, request: Request):
    context = _actor(request)
    if not verify_password(payload.current_password.get_secret_value(), context.session.user.hashed_password):
        raise AuthError("invalid_credentials")
    context.store.change_password(context.session.user, hashed_password=hash_password(payload.new_password.get_secret_value()))
    return {"status": "password_changed", "reauthenticate": True}


@router.get("/users")
def list_users(request: Request, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100)):
    context = _actor(request)
    context.require_roles({"admin"})
    return context.store.list_users(offset=offset, limit=limit)


@router.post("/users", status_code=201)
def create_user(payload: AdminUserCreate, request: Request):
    context = _actor(request)
    context.require_roles({"admin"})
    user = context.store.create_user(username=payload.username, email=str(payload.email), role=payload.role,
                                    hashed_password=hash_password(payload.password.get_secret_value()))
    request.scope["security_target_user_id"] = user.user_id
    return _user_view(context.store, user)


@router.get("/users/{target_user_id}")
def get_user(target_user_id: str, request: Request):
    context = _actor(request)
    context.require_roles({"admin"})
    user = context.store.get_user(target_user_id)
    if user is None:
        raise AuthError("user_not_found")
    request.scope["security_target_user_id"] = user.user_id
    return _user_view(context.store, user)


@router.patch("/users/{target_user_id}")
def update_user(target_user_id: str, payload: AdminUserUpdate, request: Request):
    context = _actor(request)
    context.require_roles({"admin"})
    if payload.role is None and payload.is_active is None:
        raise AuthError("request_validation_error")
    user = context.store.update_user(target_user_id, role=payload.role, is_active=payload.is_active)
    request.scope["security_target_user_id"] = user.user_id
    return _user_view(context.store, user)


@router.put("/users/{target_user_id}/quotas")
def update_quota(target_user_id: str, payload: QuotaUpdate, request: Request):
    context = _actor(request)
    context.require_roles({"admin"})
    context.store.set_quota(target_user_id, quota_type=payload.quota_type, daily_limit=payload.daily_limit)
    request.scope["security_target_user_id"] = target_user_id
    return {"quotas": context.store.quotas(target_user_id)}
