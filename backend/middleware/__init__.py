"""公网入口的限流、IP 过滤与审计边界。

本包提供纯 ASGI 中间件（不使用 BaseHTTPMiddleware，避免缓冲 SSE 流式响应），
在 main.create_app 中按以下顺序注册（从外到内，请求依次经过）：

    AuditLogMiddleware        生成 request_id，请求结束后写审计日志
      -> RequestContextMiddleware
                              按可信代理规则解析客户端 IP 与协议、匹配路由模板（只解析一次）
      -> CORSMiddleware
      -> IPFilterMiddleware   执行黑 / 白名单，拒绝不合法的代理头
      -> RateLimitMiddleware(before_auth=True)
                              认证之前：检查 IP 临时封禁，按 IP 限流挡洪水与扫描
      -> JwtAuthMiddleware / ApiKeyMiddleware
                              认证与路由级授权
      -> RateLimitMiddleware  认证之后：按用户 / 密钥身份及操作类别限流（API Key 模式另含日配额）
      -> UserQuotaMiddleware  仅 JWT 模式：按用户扣减每日配额
      -> FastAPI 应用

各模块：
- common：中间件共用的错误响应、安全响应头、环境变量校验与请求上下文检查；
- request_context：解析客户端 IP、协议与路由模板，供下游所有安全中间件使用；
- ip_filter：可信代理下的真实客户端 IP 解析规则与 IP 黑 / 白名单；
- rate_limit：基于 limits 共享存储（内存或 Redis）的限流、API Key 日配额与临时封禁；
- audit_log：每个请求一条 JSON 审计日志，只记录路由模板和凭据摘要。

中间件之间通过 ASGI scope 中以 security_ 开头的键传递信息，例如 security_client_ip、
security_route、security_error_code。security_client_ip / security_route 由 RequestContextMiddleware 写入，
IP 过滤、限流与认证中间件通过 common.require_request_context 确认它们存在，注册顺序被改错时请求直接失败。
"""
