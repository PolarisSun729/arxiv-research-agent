"""后端访问控制。

本包支持两种互斥的认证模式，由环境变量 AUTH_MODE 选择（见 settings.auth_mode）：

- jwt：多用户账号模式。账号、会话与每日配额保存在独立的 SQLite 认证库（store.py）中；
  JwtAuthMiddleware 负责令牌校验、按路由权限清单授权以及请求中身份字段的绑定。
- api_key：共享访问密钥模式。ApiKeyMiddleware 按 X-API-Key 匹配 key_config 中的密钥策略，
  速率限制与密钥日配额由 middleware.rate_limit 执行。

各模块分工：
- settings / key_config：启动时加载配置并严格校验，配置有误时直接启动失败；
- models / schemas：账号数据结构，以及账号接口的入参校验；
- passwords / jwt_handler / store：密码哈希、令牌签发与校验、账号/会话/配额的持久化；
- permissions：HTTP 路由与 Agent 工具的角色权限、配额类型清单；
- context / ownership / tool_access：在请求与 Agent 执行过程中传递可信身份，并在业务层再次校验；
- charging：在真正产生外部成本的位置扣减配额；
- request_body / errors：两种认证中间件共用的有界请求体读取与统一错误响应。
"""
