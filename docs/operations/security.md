# 安全与访问控制

默认认证模式是 `AUTH_MODE=jwt`：独立账号、角色权限、用户日配额、会话撤销与审计。`AUTH_MODE=api_key` 是面向受信任小团队的共享密钥兼容模式，只在显式配置时启用。JWT 模式下，共享密钥、客户端传入的 `user_id` 和模型生成的工具参数都不能替代已验证的账号身份。

## 1. 入口顺序与公开路径

```text
审计 → CORS → IP 过滤 → IP 限流 → 认证（JWT 或 API Key）/角色/身份 → 用户（或密钥）速率 → 日配额 → 业务
```

- 全部是纯 ASGI 中间件，不缓冲 SSE。角色拒绝发生在业务依赖初始化前；日配额不足在 SSE 响应头发出前返回 429。
- JSON 请求体默认最多 1 MiB（`AUTH_MAX_REQUEST_BYTES`），读取超时 30 秒；修改时同步检查 Nginx 限制。
- 公开入口只有精确的 `GET /health`、`GET /api/auth/config`、`POST /api/auth/login`、`POST /api/auth/register`（注册默认 403）。`/health` 只证明进程存活，不验证 Redis 或认证库。
- CORS 预检由外层处理，普通 `OPTIONS` 不绕过认证。CORS 只约束浏览器，真正的授权来自认证层。
- 不发布 `/docs`、`/redoc`、`/openapi.json`；API 响应带 `Cache-Control: no-store`；422 只返回字段位置与校验类型，不回显请求体。
- 直接启动后端默认只监听 `127.0.0.1`；Docker 中 Milvus、MinIO、Redis 及管理端口只绑定本机。

## 2. 首次部署

认证依赖（PyJWT、bcrypt、email-validator）在 [requirements.txt](../../requirements.txt)，账号库使用标准库 SQLite。

```powershell
# 本机开发：生成 .env、开发签名文件、localhost Origin 与内存限流
python scripts/init_security.py --profile development
python scripts/manage_users.py create-admin --username admin --email <你的邮箱> --env-file .env
python backend/main.py --load-mode lazy
```

```bash
# 服务器生产配置
python scripts/init_security.py --profile production --domain <你的域名>
docker compose --env-file .env.production --profile security up -d redis
python scripts/manage_users.py create-admin --username admin --email <你的邮箱> --env-file .env.production
```

- [初始化脚本](../../scripts/init_security.py)不创建默认账号，不覆盖已有环境文件或签名文件；签名、Redis、MinIO 使用独立随机值，Linux 独占创建 `0600` 文件，Windows 先设 NTFS ACL。
- 生产配置使用 `backend/config/production.jwt-secret`、`backend/data/auth/production.sqlite3`、HTTPS Origin、可信本机代理和带密码的持久化 Redis。开发与生产不共享签名或账号库。
- 配置加载优先级：进程环境 > 根 `.env` > `backend/.env`。systemd 用 `EnvironmentFile` 显式加载 `.env.production`；直接运行 `python backend/main.py` 不会自动选生产文件。用户管理 CLI 必须与服务使用同一 `AUTH_DATABASE_PATH`。
- `JWT_SECRET_KEY` 与 `JWT_SECRET_FILE` 二选一；有效期默认 60 分钟（1–1440）。缺少签名、配置非法或认证库不可用时拒绝启动。
- 模型供应商密钥只写在后端私有 `.env`。**不要配置任何 `VITE_` 密钥变量**，Vite 变量会进入浏览器产物，构建配置会拒绝这类凭据。
- 生成配置不等于公网可用：DNS、证书、Nginx、systemd 见 [CI/CD 部署手册](cicd-deployment.md)。

紧急密码恢复（会撤销该账号所有旧登录）：

```bash
python scripts/manage_users.py set-password --username admin --env-file .env.production
```

## 3. 账号、会话与角色

| 能力 | 当前行为 |
| --- | --- |
| 账号 | 独立 SQLite 保存账号、bcrypt 哈希（rounds=12）、会话与日配额；用户 ID 由服务器随机生成 |
| 注册 | 默认关闭；开放后只能创建 guest，拒绝 `role`、`user_id` 等额外字段 |
| 密码 | ≥8 字符，含大写、小写、数字；UTF-8 ≤72 字节，不截断 |
| 令牌 | 固定 HS256，校验签名、issuer、audience、时间、账号状态、版本和会话记录 |
| 撤销 | 退出撤销当前登录；改密、停用、角色变更撤销该账号所有登录。最后一名启用的管理员不能停用或降级 |
| RBAC | 每个路由和 Agent 工具显式登记角色，未登记入口默认拒绝，管理员也不例外 |

| 能力 | admin | researcher | viewer | guest |
| --- | --- | --- | --- | --- |
| 共享论文查询、详情、推荐 | ✓ | ✓ | ✓ | ✓ |
| 已有索引 QA；本人会话/笔记/偏好/画像 | ✓ | ✓ | ✓ | ✓ |
| 导入/下载论文、建索引、画像重建 | ✓ | ✓ | ✗ | ✗ |
| Agent 对话与续跑 | ✓ | ✓ | ✗ | ✗ |
| 删除共享论文、全局 trace/诊断、debug 路由 | ✓ | ✗ | ✗ | ✗ |
| 用户、角色与配额管理 | ✓ | ✗ | ✗ | ✗ |

- 论文元数据、索引和证据图片是共享资料；本人数据的 `user_id` 一律来自已验证的 `/me`。管理员管理账号使用独立的 `target_user_id`，普通业务接口不能代他人操作。
- 全局 trace 可能含他人问题，仅管理员可读，debug 路由还需 `ENABLE_DEBUG_ROUTES=true`。
- JWT 新账号不会自动认领 `local_user`、`u-demo` 或历史匿名 checkpoint；历史数据需由操作员核实归属后定向迁移。
- 没有团队租户、私有论文库、自动刷新令牌、邮箱验证、MFA 或公开忘记密码接口。

### API

| 方法与路径 | 用途 |
| --- | --- |
| `GET /api/auth/config` | 公开：认证模式、是否允许注册 |
| `POST /api/auth/register` | 可选 guest 注册 |
| `POST /api/auth/login` | 返回 `access_token`、`token_type`、`expires_in` |
| `GET /api/auth/me` | 当前账号、角色与三类配额 |
| `POST /api/auth/logout` | 撤销当前会话 |
| `POST /api/auth/password` | 修改密码，成功后需重新登录 |
| `GET/POST /api/auth/users` | 管理员分页查询或创建账号 |
| `GET/PATCH /api/auth/users/{target_user_id}` | 管理员查询、调整 role/is_active |
| `PUT /api/auth/users/{target_user_id}/quotas` | 管理员设置 quota_type、daily_limit |

受保护请求只接受 `Authorization: Bearer <token>`，不接受 URL token、Cookie 或 `X-API-Key` 替代。

### 前端会话

- 仅在 `/me` 校验通过后保存身份与令牌，使用内存 + `sessionStorage`，不用 `localStorage`。切换账号会重建页面，旧请求的迟到成功或认证失败不能覆盖新身份。
- 带凭据的请求限定在配置的 API 根地址内，拒绝自动重定向；图片和下载使用认证 Blob，令牌不进入 URL、历史或 Referer。
- 退出需服务端撤销成功；失败时保留状态供重试。`sessionStorage` 可被同源脚本读取，前端静态资源和依赖必须可信。

## 4. 速率限制与日配额

### 日配额

| 角色 | papers | qa_queries | agent_runs |
| --- | ---: | ---: | ---: |
| admin | -1 | -1 | -1 |
| researcher | 500 | 200 | 20 |
| viewer | 50 | 20 | 0 |
| guest | 10 | 5 | 0 |

- `-1` 不限，`0` 无额度；权限先于配额判断，给 guest 加 Agent 配额不会授予 Agent 权限。
- `papers` 覆盖论文列表/搜索/详情/推荐、收藏与行为记录、导入/下载/建索引、画像重建；`qa_queries` 覆盖普通与流式问答；`agent_runs` 覆盖 Agent 对话与续跑，内部论文与 QA 工具另扣对应类别。
- 在认证 SQLite 中用 `BEGIN IMMEDIATE` 原子校验并扣减，UTC 00:00 重置。配额记录丢失或存储故障返回 503，不补发额度。角色变更应用新默认限额但保留今日用量。

### 速率

```dotenv
RATE_LIMIT_STORAGE=memory://
RATE_LIMIT_IP="60/minute;10/second"
RATE_LIMIT_USER=60/minute
RATE_LIMIT_READ=60/minute
RATE_LIMIT_WRITE=20/minute
RATE_LIMIT_EXPENSIVE=5/minute
RATE_LIMIT_AUTH=10/minute
RATE_LIMIT_LOGIN_ACCOUNT=10/minute
AUTH_FAILURE_LIMIT=20/minute
RATE_LIMIT_VIOLATION_LIMIT=30/minute
ABUSE_BLOCK_SECONDS=900
```

- 固定窗口，可用分号组合；所有层必须同时通过。`RATE_LIMIT_USER` 绑定稳定用户 ID，换 token 或 IP 不重置；`RATE_LIMIT_AUTH` 按 IP 约束公开认证入口，`RATE_LIMIT_LOGIN_ACCOUNT` 按规范化用户名约束跨 IP 猜密码。
- 昂贵操作：Agent 对话与续跑、论文下载/创建、建 QA 索引、普通/流式 QA、画像重建、兴趣向量生成、推荐；论文详情 GET、点赞/点踩、行为 POST 可能回源并生成 embedding，也按昂贵计。准入阶段不读业务库，缓存命中也计数。
- 准入语义：通过后业务失败、断线或缓存命中都不退还；前一层的计数不因后一层拒绝而回滚。配额统计请求/工具调用次数，**不是 token、篇数或费用**。
- 同一 IP 一分钟内第 21 次认证失败或第 31 次速率拒绝，触发 900 秒临时封禁；封禁期间请求不延长封禁。日配额耗尽不算速率违规。

### 公网必须用持久化 Redis

`memory://` 只适合单进程开发。[docker-compose.yml](../../docker-compose.yml) 的 `security` profile 提供只绑定回环的 Redis：密码认证、AOF + `appendfsync always`、命名数据卷、`noeviction`；密码经标准输入传入，`redis-cli` 用容器内 `REDISCLI_AUTH`，不要用 `-a`。

- 多 worker/实例必须连同一个 Redis 库，不同部署用不同库。只支持 `memory://`、`redis://`、`rediss://`。
- Redis 不可用返回 `503 security_storage_unavailable`，**不回退到内存，也不放行业务**。
- Redis 负责速率与封禁，JWT 日配额在认证库；两者都不可绕过。

## 5. IP 过滤与可信代理

```dotenv
IP_FILTER_MODE=disabled      # disabled | blacklist | whitelist
IP_BLACKLIST=
IP_WHITELIST=
TRUSTED_PROXY_IPS=127.0.0.1,::1   # 同机 Nginx
```

- 支持 IPv4/IPv6/映射地址/CIDR，黑名单优先；白名单模式必须非空；无效配置拒绝启动。`disabled` 只关闭静态过滤，不关闭限流与自动封禁。
- 默认不信任任何转发头。只有 TCP 对端在 `TRUSTED_PROXY_IPS` 中才读取 XFF，从右向左剥离可信代理后取第一个不可信节点；歧义、非法或过长的头被拒绝。
- 应用必须看到原始 TCP 对端：`python backend/main.py` 已禁用 Uvicorn 代理头解析，CLI 启动需加 `--no-proxy-headers`；[systemd 模板](../../deploy/arxiv-agent-cicd.service) 把 `forwarded-allow-ips` 设为空。不要在任何启动方式中信任所有代理。
- [Nginx 模板](../../deploy/nginx.conf) 用 `$remote_addr` 覆盖 `X-Forwarded-For` 和 `X-Forwarded-Proto`。

## 6. 错误契约

| HTTP | code | 客户端行为 |
| --- | --- | --- |
| 401 | 缺失/无效/过期/已撤销登录；兼容模式下 `missing_api_key` | 重新登录 |
| 403 | 权限或身份不匹配；兼容模式下 `invalid_api_key`、`api_key_disabled`、`api_key_expired` | 凭据失效或无权限 |
| 403 | `ip_blocked`、`ip_not_allowed` | 保留凭据，检查静态 IP 策略 |
| 403 | `ip_temporarily_blocked` | 等待 `Retry-After` |
| 429 | `rate_limit_exceeded`、`quota_exceeded`（附 `quota_type`）、兼容模式 `daily_quota_exceeded` | 保留凭据，显示等待时间 |
| 503 | `security_storage_unavailable` 等存储故障 | 保留凭据，等待恢复 |

可等待的错误带整数秒 `Retry-After` 头和 JSON `retry_after`。限流响应带 `X-RateLimit-Limit/Remaining/Reset/Scope`（已通过 CORS 暴露），`Reset` 为 Unix 秒，只代表当前判定中最紧或触发拒绝的那条规则。前端不自动重放写请求。

## 7. 审计与脱敏

```dotenv
AUDIT_LOG_ENABLED=true
AUDIT_LOG_FILE=logs/audit.log    # 多 worker 设为 - 写 stdout，交给 journald
AUDIT_LOG_MAX_BYTES=10485760
AUDIT_LOG_BACKUP_COUNT=5
```

- 每行一个 JSON：UTC 时间、服务端 `request_id`（响应头 `X-Request-ID`）、规范化 IP、方法、路由模板、状态码、耗时、认证状态、拒绝码、完成状态、发送字节数；JWT 模式另记 `auth_mode`、`user_id`、`username`、`role`、`target_user_id`。不记录请求体、查询参数、路径参数、请求头或凭据。
- SSE 正常结束、异常、取消各只记一次，耗时覆盖完整周期。
- POSIX 下目录 0700、文件 0600；单文件轮转只支持单进程。写入失败向 stderr 输出 `audit_log_write_failed` 和脱敏事件，不改变已执行的业务结果。
- 日志、异常堆栈、JSON 响应、SSE（含跨分片）、笔记附件和 trace 统一经 [`secret_redaction.py`](../../backend/utils/secret_redaction.py) 脱敏，覆盖密码、签名材料、Bearer、裸 JWT 和已配置的访问密钥。只有登录专用响应可以返回新令牌。
- 历史日志和 trace 不会被自动重写，曾泄露的凭据必须轮换。

## 8. 持久化与后台任务

- JWT 带随机会话 ID，服务端保留会话记录以即时撤销，并非完全无状态。已开始的外部模型调用或已提交的索引作业不能追溯撤回。
- 后台续跑在同一进程内显式复制可信认证上下文，令牌不进入 LangGraph state。进程重启后，丢失认证上下文的待运行 resume run 标记为 `authentication_context_lost`，需重新登录后再发起恢复。
- 认证库默认 `backend/data/auth/auth.sqlite3`，与业务库独立。多 worker 必须同机并共用认证库、签名密钥和 Redis；SQLite 不放网络共享盘，多主机需先迁到共享数据库。
- 认证库（含 WAL/SHM）、签名文件和备份都需私有权限。备份用 SQLite 在线备份接口或停服后备份；从备份恢复可能复活旧登录，恢复时同时轮换签名密钥并重启全部实例（所有用户需重新登录）。
- PDF 下载校验新旧 arXiv ID、官方同篇 URL 与每次重定向，强制 HTTPS、50 MiB 上限、独占临时文件后原子替换。

## 9. API Key 兼容模式

仅 `AUTH_MODE=api_key` 时启用，适合受信任的小团队：密钥控制入口，但持有密钥的人可以选择任意 `user_id`，**不构成用户身份或租户隔离**。

- `BACKEND_API_KEYS` 逗号分隔，每个 32–256 个无空白可打印 ASCII 字符（`python -c "import secrets; print(secrets.token_urlsafe(32))"`），与模型供应商密钥分开。请求头为 `X-API-Key`；比较使用摘要 + `secrets.compare_digest`。
- 需要按密钥设置名称、速率、日配额、禁用或过期时，从 [api_keys.example.json](../../backend/config/api_keys.example.json) 创建私有 `backend/config/api_keys.json`（已忽略）或用 `API_KEY_CONFIG_FILE` 指定。优先级：显式 `API_KEY_CONFIG_FILE` > 默认 `api_keys.json` > `BACKEND_API_KEYS`；文件有误时拒绝启动，不回退到环境变量。条目用 `key` 或 `key_env` 二选一。
- 密钥计数按完整 SHA-256 摘要，跨 IP 共享；`RATE_LIMIT_KEY` 为默认密钥速率。过期每次请求即时判断，禁用和策略修改需重启。
- 轮换：先设 `BACKEND_API_KEYS=旧,新` 并重启，分发新密钥后移除旧密钥再重启。
- JWT 模式忽略 `BACKEND_API_KEYS` 和密钥策略文件。

## 10. 验证

自动化回归使用隔离账号、临时库和假业务依赖，不调用付费模型：

```powershell
python scripts/test_security.py          # 三个安全阶段 + 初始化脚本，凭据与数据库全部隔离
python scripts/test_security.py --full   # 同样隔离下运行全部后端离线回归
python -m pytest backend/tests/unit/test_arxiv_pdf_download.py -q
cd new_frontend; npm run test
```

上线后在真实 HTTPS 域名确认：管理员登录与建号；viewer 无法建索引、启动 Agent 或读全局 trace；两个账号看不到对方私人数据；问答、SSE、图片、导出正常；退出或停用后旧令牌失效；Redis/认证库故障时业务拒绝执行。不要用真实昂贵请求做循环压测，可用不调用模型的接口验证限流。

## 实现位置

- [应用入口与中间件顺序](../../backend/main.py)
- [认证与角色模块](../../backend/auth/)、[认证路由](../../backend/routers/auth_router.py)
- [限流、配额与自动封禁](../../backend/middleware/rate_limit.py)、[IP 与可信代理](../../backend/middleware/ip_filter.py)、[审计](../../backend/middleware/audit_log.py)
- [用户管理脚本](../../scripts/manage_users.py)
- [前端认证传输](../../new_frontend/src/api/auth.ts)、[前端错误契约](../../new_frontend/src/api/errors.ts)
- 回归：[stage1](../../backend/tests/api/test_security_stage1.py)、[stage2](../../backend/tests/api/test_security_stage2.py)、[stage3](../../backend/tests/api/test_security_stage3.py)、[前端](../../new_frontend/tests/security.test.mjs)
