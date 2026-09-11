# 第三阶段：JWT 账号认证、角色权限与用户配额

第三阶段已将默认入口升级为 `AUTH_MODE=jwt`。本文件说明当前实现与部署步骤；第一、二阶段的共享 API Key 仅在显式 `AUTH_MODE=api_key` 时启用。JWT 模式下，旧密钥、客户端 `user_id` 和模型生成的参数都不能替代已验证的账号身份。

## 1. 已实现的安全边界

| 能力 | 当前行为 |
| --- | --- |
| 账号 | 独立 SQLite 保存账号、bcrypt 密码哈希、会话和三类日配额；用户 ID 由服务器随机生成 |
| 注册 | 默认关闭；显式开放后只能创建 guest，拒绝 `role`、`user_id` 等额外字段 |
| 登录 | JSON 用户名/密码登录，固定 HS256；校验签名、issuer、audience、时间、账号状态、版本和会话记录 |
| 密码 | 至少 8 个字符，包含大写、小写、数字，UTF-8 最多 72 字节；bcrypt rounds=12，不截断长密码 |
| 会话撤销 | 退出撤销当前登录；改密码、停用、角色变更撤销该账号所有旧登录 |
| RBAC | 每个真实路由和 Agent 工具显式登记角色，未登记入口默认拒绝，管理员也不能绕过 |
| 用户数据 | 会话、笔记、偏好、画像按认证身份访问；拒绝伪造身份、跨用户会话及跨论文笔记来源 |
| 配额 | 在事务中校验会话并扣减，按 UTC 自然日重置；并发请求不能超领额度 |
| Agent | 每个节点/工具重新校验会话和角色；工具参数不能冒用身份，内部论文与 QA 工具另扣对应配额 |
| 防滥用 | 保留 IP、读/写/昂贵操作限流；增加用户总速率、登录/注册 IP 速率和登录账号速率 |
| 审计与脱敏 | 成功/拒绝/异常/SSE 完整生命周期记录可信用户和角色；密码、签名材料、Bearer 和裸 JWT 脱敏 |
| 前端 | 登录恢复、角色路由、个人配额/改密、管理员用户管理；普通请求、三条 SSE、Blob 与下载统一认证 |
| 运维诊断 | `/stats` 对普通账号只提供固定同步失败提示；详细同步诊断只向已认证管理员展示 |
| PDF 下载 | 校验新旧 arXiv ID、官方同篇论文 URL 和每次重定向；强制 HTTPS、50 MiB 上限、独占临时文件与原子替换 |

入口顺序：审计 → CORS → IP 过滤 → IP 限流 → JWT/角色/身份检查 → 用户速率 → 用户日配额 → 业务。JSON 请求体默认最多 1 MiB，读取超时 30 秒；SSE 响应不被缓冲。角色拒绝发生在业务依赖初始化之前，日配额不足在 SSE 响应头发出之前返回 429。

公开入口只有精确的 `GET /health`、`GET /api/auth/config`、`POST /api/auth/login`、`POST /api/auth/register`；注册默认返回 403。CORS 预检由外层处理，普通 OPTIONS 不绕过认证。`/health` 只证明进程存活，不验证 Redis 或认证库是否就绪。

## 2. 首次部署与管理员初始化

认证依赖 PyJWT、bcrypt、email-validator，用户存储使用标准库 SQLite；依赖位于仓库根目录 [requirements.txt](../requirements.txt)。本机使用 `conda activate new_rag`，然后运行 [初始化脚本](../scripts/init_security.py)：

```powershell
# 本机开发：创建 .env、开发签名文件，使用 localhost Origin 与内存限流。
python scripts/init_security.py --profile development
python scripts/manage_users.py create-admin --username admin --email admin@001769.xyz --env-file .env
python backend/main.py --load-mode lazy
```

模型密钥在私有 `.env` 中填写，不能传给前端。管理员邮箱替换为自己的地址，密码通过隐藏输入确认。脚本不会创建默认账号，也不会覆盖已有环境文件或签名文件。签名、Redis 和 MinIO 使用独立随机值；Windows 先设置 NTFS ACL 再写入秘密，Linux 独占创建 `0600` 文件。

服务器生产配置单独生成：

```bash
python scripts/init_security.py --profile production --domain arxiv.001769.xyz
docker compose --env-file .env.production --profile security up -d redis
python scripts/manage_users.py create-admin --username admin --email admin@001769.xyz --env-file .env.production
```

生成的 `.env.production` 使用 `backend/config/production.jwt-secret`、`backend/data/auth/production.sqlite3`、HTTPS Origin、可信本机代理与带密码的持久化 Redis。开发和生产不共享签名材料或账号库。生成配置不代表公网已经可用：必须按 [部署指南](DEPLOYMENT.md) 完成 DNS、证书签发、Nginx 和 systemd，且先在文件中填写模型供应商密钥。

进程环境优先于根 `.env`，根 `.env` 优先于 `backend/.env`。systemd 用 `EnvironmentFile` 显式加载 `.env.production`；直接 `python backend/main.py` 不会自动选择生产文件。用户管理 CLI 必须使用与服务相同的 `AUTH_DATABASE_PATH`；显式环境文件缺失时拒绝执行。

JWT 默认有效期 60 分钟，允许 1–1440 分钟。手工配置时 `JWT_SECRET_KEY` 与 `JWT_SECRET_FILE` 只能选一个；签名必须使用独立的强随机值，不能复用模型或共享访问密钥。缺失签名、非法配置或不可用认证库会拒绝启动。已有部署应补齐配置并协调凭据轮换，不能反复初始化。

管理员在“用户管理”创建其他账号、修改角色、启停和调整配额。最后一名启用的管理员不能停用或降级。本地紧急密码恢复：

```bash
python scripts/manage_users.py set-password --username admin --env-file .env.production
```

密码恢复撤销该账号旧登录。生产不向前端提供任何 `VITE_` 密钥；登录、SSE、Blob 和下载均使用用户 JWT。

## 3. 角色与资源权限

| 能力 | admin | researcher | viewer | guest |
| --- | --- | --- | --- | --- |
| 共享论文查询、详情、推荐 | 允许 | 允许 | 允许 | 允许 |
| 使用已有索引 QA、本人会话/笔记/偏好/画像维护 | 允许 | 允许 | 允许 | 允许 |
| 主动导入/下载论文、建立索引、画像重建 | 允许 | 允许 | 拒绝 | 拒绝 |
| Agent 对话和续跑 | 允许 | 允许 | 拒绝 | 拒绝 |
| 删除共享论文、全局 trace/诊断、可选 debug | 允许 | 拒绝 | 拒绝 | 拒绝 |
| 用户管理、角色与配额设置 | 允许 | 拒绝 | 拒绝 | 拒绝 |

viewer 在界面称为“阅读者”：为落实原计划中的 QA 配额，允许写入自己的问答会话、笔记与阅读偏好，并非所有 HTTP 写请求都禁用。guest 使用同类阅读能力但配额更低。没有索引时需由研究者或管理员创建。论文详情和收藏可能补全共享论文元数据并生成 embedding，因此仍受论文配额及昂贵操作速率限制。

论文元数据、索引和证据图片是共享资料。本人数据的 `user_id` 来自已验证的 `/me`；管理员管理账号使用独立 `target_user_id`，也不能通过普通业务接口伪造其他人的身份。全局 trace 可能包含其他人的问题，仅管理员可读取，debug 还需显式开启 `ENABLE_DEBUG_ROUTES`。

JWT 新账号不会自动认领 `local_user`、`u-demo` 或历史匿名 checkpoint。切换前需保留旧业务数据库备份；若要迁移历史个人资料，应由服务器操作员核实归属后定向迁移，不能根据客户端传入的 ID 自动绑定。本阶段没有团队/组织租户、论文私有库或管理员代用户操作能力。

## 4. 日配额与速率限制

| 角色 | papers | qa_queries | agent_runs |
| --- | ---: | ---: | ---: |
| admin | -1 | -1 | -1 |
| researcher | 500 | 200 | 20 |
| viewer | 50 | 20 | 0 |
| guest | 10 | 5 | 0 |

`-1` 表示该类每日不限额，`0` 表示无额度。权限先于配额判断，增加 guest 的 Agent 配额不会授予 Agent 权限。管理员也受 IP、用户及昂贵操作速率限制，可由管理员把其日配额改为有限值。

- `papers`：论文列表/搜索/详情/推荐、收藏及行为记录、导入/下载/索引建立、画像重建等准入次数。
- `qa_queries`：普通或流式论文问答准入次数。
- `agent_runs`：Agent 对话或续跑准入次数；内部论文和 QA 工具还消耗对应类别额度。

配额在 UTC 00:00 重置，返回 `reset_at`。SQLite 的 `BEGIN IMMEDIATE` 在校验和扣减之间持有写锁，保证同机多 worker 原子消费。配额记录丢失或存储故障时返回 503，不自动补发额度。角色变更应用新默认限额，但保留今日已用量；管理员修改限额同样不会重置用量。

计数采用准入语义：通过后业务失败、断线或缓存命中均不退还额度。速率层先前的计数不因后续拒绝而回滚。配额统计请求/工具调用次数，不是模型 token、论文篇数或实际费用，也不等同于商业计费系统。

```dotenv
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

`RATE_LIMIT_USER` 绑定稳定用户 ID，多次登录、换 token 或换 IP 不会获得新额度。`RATE_LIMIT_AUTH` 对公开认证入口按 IP 共享预算，`RATE_LIMIT_LOGIN_ACCOUNT` 对规范化用户名共享登录预算，避免跨 IP 猜密码。公共出口的用户仍共用 IP 预算，应按实际规模配置。

公网继续使用持久化 Redis，详见 [第二阶段说明](SECURITY_STAGE2_PLAN.md)。JWT 的日配额在认证 SQLite 中持久化；Redis 负责速率和封禁，两个存储都不可绕过。`memory://` 仅适合单进程开发。

## 5. API 与前端会话

| 方法与路径 | 用途 |
| --- | --- |
| `GET /api/auth/config` | 公开返回认证模式和是否允许注册 |
| `POST /api/auth/register` | 可选 guest 注册，JSON：username、email、password |
| `POST /api/auth/login` | JSON：username、password；返回 access_token、token_type、expires_in |
| `GET /api/auth/me` | 当前账号、角色和三类配额 |
| `POST /api/auth/logout` | 撤销当前会话 |
| `POST /api/auth/password` | current_password、new_password；成功后需重新登录 |
| `GET/POST /api/auth/users` | 管理员分页查询或创建账号 |
| `GET/PATCH /api/auth/users/{target_user_id}` | 管理员查询、调整 role/is_active |
| `PUT /api/auth/users/{target_user_id}/quotas` | 管理员设置 quota_type、daily_limit |

受保护请求使用 `Authorization: Bearer <access_token>`。不接受 URL token、Cookie 登录或 `X-API-Key` 替代 JWT。错误按稳定 `code` 返回：401 为缺失/无效/过期/撤销的登录；403 为权限或身份不匹配；429 为速率或 `quota_exceeded`；存储失败返回 503。可等待错误提供 `Retry-After`，日配额不足还返回 `quota_type`。

只有登录专用白名单响应可以发送新令牌；其他 JSON、日志、trace 和 SSE 继续脱敏。账号响应不包含密码哈希、会话版本或令牌。

前端仅在 `/me` 校验通过后保存身份和令牌，当前标签页使用内存及 `sessionStorage`，不使用 `localStorage` 长期保存。刷新会恢复并验证登录；切换账号会重建页面，旧请求的迟到成功或认证失败不能覆盖新身份。所有带凭据的请求限定在配置的 API 根地址内，拒绝自动重定向，图片/下载使用认证 Blob。

退出需服务端撤销成功，网络或存储失败时保留状态供重试。`sessionStorage` 可被同源脚本读取，所以前端静态资源与依赖仍必须可信。本阶段没有自动刷新令牌、邮箱验证、MFA 或公开忘记密码接口。

## 6. 后台任务、持久化与运维

JWT 包含随机会话 ID，服务端保留会话记录用于即时撤销，不能把它理解为完全无状态 JWT。账号停用、改密或角色调整后，新请求及 Agent 后续节点都会拒绝旧登录；已经开始的一次外部模型调用或已提交索引作业不能被追溯撤回。

后台续跑在同一进程中显式复制可信认证上下文，不将令牌或授权上下文存入 LangGraph state。JWT 模式进程重启后，待运行但已丢失登录上下文的 resume run 标记为 `authentication_context_lost`，要求重新登录后发起新的恢复请求；已运行中断的 run 保留原有进程重启失败语义。系统不会只凭存储中的 `user_id` 自动恢复权限。

认证库默认位于 `backend/data/auth/auth.sqlite3`，与业务库独立。多 worker 必须位于同一主机、共用同一认证库、签名密钥和 Redis。SQLite 文件不适合放在网络共享盘；多主机部署需先将账号/会话/配额存储迁移为共享数据库，不能每台机器各放一份。

认证库、WAL/SHM、签名文件和备份均需私有权限，不能放在前端静态目录或公开下载目录。备份应使用 SQLite 在线备份接口或停服务后做一致性备份，不能运行中只复制主文件。认证库或会话备份恢复可能重新引入旧登录，恢复时同时轮换 JWT 签名密钥并重启全部实例。签名轮换会要求所有用户重新登录。

[systemd 模板](../deploy/arxiv-agent.service) 使用回环监听和 `UMask=0077`。多 worker 将 `AUDIT_LOG_FILE=-` 交由 journald/日志平台收集，避免并发文件轮转。审计在原有字段上增加 `auth_mode`、`user_id`、`username`、`role` 和账号管理目标 `target_user_id`，不记录请求正文、查询参数或凭据。历史日志和 trace 不会被自动重写，曾泄露的凭据仍需轮换。

[Nginx 模板](../deploy/nginx.conf) 与默认 JSON 大小限制一致，并覆盖可信代理头。HTTPS、精确 CORS、回环绑定 Redis/Milvus/MinIO、证书及防火墙配置仍由实际部署完成。修改 `AUTH_MAX_REQUEST_BYTES` 时同时检查反向代理限制。详见 [跨平台部署指南](DEPLOYMENT.md)。

## 7. 验证与实现位置

自动化验证使用隔离账号、临时数据库和假业务依赖，不调用付费模型：

```powershell
$env:AGENT_RUNTIME_CHECKPOINT_BACKEND='memory'
python -m pytest backend/tests --basetemp=temp/pytest-security-stage3
python scripts/backend_static_check.py
python scripts/check_docs.py
cd new_frontend
npm run test
npm run build
```

第三阶段回归覆盖注册提权、密码规则、签名/claims、退出/改密/停用撤销、最后管理员保护、路由拒绝、身份伪造、真实会话/笔记隔离、并发配额、UTC 重置、存储失败、内部工具和跨线程身份、用户速率、流式认证与脱敏。PDF 回归额外覆盖路径穿越、SSRF、逐跳重定向和超量下载。

上线后还需在真实 HTTPS 域名验证：管理员登录与创建用户；阅读者无法建索引/启动 Agent/读全局 trace；两个账号看不到对方私人数据；普通问答、SSE、图片和导出正常；退出或停用后旧令牌失效；Redis/认证库故障时业务拒绝执行。离线测试不代表已经完成部署或外部模型连通验证。

- [认证与角色模块](../backend/auth/)
- [认证路由](../backend/routers/auth_router.py)
- [用户初始化脚本](../scripts/manage_users.py)
- [应用入口](../backend/main.py)
- [前端认证传输](../new_frontend/src/api/auth.ts)
- [后端安全回归](../backend/tests/api/test_security_stage3.py)
- [PDF 下载回归](../backend/tests/unit/test_arxiv_pdf_download.py)
- [前端安全回归](../new_frontend/tests/security.test.mjs)

文档更新：2026-09-11。这里列出的配置与行为以当前实现为准。
