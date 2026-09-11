# 第二阶段：请求限流、密钥配额、IP 控制与审计

本阶段在[第一阶段访问控制](SECURITY_STAGE1_PLAN.md)之上实现请求级防滥用。下文的密钥策略适用于显式 `AUTH_MODE=api_key` 的受信任团队兼容模式，不能作为多租户权限方案。

当前默认已升级为 [第三阶段 JWT 账号认证](SECURITY_STAGE3_PLAN.md)：IP 策略、Redis、读/写/昂贵速率限制及审计继续生效；认证后的计数改绑用户 ID，三类用户日配额在独立认证库中原子扣减，Agent 内部工具也检查权限与配额。JWT 模式忽略 `BACKEND_API_KEYS` 和密钥策略文件，不能用共享密钥绕过账号认证。

## 已实现的防护

| 能力 | 实际行为 |
| --- | --- |
| IP 限流 | 认证前限制来源 IP 的分钟请求量及每秒突发；更换无效密钥不能获得新额度 |
| 密钥限流 | 按完整 SHA-256 摘要计数，同一密钥跨 IP 共享额度，不记录密钥前缀 |
| 操作限流 | 读、写、昂贵操作分别按密钥共享预算；切换端点或论文 ID 不会重置类别预算 |
| 每日配额 | 原子扣减，按 UTC 自然日重置；Redis 支持跨 worker 共享 |
| 密钥策略 | 支持独立名称、速率、日配额、禁用和过期；每个请求重新判断是否过期 |
| IP 策略 | 支持 IPv4、IPv6、映射地址和 CIDR，黑名单优先；无效配置拒绝启动 |
| 临时封禁 | 认证失败或持续触发速率限制超过阈值时，自动封禁来源 IP |
| 审计 | 记录成功、拒绝、异常及完整 SSE 生命周期；日志轮转、凭据脱敏 |
| 前端 | 显示超限及等待时间；禁用/过期密钥退出登录，限流或 IP 拒绝保留凭据 |

入口顺序：审计 → CORS → IP 过滤 → IP 限流 → API Key 认证 → 密钥/操作/每日配额 → 业务参数解析与依赖。使用纯 ASGI 中间件，不缓冲 SSE 响应。

精确的 `GET /health` 是公开存活探针，不消耗额度。CORS 预检不消耗额度；普通 `OPTIONS` 请求不会整体绕过认证。审计仍记录这些请求。`/health` 不验证 Redis、模型或数据库是否可用。

## 1. 配置限流

依赖位于仓库根目录 [requirements.txt](../requirements.txt)，使用 `slowapi`、`limits` 和 `redis`：

```bash
pip install -r requirements.txt
```

复制并填写根目录 [.env.example](../.env.example)，以下是开发默认值：

```dotenv
AUTH_MODE=api_key
RATE_LIMIT_STORAGE=memory://
RATE_LIMIT_IP="60/minute;10/second"
RATE_LIMIT_KEY=60/minute
RATE_LIMIT_READ=60/minute
RATE_LIMIT_WRITE=20/minute
RATE_LIMIT_EXPENSIVE=5/minute
AUTH_FAILURE_LIMIT=20/minute
RATE_LIMIT_VIOLATION_LIMIT=30/minute
ABUSE_BLOCK_SECONDS=900
```

规则为固定窗口，可以用分号组合多个限制。所有窗口必须同时通过，提高某个密钥的 `rate_limit` 不能绕过 IP 或操作类别上限。共享出口下的使用者共同消耗 IP 预算，应按团队规模调整。

昂贵操作包括 Agent 对话和恢复执行、论文下载/创建、QA 建索引、普通/流式 QA、研究画像重建、兴趣向量生成及论文推荐。论文详情 GET、点赞/点踩和论文行为 POST 也可能回源并生成 embedding，因此同样按昂贵操作计数。准入阶段不读取业务数据库，即使本次命中本地缓存，也消耗该类别预算。其他 GET 和 POST arXiv 搜索按读操作计数，其余写请求受写预算约束。所有类别另受密钥总预算约束。

计数采用入口准入语义：请求通过安全检查后，即使业务返回 4xx/5xx 或客户端中断，也已经消耗额度；图片读取、轮询和登录校验同样计数。各层依次扣减，后续拒绝不会回滚之前已消耗的速率计数。每日配额统计通过全部限流检查的请求数，**不统计模型 token、Agent 内部工具调用数或实际费用**。

### 公网部署使用持久化 Redis

`memory://` 只适合单进程开发，重启会丢失计数，多进程会各自发放额度。新部署先运行 `python scripts/init_security.py --profile production`。初始化生成 `.env.production`，其中 `REDIS_PASSWORD` 和含相同密码的 `RATE_LIMIT_STORAGE` 一起生成，不需要手工拼接或在终端回显密码。

仓库 [docker-compose.yml](../docker-compose.yml) 提供仅绑定回环地址的 Redis，可独立启动，不会同时启动 Milvus：

```bash
docker compose --env-file .env.production --profile security up -d redis
docker compose --env-file .env.production --profile security exec redis redis-cli ping
```

模板启用密码认证、AOF 持久化、`appendfsync always`、命名数据卷和 `noeviction`，避免内存压力导致计数被淘汰。每次刷盘提高持久性，也增加 I/O 成本；这仍是请求准入计数，不是费用账单。不要删除 Redis 数据卷；监控可用空间、延迟、内存和持久化状态。容器从标准输入读取配置，密码不进入启动参数；`redis-cli` 使用容器内的 `REDISCLI_AUTH`，不要用 `-a` 回显密码。

多个 worker/实例必须连接同一个 Redis 数据库并使用一致的策略；不同部署应使用不同数据库。Redis 不可用或计数写入失败时返回 `503 security_storage_unavailable`，**不回退到内存或放行业务**。只支持 `memory://`、`redis://`、`rediss://`；跨主机 Redis 需配置受保护的网络、认证及适当的 TLS。模板适用于同机运行后端的部署，不将 6379 暴露到公网。

## 2. 每个密钥的策略

保留 `BACKEND_API_KEYS` 的逗号分隔配置；未启用文件配置时，各密钥使用 `RATE_LIMIT_KEY`，没有每日配额或过期时间。

需要独立策略时，从 [api_keys.example.json](../backend/config/api_keys.example.json) 创建私有文件：

```json
{
  "keys": [
    {
      "key_env": "RESEARCH_TEAM_ACCESS_KEY",
      "name": "研究团队",
      "rate_limit": "60/minute",
      "daily_quota": 1000,
      "enabled": true,
      "created_at": "2026-09-11",
      "expires_at": "2027-09-11T00:00:00Z"
    }
  ]
}
```

```dotenv
API_KEY_CONFIG_FILE=backend/config/api_keys.json
# 仅在服务器填写独立生成的随机访问密钥，不能使用模型供应商密钥。
RESEARCH_TEAM_ACCESS_KEY=
```

密钥用 `python -c "import secrets; print(secrets.token_urlsafe(32))"` 生成，必须是 32–256 个无空白的可打印 ASCII 字符。示例引用环境变量；私有 JSON 也支持 `key` 字段，但同一条目只能选择 `key` 或 `key_env`。

文件路径相对仓库根目录。配置优先级及失败规则：

1. 显式 `API_KEY_CONFIG_FILE` 存在时，只使用该文件。
2. 未设置该变量时，如果存在 `backend/config/api_keys.json`，自动使用它。
3. 两者都未提供时才使用 `BACKEND_API_KEYS`。

文件缺失、格式错误、重复密钥、空列表、非法额度或日期会阻止启动，不会偷偷启用环境变量中的旧密钥。私有 `backend/config/api_keys.json` 已加入忽略规则；自定义文件应放在仓库外或自行加入忽略规则，并限制读取权限。

`daily_quota` 必须是正整数，`null` 表示不限制每日请求总数。日期和未带时区的时间按 UTC 解释，日期形式在当天 UTC 00:00 到期；日配额也在 UTC 00:00 重置。`created_at` 仅作元数据。禁用、策略修改和密钥轮换需要重启应用；过期时间对每个新请求即时判断。已经开始的任务不会自动取消。

文件加载的密钥也加入日志、错误响应、trace 和跨分片 SSE 的脱敏集合。多个密钥名称可以相同，但审计中的完整摘要不同；业务权限不会因名称而改变。

## 3. IP 过滤与可信代理

```dotenv
IP_FILTER_MODE=disabled
IP_BLACKLIST=
IP_WHITELIST=
TRUSTED_PROXY_IPS=
```

`IP_FILTER_MODE` 可为 `disabled`、`blacklist` 或 `whitelist`。列表用逗号分隔，支持单 IP 和 CIDR，例如 `198.51.100.10,2001:db8::/32`。白名单模式必须提供非空白名单；同时命中黑白名单时拒绝。`disabled` 只关闭静态 IP 过滤，不关闭速率控制及自动封禁。

默认不信任任何转发头。直连客户端不能用 `X-Forwarded-For`、`X-Real-IP` 或 `Forwarded` 改变身份。只有 TCP 对端在 `TRUSTED_PROXY_IPS` 中时才读取 XFF，并从右向左剥离可信代理，采用第一个不可信节点。歧义、非法或过长的可信代理头会被拒绝。

同机 Nginx 代理使用：

```dotenv
TRUSTED_PROXY_IPS=127.0.0.1,::1
```

[Nginx 模板](../deploy/nginx.conf) 使用 `proxy_set_header X-Forwarded-For $remote_addr` 覆盖客户端传入的 XFF，并覆盖 `X-Forwarded-Proto`。应用只接受可信对端提供的单个 `http`/`https` 协议值，让 HTTPS 请求的尾斜杠重定向保持 HTTPS。多层代理需按真实拓扑配置可信节点与转发行为，不能填 `*` 或无差别信任所有地址。

应用必须看到原始 TCP 对端。`python backend/main.py` 已禁用 Uvicorn 自动解释代理头；通过 CLI 启动时使用：

```bash
cd backend
uvicorn main:app --host 127.0.0.1 --port 8001 --no-proxy-headers
```

[systemd 模板](../deploy/arxiv-agent.service) 为 Gunicorn/Uvicorn 配置了空的 `forwarded-allow-ips`，由应用统一解释可信代理。不要在其他启动命令中启用“信任所有代理”。配置 HTTPS、证书和端口防火墙仍是公网部署的必要步骤。

## 4. 异常检测与错误响应

默认同一 IP 一分钟内第 21 次认证失败，或第 31 次速率拒绝，会触发 900 秒临时封禁；触发阈值的请求保留原错误，后续请求返回封禁。存储中的封禁到期后自动解除，封禁期间的请求不会延长封禁时间。日配额耗尽本身不计为速率违规。

| HTTP | code | 客户端行为 |
| --- | --- | --- |
| 401 | `missing_api_key` | 登录并提供访问密钥 |
| 403 | `invalid_api_key`、`api_key_disabled`、`api_key_expired` | 当前凭据失效，重新登录或联系管理员 |
| 403 | `ip_blocked`、`ip_not_allowed` | 保留凭据，检查静态 IP 策略 |
| 403 | `ip_temporarily_blocked` | 等待 `Retry-After` 后重试 |
| 429 | `rate_limit_exceeded`、`daily_quota_exceeded` | 保留凭据，显示等待时间 |
| 503 | `security_storage_unavailable` | 保留凭据，等待访问控制存储恢复 |

可等待的安全错误包含整数秒 `Retry-After` 响应头及 JSON `retry_after`。前端保留错误码并显示等待时间，不自动重放写请求。旧密钥请求的迟到错误不会清除新密钥。

限流响应提供 `X-RateLimit-Limit/Remaining/Reset/Scope`；设置每日额度的密钥还提供 `X-DailyQuota-Limit/Remaining/Reset`。`Reset` 是 Unix 秒时间戳；响应显示当前判定中额度较紧或发生拒绝的规则，不能视为所有规则的完整快照。这些响应头已通过 CORS 暴露给允许的前端。

## 5. 审计日志

```dotenv
AUDIT_LOG_ENABLED=true
AUDIT_LOG_FILE=logs/audit.log
AUDIT_LOG_MAX_BYTES=10485760
AUDIT_LOG_BACKUP_COUNT=5
```

默认最多保留当前文件和五个轮转文件。POSIX 下目录新建为 0700，当前文件及轮转文件为 0600；Windows 需用部署账号的目录 ACL 管理权限。路径相对仓库根目录，初始化不可写时拒绝启动。重复创建同配置应用不会重复安装 handler。

每行一个 JSON，字段包括 UTC 时间、服务端生成的 `request_id`、规范化 IP、方法、路由模板、状态码、总耗时、密钥摘要/名称、认证状态、拒绝码、完成状态和发送字节数。响应带 `X-Request-ID` 便于关联。日志不保存原始密钥、密钥前缀、查询参数、路径参数、请求头或请求体。

SSE 正常结束、异常和取消都只记录一次；耗时覆盖完整执行周期，只有成功交给传输层的响应才记为完成。未捕获异常由最外层错误处理器生成 500 时，审计记录异常及尚未完整发送的状态。

单个文件的轮转仅支持一个进程。多个 worker/实例设置 `AUDIT_LOG_FILE=-`，写 stdout，再由 systemd/journald 或日志平台负责留存和告警。运行时文件写入/轮转失败，会向 stderr 输出 `audit_log_write_failed` 和脱敏审计事件，不改写已经执行的业务结果；若两个通道均不可写，无法保证落盘，应监控日志采集状态。

```bash
# 单进程文件日志
tail -f logs/audit.log

# stdout 审计及写入失败告警
sudo journalctl -u arxiv-agent -f
```

`AUDIT_LOG_ENABLED=false` 可显式关闭审计，公网部署保持开启。历史日志/trace 的清理和曾泄露凭据的轮换仍按第一阶段说明处理。

## 6. 验证

后端回归使用临时配置、隔离内存/模拟 Redis，不请求真实模型或开发者 Redis：

```powershell
$env:AGENT_RUNTIME_CHECKPOINT_BACKEND = 'memory'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:AUDIT_LOG_FILE = 'temp/security-tests/audit.log'
python -m pytest backend/tests/api/test_security_stage1.py backend/tests/api/test_security_stage2.py -q --basetemp=temp/pytest-security

cd new_frontend
npm run test
npm run build
```

测试覆盖改变无效密钥绕过 IP 桶、同密钥跨 IP、同前缀不同密钥、真实昂贵端点、并发日配额、跨计数器共享 Redis、UTC 重置、过期即时判断、IP/CIDR/代理欺骗、封禁解除、存储故障、日志轮转与写入故障，以及 SSE 正常/取消/发送失败。

部署后先检查真实 Redis 连通性，再通过 HTTPS 浏览器验证：登录、普通查询和 SSE 正常；超限出现 429 及等待提示；到期或禁用密钥被拒绝；审计来源是客户端 IP；可信代理前伪造 XFF 不能改变计数身份。不要用真实昂贵请求做循环压测，可用不调用模型的 `/api/auth/check` 验证限流。

## 防护边界与实现位置

本阶段控制请求准入，不能保证单次 Agent 执行的精确费用上限，也不提供并发任务上限、分布式 DDoS 防御、用户/租户授权或资源所有权校验。继续对受信任团队开放；面向不受信任用户前需落实用户身份及资源权限，并结合反向代理/云入口的流量控制。没有据此作出 QPS 或公网抗压能力承诺。

- [应用入口与中间件顺序](../backend/main.py)
- [密钥策略加载](../backend/auth/key_config.py)、[认证](../backend/auth/api_key_middleware.py)
- [限流、配额与自动封禁](../backend/middleware/rate_limit.py)
- [IP 和可信代理规则](../backend/middleware/ip_filter.py)
- [审计与轮转](../backend/middleware/audit_log.py)
- [安全回归](../backend/tests/api/test_security_stage2.py)
- [前端错误契约](../new_frontend/src/api/errors.ts)、[凭据状态](../new_frontend/src/api/auth.ts)
- [跨平台部署](DEPLOYMENT.md)
