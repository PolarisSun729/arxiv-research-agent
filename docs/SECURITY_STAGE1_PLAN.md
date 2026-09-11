# 第一阶段：基础访问控制与密钥保护

本文件保留第一阶段共享密钥兼容模式的部署说明，须显式设置 `AUTH_MODE=api_key`。当前默认已升级为 [第三阶段 JWT 账号认证](SECURITY_STAGE3_PLAN.md)，新的公网部署优先按第三阶段配置。

兼容模式面向仅允许受信任人员使用的研究工具。API Key 控制服务入口；用户 ID 仍用于选择研究数据，并不构成用户身份认证或租户隔离。

## 已实现的边界

| 能力 | 实际行为 |
| --- | --- |
| 配置校验 | 缺少可用访问密钥或配置不合法时拒绝创建应用；不会退回匿名访问。第二阶段增加可选私有密钥配置文件 |
| 统一认证 | 所有业务请求在解析请求体、初始化业务依赖前验证 `X-API-Key`；覆盖 SSE、恢复执行、图片、笔记导出、trace 下载和可选 debug 路由 |
| 多密钥 | 逗号分隔，去重后保存摘要；固定长度摘要逐个使用 `secrets.compare_digest` 比较，支持过渡轮换 |
| CORS | 仅允许 `ALLOWED_ORIGINS` 中的明确 Origin；拒绝通配符、路径、凭据或空配置；预检由外层 CORS 处理 |
| 公开探针 | 精确路径 `GET /health` 和 `GET /api/auth/config` 公开，仅返回存活信息或认证模式 |
| 错误契约 | 缺少密钥返回 401 和 `WWW-Authenticate: ApiKey`；错误/重复密钥头返回 403，不回显提交的密钥 |
| 日志与响应 | 日志、异常堆栈、JSON 响应、SSE、笔记 Markdown 附件和请求/检索/研究 trace 使用密钥脱敏；附件覆盖标题、正文与关联来源；旧的关闭脱敏开关不再生效 |
| 脱敏计算成本 | 字段按状态单向扫描；SSE 未决词按分片追加，完整词只扫描一次；配置密钥的前缀缓存长度由配置决定，避免匿名长字段及重复敏感词导致二次增长 |
| 参数错误 | 422 仅返回字段位置及校验类型，不回显请求体或校验上下文 |
| 文档与缓存 | 不发布 `/docs`、`/redoc`、`/openapi.json`；API 响应使用 `Cache-Control: no-store` |
| 监听范围 | 直接启动后端默认只监听 `127.0.0.1`；Docker 中 Milvus、MinIO 及其管理端口只绑定本机 |

CORS 限制的是浏览器跨域行为，无法阻止 curl 或伪造 Origin 的客户端。真正的授权由 API Key 校验完成；来源域名匹配也不能绕过认证。

## 1. 配置后端

复制仓库根目录的 [环境变量模板](../.env.example) 为 `.env`，再填写实际值。应用会按“进程环境 > 根目录 .env > backend/.env”的优先级加载，启动目录不影响配置读取。

在自己的终端生成访问密钥：

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

将生成的值填入服务器的 `.env`：

```dotenv
AUTH_MODE=api_key
BACKEND_API_KEYS=
ALLOWED_ORIGINS=https://research.example.com
ALIYUN_API_KEY=
BACKEND_HOST=127.0.0.1
ENABLE_DEBUG_ROUTES=false
```

- `BACKEND_API_KEYS` 中每个值必须是 32 至 256 个无空白的可打印 ASCII 字符。使用上述随机生成命令，不使用固定示例值或弱口令。
- 访问密钥与 `ALIYUN_API_KEY` 必须分开。供应商密钥只保存在后端，不能交给浏览器。
- `ALLOWED_ORIGINS` 只填协议、主机和可选端口，不带结尾斜杠。例如开发环境可填 `http://localhost:5173,http://127.0.0.1:5173`。
- `.env`、`backend/.env`、`new_frontend/.env.local` 已被 Git 忽略。Linux 上将凭据文件权限设为 `600`。
- 配置变更后重启后端。密钥不会写入源码、模板或启动日志。

启动：

```powershell
python backend/main.py --load-mode lazy
```

容器确需监听其他地址时，显式设置 `BACKEND_HOST`，并通过容器端口映射和防火墙限制入口。Gunicorn 的监听地址以其启动参数为准；[systemd 模板](../deploy/arxiv-agent.service) 已绑定回环地址。

## 2. 配置前端与登录

实际前端目录是 `new_frontend`。复制 [前端配置模板](../new_frontend/.env.example) 为 `.env.local`：

```dotenv
VITE_API_BASE_URL=/api
VITE_ENABLE_DEBUG_ROUTES=false
```

默认由 Vite 或 Nginx 代理 `/api`。跨域部署可填写 `https://api.example.com/api`，同时将前端 Origin 加入后端允许列表。

**不要配置 `VITE_BACKEND_API_KEY` 或其他 `VITE_` 密钥变量。** Vite 变量会进入浏览器构建产物，任何访客都能读取；构建配置会拒绝这类凭据。

1. 打开前端登录页，输入管理员私下分配的访问密钥。
2. 前端通过受保护的 `/api/auth/check` 校验；该接口不调用模型或数据库。
3. 验证成功后在当前标签页的内存和 `sessionStorage` 中保存密钥，并统一附加到 Axios 和 fetch 请求头。
4. 图片先经认证请求读取为 Blob，再展示和预览；下载使用同样的方式。密钥不进入 URL、浏览器历史或 Referer。
5. 点击“退出”清除凭据并释放页面会话。收到当前密钥的认证失败响应后返回登录页；旧请求的迟到错误不会清除新密钥。

`sessionStorage` 仍可被同源脚本读取。本阶段不是 HttpOnly 会话系统，部署前端必须可信，不能运行不受信任的脚本。

## 3. 公网入口

使用 [Nginx 模板](../deploy/nginx.conf) 代理前后端，并为实际域名安装有效证书、启用 HTTP 到 HTTPS 的跳转。模板的 HTTP 入口用于初始证书配置，**发送访问密钥前必须启用 HTTPS**。

后端、Milvus 和 MinIO 的端口不直接对公网开放。模板不会由 Nginx 自动注入共享 API Key，否则匿名访客也会获得后端调用能力。

Nginx 访问日志使用不包含查询参数或认证头的格式。新 trace 会脱敏；部署前仍需自行处理旧版本已经生成的敏感日志/trace，并轮换曾经泄露的密钥。新代码无法撤回已泄露的凭据。

## 4. 验证

用 `curl.exe`（Windows）或 `curl`（Linux）检查：

```powershell
# 无密钥：401，code=missing_api_key
curl.exe -i http://127.0.0.1:8001/api/auth/check

# 错误密钥：403，code=invalid_api_key（值放在变量里，避免命令行字面量被密钥扫描误报）
$wrongKey = 'deliberately-invalid'
curl.exe -i -H "X-API-Key: $wrongKey" http://127.0.0.1:8001/api/auth/check

# 公开存活检查：200
curl.exe -i http://127.0.0.1:8001/health
```

有效密钥可通过前端登录验证，或在受信任终端中以请求头调用 `/api/auth/check`；成功响应为 `{"status":"authenticated"}`。不要把真实密钥粘贴到共享日志、截图或公开工单中。

自动化回归：

```powershell
python -m pytest backend/tests/api/test_security_stage1.py backend/tests/api/test_qa_router.py backend/tests/smoke/test_backend_startup.py -q
cd new_frontend
npm run test
npm run build
```

回归覆盖未认证入口、错误/重复/非 ASCII 凭据、轮换、无认证预检、跨域 500 响应、参数与异常泄露、日志及 trace 脱敏、BM25 匹配词保留、跨分片凭据与流式取消，以及前端普通请求、三条 SSE 链路、Blob 读取、退出和迟到认证错误。

部署后的浏览器验证还应确认：合法密钥可登录；论文与流式问答能加载；证据图片可预览；trace 可下载；清除密钥后不能继续请求业务接口；实际 HTTPS 域名与 CORS 配置一致。

## 5. 轮换与适用范围

轮换时先配置 `BACKEND_API_KEYS=旧密钥,新密钥` 并重启，将新密钥分发给受信任人员；切换完成后移除旧密钥并再次重启。已开始执行的请求不会因轮换被自动取消，后续请求会重新校验。

第一阶段提供共享入口访问控制；[第二阶段](SECURITY_STAGE2_PLAN.md)补充请求限流、每个密钥的日配额、禁用/过期、IP 控制和审计。持有有效共享密钥的人仍可以选择其他用户 ID，因此只适合受信任的小团队。[第三阶段](SECURITY_STAGE3_PLAN.md)已增加独立用户身份、资源所有权校验和角色权限；开启 JWT 模式后共享密钥不能绕过登录。

## 实现位置

- [入口与 CORS](../backend/main.py)
- [密钥配置、验证与中间件](../backend/auth/api_key_middleware.py)
- [共用脱敏逻辑](../backend/utils/secret_redaction.py)
- [前端认证与受保护传输](../new_frontend/src/api/auth.ts)
- [后端回归](../backend/tests/api/test_security_stage1.py)
- [前端回归](../new_frontend/tests/security.test.mjs)

文档更新：2026-09-11。配置与验证步骤以当前实现为准。
