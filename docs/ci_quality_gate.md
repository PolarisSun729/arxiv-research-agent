# CI 质量门禁

CI 的唯一主命令是：

```bash
python scripts/check_quality.py ci
```

GitHub Actions 配置位于 `.github/workflows/quality-gate.yml`。工作流也支持 `workflow_dispatch` 手动触发，便于在不推送新提交时复现同一门禁。

该命令覆盖：

1. `doctor-basic`：检查本地运行环境、依赖、配置加载、目录和 SQLite 临时写入；
2. `backend-static`：编译所有后端 Python 文件、导入关键入口模块，并在存在 ruff 时运行基础 lint；
3. `backend-tests`：运行后端现有测试集，不重复运行 smoke 目录；
4. `backend-startup-smoke`：lazy 模式创建 FastAPI app，进入 lifespan，验证 router 注册、基础 API 响应、统一错误结构和 lazy 启动不预热重型依赖；
5. `frontend-tests`：运行 `npm run test`；
6. `frontend-build`：运行 `npm run build`。

`backend-static` 和普通后端测试是两个独立阶段。static 负责提前发现语法、编译、关键 import 和低误伤 lint 问题；pytest 负责业务行为回归。static 阶段固定保持 lazy/offline，不连接真实 Milvus、arXiv、LLM、Embedding、rerank 或 PDF 解析服务。

GitHub Actions 配置位于 `.github/workflows/quality-gate.yml`，使用 `windows-latest`，因为仓库当前只有 `requirements_win.txt`。

本地开发者可以用同一条命令复现 CI 阶段：

```bash
python scripts/check_quality.py ci
```

脚本会逐阶段打印开始、命令、`PASS` / `FAIL`、耗时和失败摘要；任一阶段失败时整体返回非 0 退出码。

## CI 不检查什么

默认 CI 不做真实外部连接：

- 不连接真实 Milvus；
- 不调用 arXiv API / OAI；
- 不调用真实 Embedding / LLM / rerank；
- 不下载真实 PDF；
- 不构建真实大索引；
- 不要求真实 API key。

这些检查属于本地或手动运行的 full doctor：

```bash
python scripts/doctor.py full
python scripts/doctor.py full --check-paid
```

`--check-paid` 会真实调用可能计费的模型服务，只能在明确需要排查真实运行能力时使用。

## 常见失败

- `backend-static` 失败：优先看子项是 `compileall`、`import-smoke` 还是 `ruff-optional`，再复制对应复现命令排查。
- `backend-startup-smoke` 失败：通常是 router import、FastAPI app 创建、全局异常契约或 lazy 装配被改坏。
- `backend-tests` 失败：复制汇总里的复现命令单独跑。
- `frontend-tests` 失败：进入 `new_frontend` 运行 `npm run test`。
- `frontend-build` 失败：进入 `new_frontend` 运行 `npm run build`，通常是 TypeScript / Vue / Vite 编译问题。
- `doctor-basic` 失败：优先安装缺失依赖，确认 `backend/utils/config.py` 可加载，确认本地目录和 SQLite 临时写入可用。

## 分层边界

- 默认自动化测试：`python scripts/check_quality.py`，适合本地提交前验证。
- CI 主命令：`python scripts/check_quality.py ci`，适合新机器和 PR 检查。
- 本地环境体检：`python scripts/doctor.py basic`，只检查本地环境。
- 真实外部服务连接：`python scripts/doctor.py full`，用于排查 Milvus、arXiv、OAI 等连接。
- 付费模型连接：`python scripts/doctor.py full --check-paid`，用于最小化检查 Embedding / LLM / rerank。
- 手动端到端集成：真实下载、真实索引构建、真实问答和真实推荐，不进入默认 CI。

## Codex 验收要求

Codex 修改代码后的验收标准见 [`docs/codex_acceptance.md`](codex_acceptance.md)。最终回复必须说明运行了哪些检查、哪些通过、哪些失败、失败是否与本次修改相关；如果无法运行完整门禁，必须说明原因和已运行的替代检查。
