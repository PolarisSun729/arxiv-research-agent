# 维护与质量

## 变更完成标准

一次代码变更只有同时满足以下条件才算完成：

1. 业务契约、状态真源和降级行为保持一致，或已明确完成迁移。
2. 风险相称的自动化检查已经运行，并记录真实结果与环境阻塞。
3. 修改了能力边界、状态契约、配置语义、持久化语义或可观察性时，对应维护者文档已在同一提交更新。
4. 文档中的相对链接和代码路径通过自动校验。

完整映射见 [维护者文档入口](../README.md)。

## 文档校验

[`scripts/check_docs.py`](../../scripts/check_docs.py) 是无外部依赖的文档结构校验：

- 扫描 `docs/` 和仓库 `README.md` 中的相对 Markdown 链接。
- 验证链接目标位于仓库内且文件或目录实际存在。
- 跳过锚点、网络地址、邮件地址和数据 URI，不访问网络。
- 失败时输出文档、行号和失效目标，必须先修复而不是通过删除链接掩盖问题。

本地单独运行：

```powershell
python scripts/check_docs.py
python scripts/check_quality.py docs
```

## 统一质量门禁

[`scripts/check_quality.py`](../../scripts/check_quality.py) 是唯一的质量编排入口。常用目标如下：

| 目标 | 内容 | 适用场景 |
| --- | --- | --- |
| `docs` | Markdown 链接与路径校验 | 仅修改文档时的最小检查。 |
| `static` | 后端编译、关键导入和低误伤 lint | 改动 Python 模块边界。 |
| `backend` | 后端静态检查、测试和启动烟测 | 改动后端业务。 |
| `frontend` | 前端测试与生产构建 | 改动前端接口消费或展示。 |
| `smoke` | 编译、启动烟测和最小前端错误测试 | 快速本地确认。 |
| `deployment-tests` | 发布包、依赖锁、共享数据和失败回退的离线测试 | 修改部署脚本。 |
| 默认或 `ci` | 文档、部署测试、环境体检、后端、前端的组合检查 | 提交前或 CI。 |

`doctor-full` 会触及真实依赖，始终保持显式运行；默认门禁只执行离线安全检查。不要为了让质量门禁通过而把网络、模型、向量库或真实 arXiv 调用加回默认流程。

## 按能力选择验证

| 修改范围 | 至少验证 | 额外关注 |
| --- | --- | --- |
| 文档 | `python scripts/check_docs.py` | 所有链接均指向当前文件或目录。 |
| CI/CD 与发布脚本 | `python scripts/check_quality.py deployment-tests docs`、`bash -n deploy/publish_ssh.sh` | Linux 真实链接用例、同一 SHA 的构建与发布、持久化目录、失败回退和依赖缓存。 |
| Router、依赖装配、错误出口 | `python scripts/check_quality.py backend` | 路由注册、错误 payload 和 lazy 启动烟测。 |
| Agent、计划、确认、恢复 | 后端相关测试与 `backend` 目标 | 两层 checkpoint、确认一次性消费、SSE/projection。 |
| QA 索引与检索 | 后端相关测试与 `backend` 目标 | 索引版本、任务状态、rerank 降级和上下文预算。 |
| 画像与推荐 | 后端相关测试与 `backend` 目标 | manual/generated/effective 层、负反馈和多样性。 |
| 本地持久化或产物路径 | 路径单元测试、`static`、`docs` 与 `backend` 目标 | 数据库、QA 缓存、向量库资产、生成结果和每日 arXiv 下载目录必须固定相对 `backend`；错误根目录数据库配置必须失败，根目录产物残留只允许 Doctor 警告。 |
| 前后端响应契约 | `backend` 与 `frontend` 目标 | 错误结构、流事件和字段投影。 |
| 研究与评测 | 真实模型契约的离线贯通测试、指标测试与 `backend` 目标 | 执行失败分母、引用 verdict、修复前后快照、token 缺失和配置不可比条件；真实付费 golden 不进入默认门禁。 |

## 评审规则

- 不接受“文档之后再补”：只要改动影响 [README](../README.md) 定义的任何责任边界，就在当前提交同步维护。
- 不新增历史计划、一次性修复报告或逐 Router 的长篇阅读笔记到 `docs/`；需要追溯时使用提交历史和测试。
- 不以日期作为正确性证明。文档应以代码锚点、状态真源和自动链接校验保持可信。
- 新增调试输出、trace 或本地资产入口时，必须证明普通响应不会泄露内部路径、原始模型数据或敏感上下文。
- 验收回复必须说明实际执行的检查、通过/失败结果和未执行的原因；不要把未运行的命令描述为已通过。

## CI 约束

GitHub Actions 的质量工作流调用 `python scripts/check_quality.py ci`。因此文档校验与后端/前端质量检查使用同一入口，避免本地规则与 CI 规则漂移。变更质量门禁本身时，必须更新本页、[测试指南](testing.md) 和相关 workflow 的实际行为说明。

质量工作流使用 Debian 12 / Python 3.11 容器、Node.js 22 和 CPU Torch。PR 和单独手动运行检查质量；main 推送由 [release 工作流](../../.github/workflows/deploy.yml) 复用质量门和密钥扫描，开启 `package_release` 后在检查通过时生成离线包。仅 main 且仓库变量 `DEPLOY_ENABLED=true` 才进入 production 部署，发布不取消正在进行的运行。服务器前置条件及开关顺序见 [CI/CD 部署手册](cicd-deployment.md)。不能用本地静态检查或健康接口通过代替完整 Debian 依赖安装和真实业务验收。
