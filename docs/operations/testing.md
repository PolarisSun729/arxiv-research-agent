# 后端测试指南

## 1. 环境

- Python 环境：`conda activate new_rag`
- 仓库级质量命令必须在仓库根目录执行。
- 仅后端的测试命令必须在 `backend/` 下执行。
- 测试入口：`pytest`，同时收集 pytest 函数和 `unittest.TestCase` 用例。

本仓库的绝大多数自动化测试都设计为离线运行，依赖假服务、桩或临时 SQLite 文件。

## 2. 核心测试规则

### 2.1 不得连接真实外部服务的测试

以下自动化测试不得调用真实外部服务：

- `tests/unit/` 下的单元测试
- `tests/api/` 下的 API 测试
- `tests/integration/` 下的集成测试
- `tests/golden/` 下的 golden 烟测
- `tests/unit/agents/` 下的 Agent 相关测试

自动化测试运行时，不得连接：

- 真实 arXiv API
- 真实 arXiv OAI 网络端点
- 真实 LLM provider
- 真实 embedding provider
- 真实 rerank provider
- 真实 Milvus / 向量数据库服务
- 真实 PDF 下载 / Docling / PyMuPDF 解析后端

这些测试应使用 `tests/helpers/` 中已有的假服务、mock 对象、动态导入桩或内存假服务。

### 2.2 允许使用临时 SQLite 的测试

以下测试只允许使用临时 SQLite：

- 数据库服务测试
- 记忆 / 推荐集成测试
- 论文 QA 集成测试
- QA 索引任务流程测试
- 任何显式使用 `tests/helpers/sqlite.py` 的 Router 或服务测试

不要让自动化测试指向真实项目数据库，例如持久化的推荐库或 OAI 库文件。

### 2.3 仅限手动执行的外部集成

仅限手动执行的外部集成检查不属于自动化测试，不得进入默认 CI，包括：

- 真实 arXiv 搜索与下载验证
- 真实模型 provider 调用
- 真实 embedding / rerank provider 验证
- 真实 Milvus 集成
- 依赖外部或重型运行时的端到端 PDF 解析
- 类生产环境烟测

## 3. 如何运行测试

推荐的仓库级质量门禁：

```bash
python scripts/check_quality.py
```

默认门禁依次运行文档校验、部署测试、基础环境体检、后端静态检查、后端自动化测试、后端启动烟测、前端测试和前端构建检查，最后输出汇总。它把离线安全的前后端检查集中在一处，是提交代码前的首选命令。

CI 通过 `python scripts/check_quality.py ci` 使用同一套分阶段门禁。提交变更时应说明实际运行了哪些检查、通过/失败结果以及环境导致的阻塞，见 [维护与质量](maintenance-quality.md)。

后端静态检查不是普通单元测试。它在较重的测试套件之前运行 `compileall`、对关键的 app/router/service/agent/tool/config 模块做导入烟测，并可选执行低误伤的 `ruff` 规则。

后端启动烟测也独立于完整集成测试。它以 lazy 模式创建 FastAPI 应用，通过 `TestClient` 进入 lifespan，断言核心路由和稳定的错误 payload，并使用假服务，因此不会连接真实的 LLM、Embedding、Milvus、arXiv、rerank 或 PDF 解析后端。

在仓库根目录可用的分阶段入口：

```bash
python scripts/check_quality.py backend
python scripts/check_quality.py frontend
python scripts/check_quality.py smoke
python scripts/check_quality.py static
python scripts/check_quality.py compile
python scripts/check_quality.py backend-startup-smoke
```

`static` 是推荐的后端静态检查层；`compile` 只运行范围更窄的 Python 编译检查。

下面仅后端的命令默认已执行：

```bash
conda activate new_rag
cd backend
```

### 3.1 查看现有测试收集结果

```bash
python -m pytest tests --collect-only -q
```

### 3.2 运行单元测试

```bash
python -m pytest tests/unit
```

### 3.3 运行 API 测试

```bash
python -m pytest tests/api
```

### 3.4 运行集成测试

```bash
python -m pytest tests/integration
```

### 3.5 运行 golden 烟测

```bash
python -m pytest tests/golden/test_rag_golden_smoke.py
```

### 3.6 运行 RAG golden 流水线测试

```bash
python -m pytest tests/golden/test_rag_golden_pipeline.py
```

该套件使用假 embedding、内存向量库、固定 chunk 和临时 trace 输出，驱动真实的 `EnhancedRetrievalService` 编排。它验证证据 hit@k、关键检索调试阶段、trace 导出以及图表证据流转，不调用真实模型或 Milvus。

### 3.7 运行完整自动化测试套件

```bash
python -m pytest tests --ignore=tests/smoke
python -m pytest tests/smoke
```

这包括遵循 `test_*.py` 命名规则的单元、API、集成、辅助/基础设施和 golden 烟测。

单独使用 `unittest discover` 会漏掉 pytest 函数、参数化和 fixture，包括研究/评测回归套件，不能代替完整门禁。

### 3.8 研究问答与评测契约

```powershell
python -m pytest tests/unit/services/evaluation tests/unit/services/paper_evidence_research tests/integration/test_research_qa_contract.py tests/unit/services/test_llm_call_metrics.py
python -m services.evaluation.golden_runner --cases tests/golden/data/smoke_golden_set.jsonl --validate-only --allow-unlabeled
```

这些测试使用真实的研究请求/结果模型、图执行和临时 SQLite 会话，只替换 provider 和索引 I/O。覆盖范围包括同步与 SSE 等价、安全错误、持久化记录重评分、已验证引用、检索失败、修复增益、运行分母和 provider 调用计数。Agent 图的桩必须在其自身导入后恢复，确保这些测试仍运行在真实 LangGraph 上。

当前 12 条烟测用例尚缺人工标注的可回答性、证据和参考答案。不带 `--allow-unlabeled` 的 `--validate-only` 会按预期以非零码退出；两条校验命令都不调用模型。不带 `--validate-only` 的 golden runner 会使用真实研究引擎，不属于自动化测试。标注、指标和基线要求见 [生成效果评测](../capabilities/evaluation.md)。

### 3.9 三阶段安全与部署初始化

使用 [隔离入口](../../scripts/test_security.py)，它为认证/业务 SQLite、审计、trace、缓存和合成凭据建立独立临时目录，阻止读取开发/生产 dotenv 和回退打开默认账号库。测试生成的临时 dotenv 仍允许读取，以覆盖用户管理 CLI。默认包含三阶段 API 测试与安全初始化测试；`--full` 包含全部后端离线回归。

```powershell
conda activate new_rag
python scripts/test_security.py
python scripts/test_security.py -k "redact or notes_markdown_export"
python scripts/test_security.py --full
```

Linux/Git Bash 下命令相同，用哪个 Python 解释器运行就使用哪个环境。脱敏性能和匿名长字段回归使用有界子进程，避免错误正则挂死测试；SSE 回归验证跨分片凭据及普通文本完整性。默认回归使用模拟 Redis；实际容器持久化、TLS 和付费模型验收需要另行执行，不能用离线通过代替。

## 4. 常用定向命令

### 发布与回退脚本

在仓库根目录执行：

```bash
python scripts/check_quality.py deployment-tests docs
bash -n deploy/publish_ssh.sh
```

[`tests/deploy/test_release.py`](../../tests/deploy/test_release.py) 只使用标准库、临时文件和模拟 pip/systemd/HTTP，覆盖归档穿越、凭据及数据隔离、wheel 哈希与 CPU 约束、版本切换、缓存和失败回退。跨平台状态测试替换链接边界；真实符号链接测试在不具备相应权限的 Windows 上明确 skip，在 Debian CI 必须运行。该目录没有 pytest 专属用例，因此仅这个阶段使用 `unittest discover`，后端仍使用 pytest。

质量工作流以 Debian 12 / Python 3.11 为构建目标。离线用例不证明实际 wheel 安装、systemd/Nginx/Redis 配置、模型访问或 2GB 容量已经通过；首次服务器验收见 [CI/CD 部署手册](cicd-deployment.md)。

容器工作流在 Checkout 后显式信任当前工作区，并提前验证 Git 可读取 HEAD。修改这一步时，应在隔离 Git 配置和临时仓库中复现属主不匹配，验证后续进程能读取提交、归档源码，同时确认其他仓库仍被拒绝；普通发布单元测试不能代替这项运行环境检查。

### Agent 测试

```bash
python -m pytest tests/unit/agents/arxiv_search_agent tests/api/test_agent_router.py
```

### 检索 / RAG 测试

```bash
python -m pytest tests/unit/services/retrieval tests/integration/test_enhanced_retrieval_service.py
python -m pytest tests/golden/test_rag_golden_pipeline.py
```

### 论文 QA 测试

```bash
python -m pytest tests/integration/test_paper_qa_service.py tests/integration/test_qa_index_build_flow.py tests/integration/test_research_qa_contract.py
```

### 记忆 / 推荐 / 意图测试

```bash
python -m pytest tests/unit/services/intent tests/integration/test_memory_service.py tests/integration/test_recommendation_flow.py
```

记忆集成套件同时覆盖 Agent 会话记忆的读写闭环：Agent 最终状态被归约为一个很小的持久化记忆 patch，随后重新加载并与前端上下文合并。它能发现 Agent 状态、已选论文上下文或工具调用摘要无法跨轮保留的回归。

### 数据库服务测试

```bash
python -m pytest tests/unit -k "sqlite or database"
```

## 5. 覆盖率命令

在自动化测试套件上统计覆盖率：

```bash
python -m coverage run -m pytest tests --ignore=tests/smoke
python -m coverage report -m
```

可选的 HTML 输出：

```bash
python -m coverage html
```

然后在本地打开 `htmlcov/index.html`。

## 6. 覆盖率验收说明

- 覆盖率用于观察回归，不是连接外部服务的许可。
- 覆盖率运行必须保持离线安全。
- 如果某个测试需要真实上游服务，应移到单独的手动脚本，而不是加入默认 pytest 收集。
- 优先编写聚焦的、基于假服务的测试，而不是宽泛但不稳定的集成覆盖。

## 7. 现有辅助工具位置

常用的可复用测试辅助工具位于：

- `tests/helpers/sqlite.py`
- `tests/helpers/fake_embedding_service.py`
- `tests/helpers/fake_generation_service.py`
- `tests/helpers/fake_vector_store_service.py`
- `tests/helpers/fake_arxiv_service.py`
- `tests/helpers/retrieval.py`
- `tests/helpers/agent_runtime.py`

引入新的重型 mock 之前，先使用这些辅助工具。

## 8. 问题排查

### 导入或依赖错误

- 确认环境为 `conda activate new_rag`
- 在 `backend/` 下执行命令
- 优先使用已有的、会为可选运行时依赖打桩的测试辅助工具

### 意外的真实网络或模型调用

- 检查测试是否忘记覆盖依赖
- 确认假服务注入或 `sys.modules` 桩已生效
- 不要接受只因真实外部服务恰好可达才通过的修复

### 担心 SQLite 污染

- 使用 `tests/helpers/sqlite.py` 中的临时 SQLite 辅助工具
- 不要在自动化测试中复用仓库里的真实数据库文件

### Windows 临时目录权限

当 pytest 或 Ruff 无法写入陈旧缓存时，使用一个新的仓库内临时目录。例如在仓库根目录执行：

```powershell
$testTemp = Join-Path (Get-Location).Path 'temp/quality-local'
New-Item -ItemType Directory -Path $testTemp -Force | Out-Null
$env:TEMP = $testTemp
$env:TMP = $testTemp
$env:RUFF_CACHE_DIR = Join-Path $testTemp 'ruff'
$env:PYTHONPYCACHEPREFIX = Join-Path $testTemp 'pycache'
$env:PYTHONUTF8 = '1'
$env:AGENT_RUNTIME_CHECKPOINT_BACKEND = 'memory'
$env:PYTEST_ADDOPTS = '-p no:cacheprovider --basetemp=temp/quality-local/pytest'
python scripts/check_quality.py backend
```
