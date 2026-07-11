# 一键质量门禁

本项目提供一个统一的离线安全检查入口，用来在新机器、新环境或修改代码后快速验证基础质量：

```bash
python scripts/check_quality.py
```

默认入口会按固定顺序执行：

1. 环境体检 basic：`python scripts/doctor.py basic`
2. 后端静态检查：`python scripts/backend_static_check.py`
3. 后端自动化测试：`python -m pytest backend/tests --ignore=backend/tests/smoke`
4. 后端启动烟测：`python -m pytest backend/tests/smoke`
5. 前端自动化测试：`cd new_frontend && npm run test`
6. 前端构建检查：`cd new_frontend && npm run build`

脚本会在最后输出每个阶段的 `PASS` / `FAIL`、耗时、退出码、失败命令和失败摘要。任一阶段失败时，脚本返回非 0 退出码，后续可以直接接入 CI。

## 默认检查边界

默认质量门禁只编排已有的离线安全检查，不会主动执行 arXiv OAI 同步、真实 PDF 下载、真实 LLM 调用、Embedding、rerank 或 Milvus 写入流程。

basic doctor 是默认入口的第一阶段，只检查 Python / Node / 依赖安装 / 配置加载 / 本地目录 / SQLite 临时写入 / FastAPI lazy 创建等本地运行条件。它不会访问外部网络，也不会要求 Milvus、arXiv、LLM、Embedding 或 rerank 服务可用。

后端静态检查是独立阶段，不等同于普通单元测试。它会先扫描已删除的 legacy 入口，再执行 `compileall` 覆盖 `backend/` 下所有 Python 文件，并导入应用入口、依赖装配、router、service、Agent、tool、config 等关键模块，用来提前发现语法错误、基础编译错误、模块重命名遗漏和导出契约断裂。导入检查固定使用 lazy/offline 模式，不会实例化真实 LLM、Embedding、Milvus、arXiv 或 PDF 解析链路。Agent 确认恢复状态以 `pending_confirmation` / `runtime_state` / `resume` 为准，不再新增独立 legacy 映射入口；向量存储服务统一走 `dependencies.get_vector_store_service()` 或 `services.storage.vector_store_service.VectorStoreService`，不再恢复 archive 旧实现。

如果当前环境安装了 `ruff`，静态检查会额外运行：

```bash
ruff check backend --select E9,F821,F822,F823
```

这组规则只覆盖语法错误和未定义名称等低误伤问题，不把历史风格问题纳入第一版门禁。未安装 `ruff` 时会明确输出跳过信息，默认检查不会失败。未来可以在这个入口继续扩展 mypy / pyright，但不会在第一版阻塞已有代码。

后端测试入口使用 `pytest`，原因是当前 `backend/tests` 中同时存在 `unittest.TestCase` 测试和少量 pytest 参数化测试。`pytest.ini` 已将 `testpaths` 指向 `backend/tests`，因此也可以直接运行：

```bash
python -m pytest
```

如果只想验证传统 unittest 风格测试，仍可使用：

```bash
python -m unittest discover -s backend/tests -p "test*.py"
```

但这个命令不会完整覆盖 pytest 参数化测试，所以推荐的一键入口仍然是 `python scripts/check_quality.py`。

启动烟测位于 `backend/tests/smoke`。它只验证 FastAPI lazy app 创建、lifespan 可进入、核心 router 注册、基础 API 响应、统一错误结构和 lazy 启动不会预热重型依赖，不连接真实 Milvus、LLM、Embedding、rerank、arXiv 或 PDF 解析后端。

smoke test 和 full integration test 的边界如下：

- smoke test 只证明应用入口、router 装配、依赖注入表面和错误契约没有被改坏，使用 fake service 和 `TestClient`，不跑完整业务链路。
- full integration test 才验证 RAG、Recommendation、Paper QA、Memory 等业务流程，仍应使用 fake 或临时 SQLite，不进入默认真实外部服务连接。
- 真实 Milvus、真实 arXiv、真实模型调用、真实 PDF 下载解析属于手动外部集成检查，不进入默认 smoke。

## Paper QA 边界审查规则

Paper QA 相关改动除运行测试外，还需要在 code review 中检查服务边界：

1. `PaperQAService` 只应新增主流程编排入口，不应为了测试方便新增 `SessionService`、`ContextPackBuilder`、`AnswerGenerator` 或 `qa_utils` 的透传 wrapper。
2. 组件能力测试应直接放到 `backend/tests/unit/services/paper_qa/`，集成测试只覆盖 QA 主流程、副作用和输出结构。
3. 新增或修改 LLM 调用、规则兜底、参数校验、状态流转、debug/trace 记录时，应同步更新中文注释、边界文档和对应断言。
4. 若确实需要保留 router 兼容入口，需要在 `docs/paper_qa_service_wrapper_boundary.md` 说明调用方、迁移条件和删除时机。

## 分阶段命令

查看所有目标和阶段：

```bash
python scripts/check_quality.py --list
```

只跑后端：

```bash
python scripts/check_quality.py backend
```

只跑前端：

```bash
python scripts/check_quality.py frontend
```

只跑后端静态检查：

```bash
python scripts/check_quality.py static
```

只跑纯 Python 编译检查：

```bash
python scripts/check_quality.py compile
```

`static` 是推荐的后端静态检查入口，`compile` 仅保留为最小编译检查入口。

只跑轻量 smoke 检查：

```bash
python scripts/check_quality.py smoke
```

只跑后端启动烟测：

```bash
python scripts/check_quality.py backend-startup-smoke
```

CI 主命令：

```bash
python scripts/check_quality.py ci
```

`ci` 与默认 `all` 一样包含 `doctor-basic`，适合作为新机器或 GitHub Actions 的唯一质量门禁命令。

显式运行环境体检 basic：

```bash
python scripts/check_quality.py doctor
```

显式运行环境体检 full：

```bash
python scripts/doctor.py full
```

只跑某个具体阶段：

```bash
python scripts/check_quality.py backend-startup-smoke
python scripts/check_quality.py backend-tests
python scripts/check_quality.py backend-static
python scripts/check_quality.py backend-compile
python scripts/check_quality.py frontend-tests
python scripts/check_quality.py frontend-build
```

环境体检分为默认 basic 和显式 full 两层。`python scripts/check_quality.py` 会运行 basic doctor，但不会自动运行 full doctor，也不会因为 Milvus、arXiv、LLM、Embedding 或 rerank 服务不可用而失败。full doctor 请直接运行 `python scripts/doctor.py full`，避免质量门和 CI 误触发真实外部连接。doctor 的完整说明见 [`docs/doctor.md`](doctor.md)，CI 说明见 [`docs/ci_quality_gate.md`](ci_quality_gate.md)。

Codex 修改代码后的统一验收口径见 [`docs/codex_acceptance.md`](codex_acceptance.md)。最终回复需要列出实际运行的检查、通过/失败结果，以及未运行完整门禁时的原因。

默认会尽量执行完所有选中阶段再汇总。如果希望首个失败后立即停止：

```bash
python scripts/check_quality.py --fail-fast
```

## 失败排查

脚本失败时会打印类似信息：

```text
FAIL backend-tests             12.3s  exit=1
     复现: python -m pytest backend/tests
     摘要: ModuleNotFoundError: No module named '...'
```

排查顺序建议：

1. 先复制 `复现` 命令单独运行，确认失败是否稳定。
2. 如果失败在 `backend-static`，优先按失败子项修复语法、import 或 ruff 报错。
3. 如果失败在 `backend-compile`，优先修复 Python 语法或基础编译错误。
4. 如果失败在 `backend-tests`，先确认 Python 依赖是否安装完整，再看测试失败栈。
5. 如果失败在 `frontend-tests`，进入 `new_frontend` 单独运行 `npm run test`。
6. 如果失败在 `frontend-build`，进入 `new_frontend` 单独运行 `npm run build`，通常需要处理 TypeScript 或 Vite 构建错误。

前端依赖不会在质量门禁中自动安装。新环境请先执行：

```bash
cd new_frontend
npm install
```
