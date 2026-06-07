# 环境体检 doctor

doctor 用来检查“当前机器能不能真实运行项目”。它和一键质量门禁分开：默认测试继续使用 fake service、stub 和临时 SQLite，不依赖 arXiv、Milvus、LLM、Embedding 或 rerank。

## basic 模式

默认命令：

```bash
python scripts/doctor.py
```

等价于：

```bash
python scripts/doctor.py basic
```

basic 只检查本地环境和本地依赖，不真实调用付费 API，不依赖外部网络。覆盖项包括：

- Python 版本；
- 后端关键 Python 包；
- Node / npm 版本；
- `new_frontend/node_modules` 和 Vite 本地依赖；
- 本地数据目录是否存在或父目录是否可写；
- SQLite 配置目录是否能创建临时数据库；
- API key 环境变量是否显式存在；
- `backend/utils/config.py` 是否可加载；
- FastAPI app 是否能在 lazy 模式创建；
- PyMuPDF / Docling / pypdf / pdfplumber 是否可导入。

## full 模式

full 会在 basic 基础上检查真实连接能力：

```bash
python scripts/doctor.py full
```

默认 full 会检查：

- Milvus / 向量库 TCP 与客户端连接；
- arXiv API 最小查询；
- arXiv OAI `Identify` 请求。

Embedding、LLM、rerank 属于可能计费的模型服务，full 默认显示 `SKIP`。如果明确允许真实调用最小模型请求，再运行：

```bash
python scripts/doctor.py full --check-paid
```

该命令只做最小健康检查，不下载大量论文，不构建真实索引，不写入正式数据库，也不会打印任何 API key。

## 统一入口

doctor 也可以通过质量入口显式调用：

```bash
python scripts/check_quality.py doctor
python scripts/check_quality.py doctor-full
```

注意：`python scripts/check_quality.py` 默认会运行 basic doctor，但不会自动运行 full doctor；真实外部服务连接检查必须显式执行 `doctor-full` 或 `python scripts/doctor.py full`。

## 输出含义

每个检查项都会输出：

- 检查项名称；
- `PASS` / `WARN` / `FAIL` / `SKIP`；
- 失败或跳过原因；
- 建议修复方式；
- 是否属于必需项；
- 是否影响默认测试；
- 是否影响真实运行。

如果希望复制给 AI 或 CI 继续分析，可以使用 JSON 输出：

```bash
python scripts/doctor.py basic --json
python scripts/doctor.py full --json
```

## 常见结论

- 缺少关键 Python 包：先安装平台对应 requirements，例如 `pip install -r requirements_win.txt`。
- 缺少前端依赖：进入 `new_frontend` 执行 `npm install`。
- Milvus 连接失败：启动 Milvus，或检查 `MILVUS_URI`。
- API key 未显式配置：设置 `QWEN_API_KEY`、`EMBEDDING_DASHSCOPE_API_KEY`、`RERANK_DASHSCOPE_API_KEY` 等环境变量。doctor 只提示变量状态，不打印密钥值。
- arXiv / OAI 连接失败：检查网络、代理或 `ARXIV_PROXY_URL` / `ARXIV_OAI_ENDPOINT`。
