# arXiv Paper RAG Framework

一个面向 arXiv 论文检索、论文问答、个性化推荐与本地数据管理的全栈 RAG 系统。

项目采用前后端分离架构：

- 前端：`new_frontend/`，基于 Vue 3、TypeScript、Vite、Pinia、Element Plus
- 后端：`backend/`，基于 FastAPI、Uvicorn
- 本地开发代理：`/api -> http://127.0.0.1:8001`

![项目界面](images/RAG-fontend.png)

---

## 1. 功能特性

- arXiv 论文检索
  - 原始关键词检索
  - 结构化检索
  - arXiv category 过滤
  - 本地 OAI 元数据检索
- 论文级 RAG 问答
  - 单篇论文问答
  - evidence chunk 检索
  - 解析后论文上下文召回
- 个性化推荐
  - like / dislike 反馈
  - 用户兴趣建模
  - 基于聚类的兴趣表示
- Agent 检索
  - 自然语言论文检索
  - tool-based search workflow
- 本地数据管理
  - SQLite 元数据存储
  - Milvus 向量存储
  - Docling 解析资产
  - 检索与生成 trace 文件

---

## 2. 技术栈

| 层级 | 技术选型 |
| --- | --- |
| 前端 | Vue 3, TypeScript, Vite, Pinia, Element Plus |
| 后端 | Python, FastAPI, Uvicorn |
| 向量数据库 | Milvus |
| 元数据存储 | SQLite |
| 检索链路 | Embedding, Rerank, chunk retrieval |
| 文档解析 | Docling, PyMuPDF, pandas |

---

## 3. 运行环境

### 3.1 基础环境

- Python 3.10+
- Node.js 18+
- npm 9+

### 3.2 Python 依赖文件

根据运行平台选择对应的 requirements 文件：

| 平台 | 依赖文件 |
| --- | --- |
| Windows | `requirements_win.txt` |
| Ubuntu / Linux | `requirements_ubun.txt` |
| macOS | `requirements_mac.txt` |

### 3.3 外部依赖

前后端基础联调不要求所有外部服务都可用。完整的检索、问答和推荐链路通常需要：

- Milvus 服务
- 可写的 SQLite 数据库路径
- Embedding provider
- Rerank provider
- LLM provider API Key
- arXiv / 模型服务网络代理

---

## 4. 快速启动

### 4.1 克隆仓库

```bash
git clone https://github.com/huangjia2019/rag-project01-framework.git
cd rag-project01-framework
```

### 4.2 安装后端依赖

Windows 示例：

```bash
pip install -r requirements_win.txt
```

Linux / macOS 请切换为对应 requirements 文件。

### 4.3 安装前端依赖

```bash
cd new_frontend
npm install
cd ..
```

### 4.4 启动后端

```bash
cd backend
python main.py
```

后端默认监听端口：`8001`。

### 4.5 启动前端

```bash
cd new_frontend
npm run dev
```

Vite dev server 会将 `/api` 请求代理到 `http://127.0.0.1:8001`。

### 4.6 基础验证

- 后端进程监听 `8001`
- 前端开发服务可访问
- `/api` 请求能够转发到 FastAPI 后端
- 搜索页面能够发起后端请求

---

## 5. 后端加载模式

后端支持两种服务加载模式：

- `lazy`：延迟初始化，启动速度更快，首次请求延迟更高
- `preload`：启动阶段预热服务，启动速度更慢，首次请求更稳定

当前 `backend/utils/config.py` 中的默认值为 `preload`。

命令行覆盖：

```bash
cd backend
python main.py --load-mode lazy
```

PowerShell：

```powershell
$env:BACKEND_SERVICE_LOAD_MODE = "lazy"
python main.py
```

CMD：

```cmd
set BACKEND_SERVICE_LOAD_MODE=lazy
python main.py
```

bash / zsh：

```bash
BACKEND_SERVICE_LOAD_MODE=lazy python main.py
```

---

## 6. 配置说明

运行时配置定义在：

```text
backend/utils/config.py
```

Python 配置示例：

- [`backend/utils/config.example.py`](backend/utils/config.example.py)

详细配置文档：

- [`docs/configuration.md`](docs/configuration.md)

核心环境变量：

**本文默认使用qwen系列api-key，且由于arixv国内访问异常，默认arxiv请求走代理，代理端口默认为7897**

| 环境变量 | 说明 | 默认值 / 示例 |
| --- | --- | --- |
| `BACKEND_SERVICE_LOAD_MODE` | 后端服务加载模式 | `preload` |
| `ARXIV_DATA_SOURCE` | arXiv 数据源 | `local` |
| `ARXIV_PROXY_URL` | arXiv 网络代理 | `http://127.0.0.1:7897` |
| `MILVUS_URI` | Milvus 服务地址 | `http://localhost:19530` |
| `SQLITE_DATABASE_PATH` | 推荐系统 SQLite 路径 | `06-database/recommendation.db` |
| `OAI_SQLITE_DATABASE_PATH` | arXiv OAI SQLite 路径 | `backend/06-database/arxiv_oai.db` |
| `EMBEDDING_PROVIDER` | Embedding 服务提供方 | `dashscope` |
| `EMBEDDING_MODEL` | Embedding 模型 | `qwen3-vl-embedding` |
| `RERANK_PROVIDER` | Rerank 服务提供方 | provider-specific |
| `OPENAI_API_KEY` | OpenAI API Key | empty |
| `DEEPSEEK_API_KEY` | DeepSeek API Key | empty |
| `QWEN_API_KEY` | Qwen API Key | empty |

配置约定：

- 不要提交真实 API Key。
- 优先通过环境变量覆盖运行时配置。
- 检索或问答异常时，优先检查 Milvus、Embedding、Rerank、SQLite 路径和本地数据。
- arXiv 请求异常时，优先检查代理配置。

---

## 7. 项目结构

```text
backend/
  main.py                 # FastAPI 应用入口
  routers/                # API 路由
  services/               # 业务服务层
  tools/                  # 工具层与工具注册
  agents/                 # Agent 工作流
  utils/                  # 配置与通用工具
  06-database/            # SQLite 数据库
  03-vector-store/        # 向量存储相关文件
  03-docling-assets/      # 文档解析资产
  04-search-results/      # 检索输出
  05-generation-results/  # 生成输出
  06-daily-arxiv-paper/   # arXiv 日更数据
  07-arxiv-tools/         # arXiv 同步脚本与工具

new_frontend/
  src/
    api/                  # API client 封装
    router/               # Vue Router 配置
    stores/               # Pinia 状态管理
    views/                # 页面视图
    components/           # 通用组件
```

常用开发入口：

- 后端入口：`backend/main.py`
- 后端配置：`backend/utils/config.py`
- 后端路由：`backend/routers/`
- 后端服务：`backend/services/`
- 前端页面：`new_frontend/src/views/`
- 前端 API client：`new_frontend/src/api/`
- 前端路由：`new_frontend/src/router/`

---

## 8. 前后端联调

本地开发请求链路：

```text
Browser -> Vite dev server -> /api proxy -> FastAPI backend -> routers -> services
```

代理配置：

- 文件：`new_frontend/vite.config.ts`
- target：`http://127.0.0.1:8001`

后端路由前缀：

- `/api`

---

## 9. 前端路由

- `/`：Dashboard
- `/search`：论文检索
- `/paper/:id`：论文详情与 RAG 问答
- `/recommendations`：推荐列表
- `/profile`：研究兴趣画像
- `/agent-search`：Agent 检索
- `/labeled`：已标注论文
- `/chunks`：chunk 查看器

`/agent-graph` 当前重定向到 `/`。

---

## 10. API 概览

后端路由统一挂载在 `/api` 前缀下。

| 前缀 | 说明 |
| --- | --- |
| `/api/arxiv/*` | arXiv 检索、分类、下载 |
| `/api/agent/*` | Agent 检索 |
| `/api/user/*` | 用户反馈、兴趣向量、推荐 |
| `/api/paper/*` | 论文管理与论文级问答 |
| `/api/chunks/*` | chunk 文件查看 |

路由文件：

- `backend/routers/arxiv_router.py`
- `backend/routers/agent_router.py`
- `backend/routers/user_router.py`
- `backend/routers/paper_router.py`
- `backend/routers/qa_router.py`
- `backend/routers/chunk_router.py`

---

## 11. 数据目录

| 路径 | 说明 |
| --- | --- |
| `backend/06-database/` | SQLite 数据库文件 |
| `backend/03-vector-store/` | 向量存储相关文件 |
| `backend/03-docling-assets/` | Docling 解析资产 |
| `backend/04-search-results/` | 检索输出 |
| `backend/05-generation-results/` | 生成输出 |
| `backend/06-daily-arxiv-paper/` | arXiv 日更数据 |
| `backend/07-arxiv-tools/` | arXiv 同步工具 |
| `temp/` | 调试 trace 与临时文件 |

部署或迁移时，应将数据库文件、向量存储文件和文档解析资产视为持久化数据，除非明确需要重建索引或重新解析。

---

## 12. 构建与部署

### 12.1 前端构建

```bash
cd new_frontend
npm run build
```

预览构建产物：

```bash
npm run preview
```

### 12.2 后端运行

```bash
cd backend
python main.py
```

### 12.3 部署关注点

- 前端静态资源托管
- 后端进程管理
- 反向代理、CORS、API 路由转发
- Milvus 服务可用性
- SQLite 与向量存储持久化
- 模型服务凭证管理
- 数据目录挂载与备份

计划补充独立部署文档：`docs/deployment.md`。

---

## 13. arXiv OAI 同步

同步脚本位于：

```text
backend/07-arxiv-tools/
```

常用入口：

- `backend/07-arxiv-tools/sync_arxiv_oai.py`
- `backend/07-arxiv-tools/sync_arxiv_oai_last_day.ps1`
- `backend/07-arxiv-tools/sync_arxiv_oai_last_day.cmd`
- `backend/07-arxiv-tools/sync_arxiv_oai_last_6_months.cmd`
- `backend/07-arxiv-tools/sync_arxiv_oai_since_last_run.cmd`

默认 OAI 数据库：

```text
backend/06-database/arxiv_oai.db
```

OAI 同步用于构建和更新本地 arXiv 元数据，支撑检索、推荐和向量化工作流。

---

## 14. 故障排查

### 14.1 前端请求失败

检查后端是否监听 `8001`，以及 `new_frontend/vite.config.ts` 是否仍将 `/api` 代理到 `http://127.0.0.1:8001`。

### 14.2 后端启动慢

`preload` 模式会在启动阶段执行服务预热。开发阶段可切换为 `lazy` 以缩短启动时间。

### 14.3 检索、问答或推荐结果不完整

检查：

- Milvus 连接
- Embedding provider 配置
- Rerank provider 配置
- SQLite / OAI 数据库路径
- 本地数据完整性
- 网络代理配置

---

## 15. 文档

已提供：

- [`docs/configuration.md`](docs/configuration.md)
- [`backend/utils/config.example.py`](backend/utils/config.example.py)

计划补充：

- `docs/api.md`
- `docs/deployment.md`
- `docs/faq.md`

---

## License

如需开源发布，建议补充明确的 License 文件与说明。
