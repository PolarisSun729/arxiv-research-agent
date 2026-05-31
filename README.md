# arXiv Paper RAG Framework

一个面向 arXiv 论文场景的全栈 RAG 项目，支持论文检索、论文详情问答、推荐系统、Agent 搜索和本地数据管理。

项目当前的前端位于 `new_frontend/`，后端位于 `backend/`。前端通过 `/api` 代理到本地后端，默认后端地址为 `http://127.0.0.1:8001`。

![项目界面](images/RAG-fontend.png)

## 项目功能

- arXiv 检索：支持 raw 查询和结构化查询
- 论文详情问答：针对单篇论文做 RAG 问答，并返回检索证据
- 推荐系统：基于用户喜欢/不喜欢的论文生成推荐结果
- Agent 搜索：提供自然语言 arXiv 搜索入口
- 论文管理：支持论文入库、删除、分类浏览和 chunk 查看
- 本地数据管理：支持本地 OAI 数据库、向量库和检索 trace 输出

## 技术栈

- 后端：Python、FastAPI、Uvicorn
- 前端：Vue 3、TypeScript、Vite、Pinia、Element Plus
- 检索与向量库：Milvus、SQLite、Rerank、Embedding
- 文档处理：Docling、PyMuPDF、pandas

## 目录结构

```text
backend/
  main.py                 # FastAPI 入口
  routers/                # API 路由层
  services/               # 核心业务服务
  tools/                  # 工具层与工具注册
  agents/                 # arXiv 搜索 Agent
  utils/                  # 配置与公共工具
  06-database/            # 本地 SQLite 数据库
  03-docling-assets/      # Docling 处理后的资源
  03-vector-store/        # 向量存储相关文件
  04-search-results/      # 检索结果
  05-generation-results/  # 生成结果
  06-daily-arxiv-paper/   # 每日 arXiv 数据

new_frontend/
  src/
    views/                # 页面
    components/           # 通用组件
    api/                  # 前端请求封装
    stores/               # Pinia 状态
    router/               # 路由
```

## 核心页面

- `/`：Dashboard
- `/search`：论文搜索
- `/paper/:id`：论文详情与 RAG 问答
- `/recommendations`：推荐列表
- `/agent-search`：Agent 搜索
- `/labeled`：已标注论文
- `/chunks`：chunk 查看

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/huangjia2019/rag-project01-framework.git
cd rag-project01-framework
```

### 2. 启动后端

后端使用 Python 3.10+。依赖文件按平台区分：

- Windows：`requirements_win.txt`
- Ubuntu / Linux：`requirements_ubun.txt`
- macOS：`requirements_mac.txt`

安装依赖：

```bash
pip install -r requirements_win.txt
```

启动后端：

```bash
cd backend
python main.py
```

也可以直接使用 Uvicorn：

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8001
```

如果想在启动时预加载服务，可以指定：

```bash
python main.py --load-mode preload
```

### 3. 启动前端

```bash
cd new_frontend
npm install
npm run dev
```

前端默认代理到 `http://127.0.0.1:8001`，因此后端先启动更稳妥。

### 4. 打包前端

```bash
cd new_frontend
npm run build
```

## 环境变量

项目支持通过环境变量调整数据源、模型和检索行为。常用项如下：

- `ARXIV_DATA_SOURCE`：arXiv 数据源，默认 `local`
- `ARXIV_PROXY_URL`：arXiv 请求代理，默认 `http://127.0.0.1:7897`
- `BACKEND_SERVICE_LOAD_MODE`：后端加载模式，默认 `lazy`
- `MILVUS_URI`：Milvus 地址，默认 `http://localhost:19530`
- `SQLITE_DATABASE_PATH`：推荐系统 SQLite 路径
- `OAI_SQLITE_DATABASE_PATH`：arXiv OAI SQLite 路径
- `EMBEDDING_PROVIDER`：Embedding 提供方
- `RERANK_PROVIDER`：Rerank 提供方
- `OPENAI_API_KEY`、`DEEPSEEK_API_KEY`、`QWEN_API_KEY`：生成模型相关密钥
- `RETRIEVAL_DEBUG`：是否输出检索调试信息

后端配置主要集中在 `backend/utils/config.py`，需要更改默认行为时优先查看这里。

## 数据与产物目录

以下目录会在运行过程中存放数据或中间结果：

- `backend/06-database/`：SQLite 数据库文件
- `backend/03-vector-store/`：向量库相关文件
- `backend/04-search-results/`：搜索结果
- `backend/05-generation-results/`：生成结果
- `backend/06-daily-arxiv-paper/`：按天归档的数据
- `backend/07-arxiv-tools/`：arXiv 同步与工具脚本
- `temp/`：调试 trace 和临时文件

## 常见接口

后端主要接口前缀为 `/api`，包括：

- `/api/arxiv/*`：arXiv 检索、分类、下载
- `/api/agent/arxiv-search`：Agent arXiv 搜索
- `/api/user/*`：喜欢、不喜欢、兴趣向量和推荐
- `/api/paper/*`：论文管理与问答
- `/api/chunks/*`：chunk 文件查看

## arXiv OAI 同步

仓库根目录提供了同步脚本：

- `sync_arxiv_oai.py`
- `sync_arxiv_oai_last_day.ps1`
- `sync_arxiv_oai_last_day.cmd`
- `sync_arxiv_oai_last_6_months.cmd`
- `sync_arxiv_oai_since_last_run.cmd`

默认的 OAI 数据库位于 `backend/06-database/arxiv_oai.db`。

## 说明

- 前端目录是 `new_frontend/`，不是旧的 `frontend/`
- 默认使用本地数据源和本地数据库文件
- 如果要使用在线模型、Milvus 或代理，请先按你的实际环境调整 `backend/utils/config.py` 对应的环境变量

如果你愿意，我可以继续把 README 再整理成更适合 GitHub 展示的版本，比如补上「功能截图」「部署步骤」「FAQ」和更完整的接口说明。
