# 系统整体架构总览

本文档基于以下入口文件做只读梳理：

- 项目根目录
- `README.md`
- `backend/main.py`
- `new_frontend/src`

目标是给维护者一个“先看哪里、每层负责什么、层和层之间如何连接”的系统级视图。本文只陈述当前仓库代码中能够直接确认的事实；无法从现有入口和关键实现中直接确认的内容，统一标记为 `TODO`。

## 1. 系统概览

这是一个围绕 arXiv 论文的全栈 RAG 系统，前后端分离，核心能力包括：

- arXiv 论文检索
- 论文入库与元数据管理
- 单篇论文 QA 与流式问答
- 基于用户偏好/画像的推荐
- 面向自然语言多意图的 Agent 检索入口
- 本地 SQLite + Milvus 的持久化与检索支撑
- Docling / PyMuPDF 驱动的论文解析与索引构建

从运行路径看，前端统一通过 `/api` 访问 FastAPI，后端再按路由分发到 service、agent、tool、storage 和模型能力层。

## 2. 整体架构 ASCII 图

```text
+------------------------------ Frontend: Vue 3 / Vite ------------------------------+
| views/ components/ composables/ stores/ api/ router/ types                         |
|                                                                                    |
|  Dashboard   Search   PaperDetail   Recommendations   Profile   AgentSearch        |
|      |          |           |               |              |            |           |
|      +----------+-----------+---------------+--------------+------------+-----------+
|                                         api/request.ts  (baseURL=/api)             |
+-----------------------------------------------|------------------------------------+
                                                |
                                                v
+------------------------------ Backend Entry: FastAPI -------------------------------+
| backend/main.py                                                                      |
|  include_router(..., prefix="/api")                                                  |
+------------------+------------------+------------------+----------------------------+
                   |                  |                  |
                   v                  v                  v
        +----------------+  +----------------+  +----------------+
        | arxiv_router   |  | paper_router   |  | user_router    |
        +----------------+  +----------------+  +----------------+
                   |                  |                  |
                   +---------+--------+---------+--------+
                             |                  |
                             v                  v
                    +----------------+  +----------------+
                    | qa_router      |  | agent_router   |
                    +----------------+  +----------------+
                             |                  |
                             |                  +--> agents/arxiv_search_agent/*
                             |                       |
                             |                       +--> planner / executor / graph
                             |                       +--> agent tool registry
                             |                       +--> backend tools invocation
                             |
                             +---------------------------------------------+
                                                                           |
                                                                           v
+-------------------------------------- Services -------------------------------------+
| arxiv/ document/ embedding/ intent/ llm/ memory/ paper_qa/ recommendation/         |
| retrieval/ storage                                                                   |
|                                                                                     |
| dependencies.py 负责构造并缓存单例 service：                                         |
| - DatabaseService / ArxivOaiDatabaseService                                         |
| - EmbeddingService / VectorStoreService / GenerationService                         |
| - MemoryService / EnhancedRetrievalService                                          |
| - RecommendationService / PaperQAService / PaperQAIndexBuilder / IndexJobManager    |
+------------------------------+-----------------------------+------------------------+
                               |                             |
                               v                             v
                    +----------------------+      +---------------------------+
                    | Storage / DB         |      | Model Providers           |
                    |                      |      |                           |
                    | SQLite               |      | Embedding provider        |
                    | - recommendation.db  |      | - dashscope / openai /   |
                    | - arxiv_oai.db       |      |   TODO: 其它运行时提供商 |
                    |                      |      |                           |
                    | Milvus               |      | LLM provider             |
                    | - abstract vectors   |      | - qwen / openai /        |
                    | - paper QA chunks    |      |   deepseek / local HF    |
                    +----------------------+      |                           |
                                                  | Rerank / compress         |
                                                  | - Qwen 压缩               |
                                                  | - CrossEncoder / local    |
                                                  |   TODO: 实际部署默认值    |
                                                  +---------------------------+
```

## 3. 分层与职责边界

### 3.1 Frontend

代码入口：

- `new_frontend/src/main.ts`
- `new_frontend/src/App.vue`
- `new_frontend/src/router/index.ts`
- `new_frontend/src/api/*`
- `new_frontend/src/views/*`
- `new_frontend/src/composables/*`
- `new_frontend/src/stores/paperStore.ts`

已确认职责：

- `router/index.ts` 负责页面级路由注册，当前核心页面包括：
  - `/` Dashboard
  - `/search`
  - `/paper/:id`
  - `/recommendations`
  - `/profile`
  - `/agent-search`
  - `/labeled`
  - `/chunks`（仅 `VITE_ENABLE_DEBUG_ROUTES=true` 时注册）
- `api/request.ts` 统一定义前端到后端的 HTTP 客户端，`baseURL=/api`
- `api/agent.ts` 封装 Agent 同步调用、SSE 流式消费、图结构拉取
- `api/papers.ts` 封装论文详情、推荐、QA、会话、笔记、诊断、索引构建等能力
- `views/*` 负责页面编排
- `composables/useAgentSearchChat.ts`、`usePaperRagChat.ts` 负责把页面状态和接口调用收口成可复用交互逻辑
- `stores/paperStore.ts` 承载论文、用户偏好、研究画像、笔记等跨页面状态
- `components/rag-chat/*`、`components/agent-search/*` 提供问答展示、证据面板、Agent 结果展示等 UI 组件

职责边界：

- 前端负责“页面交互、状态呈现、SSE 消费、请求参数组织、响应数据归一化”
- 前端不直接实现检索、向量召回、模型调用、索引构建
- 前端把业务语义映射成 `/api/...` 请求，但不承担真正的业务编排

### 3.2 Backend Routers

代码入口：

- `backend/main.py`
- `backend/routers/*.py`

已确认挂载顺序：

- `/api/arxiv/*` -> `arxiv_router`
- `/api/agent/*` -> `agent_router`
- `/api/user/*` -> `user_router`
- `/api/paper`、`/api/papers`、`/api/stats` 等 -> `paper_router`
- `/api/paper/{arxiv_id}/*` -> `qa_router`
- `/api/debug/chunks/*` -> `chunk_router`（仅 `ENABLE_DEBUG_ROUTES=true` 时注册，属于本地调试入口）

已确认职责：

- `main.py` 负责 FastAPI 应用创建、CORS、中间生命周期、路由挂载
- `dependencies.py` 负责 service 单例构造与 preload / lazy 策略
- `arxiv_router.py` 负责 arXiv 搜索、字段/分类读取、PDF 下载、搜索并保存
- `paper_router.py` 负责论文基础元数据、看板统计、同步状态、单篇论文入库/查询
- `qa_router.py` 负责论文 QA、QA 索引状态、诊断、trace 导出、会话、消息、笔记
- `user_router.py` 负责偏好、喜欢/不喜欢、通用行为埋点、研究画像、兴趣向量、推荐
- `agent_router.py` 负责 Agent chat、Agent chat SSE、Agent graph 导出
- `chunk_router.py` 负责调试用 chunk 文件列表和文件内容查看；默认不注册，避免正式 API 暴露本地解析产物

职责边界：

- router 层主要负责参数接收、依赖注入、错误码转换、响应结构统一
- router 层不应承载复杂业务流程
- 复杂业务下沉到 services 或 agents

### 3.3 Services

代码目录：

- `backend/services/arxiv`
- `backend/services/document`
- `backend/services/embedding`
- `backend/services/intent`
- `backend/services/llm`
- `backend/services/memory`
- `backend/services/paper_qa`
- `backend/services/recommendation`
- `backend/services/retrieval`
- `backend/services/storage`

已确认职责边界：

- `services/arxiv/*`
  - 负责 arXiv API 检索、本地 OAI 数据检索、查询拼装与 OAI 同步
  - `get_arxiv_service()` 根据 `ARXIV_DATA_SOURCE` 在 API / local 之间切换
- `services/document/*`
  - 负责 PDF 加载、Docling/PyMuPDF 解析、chunk 切分
  - 为后续 embedding 和 QA 索引提供结构化文档块
- `services/embedding/embedding_service.py`
  - 负责 embedding 文本构造与 embedding 调用
- `services/llm/generation_service.py`
  - 负责通用 LLM 调用、Qwen 模型路由、rerank 文本压缩、生成结果输出
- `services/retrieval/enhanced_retrieval_service.py`
  - 负责查询规划、路由召回、融合、重排、trace 导出、与 memory/intent 的联动
- `services/intent/intent_service.py`
  - 负责问题意图识别与检索意图特征抽取
- `services/paper_qa/*`
  - 负责单篇论文 QA 索引构建、索引作业状态管理、论文问答主流程
- `services/recommendation/*`
  - 负责候选召回、兴趣向量、用户偏好落库、推荐排序、推荐解释
- `services/memory/*`
  - 负责用户研究画像、短期记忆、会话记忆等能力
- `services/storage/database_service.py`
  - 负责 SQLite 业务数据读写
- `services/storage/vector_store_service.py`
  - 负责 Milvus collection 管理、向量写入、向量检索、QA chunk schema

职责边界总结：

- service 层是业务能力的主体
- service 之间允许组合，但组合入口集中在 `dependencies.py`
- storage service 只处理持久化，不直接决定页面交互
- llm / embedding / retrieval 是“能力层”，paper_qa / recommendation / arxiv 是“业务编排层”

### 3.4 Agents

代码目录：

- `backend/agents/arxiv_search_agent/*`

已确认职责：

- `__init__.py` 提供统一导出入口
- `service.py` 提供 `run_arxiv_search_agent`、`stream_arxiv_search_agent`
- `planner.py` 负责把用户目标映射成可执行计划
- `plan_executor.py` 负责执行计划步骤
- `replanner.py` 负责失败/低置信度后的计划修正
- `observer.py` 负责步骤结果观察与置信度判断
- `graph.py` 负责 Agent 图结构构建与 Mermaid 导出
- `tool_registry.py` 维护 Agent 自身的 planner tool registry
- `schemas.py`、`state.py` 定义 Agent 请求/响应/事件/状态模型

已确认执行关系：

- `agent_router.py` 不直接写业务逻辑，而是调用 `agents.arxiv_search_agent`
- Agent 内部不是直接把 HTTP 请求映射为单个 service 调用，而是先做计划、再逐步执行
- Agent 在执行步骤时会联动后端 `tools` 层和部分 service 能力

职责边界：

- Agent 层负责“多步规划、对话状态、待确认动作、工具编排、流式事件输出”
- Agent 不是底层存储或模型适配层
- Agent 也不是普通 router/service 的替代，而是位于它们之上的工作流编排层

### 3.5 Tools

代码目录：

- `backend/tools/*`

已确认职责：

- `tool_registry.py` 维护后端统一 `TOOL_REGISTRY`
- 每个 tool 用 `ToolSpec` 定义：
  - `name`
  - `description`
  - `input_schema`
  - `func`
  - `result_tool_name`
- 当前已确认注册的能力包括：
  - `search_arxiv_raw`
  - `search_arxiv_structured`
  - `get_paper_metadata`
  - `recommend_papers`
  - `record_paper_preference`
  - `remove_paper_preference`
  - `check_paper_qa_index`
  - `build_paper_qa_index`
  - `answer_paper_question`
- `invoke_tool()` 负责：
  - schema 校验
  - 调用具体 tool 函数
  - 统一封装返回 envelope

职责边界：

- tools 层负责把可被 Agent 调用的离散能力标准化
- tools 是“Agent 可消费的稳定接口层”
- tools 不等于 HTTP router，也不等于底层 service
- 具体业务实现仍然落在 `arxiv_tools.py`、`paper_qa_tools.py`、`recommendation_tools.py` 背后的 service 调用上

### 3.6 Storage / Database / Vector Store

代码入口：

- `backend/services/storage/database_service.py`
- `backend/services/storage/vector_store_service.py`
- `backend/services/arxiv/arxiv_oai_service.py`
- `backend/utils/config.py`

已确认存储分层：

- SQLite
  - `SQLITE_DATABASE_PATH`，默认 `06-database/recommendation.db`
  - `OAI_SQLITE_DATABASE_PATH`，默认 `backend/06-database/arxiv_oai.db`
- Milvus
  - `MILVUS_URI`，默认 `http://localhost:19530`

已确认 SQLite 负责的数据：

- 论文基础表 `arxiv_papers`
- 用户 like / dislike
- 用户兴趣向量
- 论文 QA 索引状态
- QA 索引作业状态
- 论文聊天会话与消息
- 用户论文行为
- 用户研究画像
- 论文笔记
- Agent session

已确认 Milvus 负责的数据：

- 论文 abstract embedding
- 论文 QA chunk embedding
- chunk 相关扩展元数据，如：
  - `content`
  - `rerank_text`
  - `chunk_type`
  - `asset_kind`
  - `asset_path`
  - `asset_summary`
  - `asset_caption`
  - `order_index`
  - `document_name`
  - 以及更多 QA / 检索回溯字段

职责边界：

- SQLite 负责强事务/业务对象状态
- Milvus 负责高维向量检索与 chunk 召回
- OAI SQLite 更偏 arXiv 本地元数据检索与同步缓存
- 文档资产文件和 trace 文件属于文件系统产物，不属于 SQLite/Milvus 本身

### 3.7 Model Provider / Embedding / Rerank / LLM

代码入口：

- `backend/utils/config.py`
- `backend/services/embedding/embedding_service.py`
- `backend/services/llm/generation_service.py`
- `backend/services/retrieval/rerank_service.py`
- `backend/services/retrieval/retrieval_pipeline.py`
- `backend/services/retrieval/retrieval_rules.py`
- `backend/services/retrieval/enhanced_retrieval_service.py`

已确认配置入口：

- Embedding
  - `EMBEDDING_PROVIDER`
  - `EMBEDDING_MODEL`
  - `EMBEDDING_BASE_URL`
  - `EMBEDDING_DIMENSION`
- Generation / LLM
  - `QWEN_API_KEY`
  - `OPENAI_API_KEY`
  - `DEEPSEEK_API_KEY`
  - 以及 `GENERATION_CONFIG` 中的 Qwen 模型路由配置
- Retrieval / Rerank
  - `ENHANCED_RETRIEVAL_*`
  - `MEMORY_RUNTIME_*`
  - rerank 相关策略在 retrieval service 中集中管理

已确认模型职责：

- `EmbeddingService`
  - 负责文本转向量
  - 为推荐、检索、QA 索引构建提供 embedding
- `GenerationService`
  - 负责通用生成
  - 负责 Qwen 大/小模型路由
  - 负责 `compress_chunk_for_rerank()` / `compress_chunks_for_rerank()`
- `EnhancedRetrievalService`??????????
  - 负责检索路线规划、query rewrite、HyDE、融合、final top-k、trace 导出
- `RerankService`???/?? rerank?fallback ??????
  - 负责重排逻辑
  - 与 asset-aware 文本、局部 LLM rerank / CrossEncoder 协同

`TODO`：

- 当前生产默认使用哪个 provider 组合，需要结合运行时环境变量和部署方式确认，仓库代码本身只能确认“支持哪些配置入口”，不能确认“当前真实线上默认值”
- `services/retrieval/rerank_service.py` 的完整默认策略虽然能继续深挖，但不在本轮入口梳理的最小必要范围内；若后续要做检索质量治理，建议单独补一篇检索链路文档

## 4. 关键调用链

### 4.1 前端到后端

```text
Vue View / Component
  -> api/request.ts (/api)
  -> FastAPI router
  -> dependencies.py 注入 service
  -> service / agent / tool
  -> SQLite / Milvus / 外部模型 / arXiv
```

### 4.2 论文 QA 主链路

根据 `qa_router.py`、`dependencies.py`、`paper_qa` / `retrieval` / `llm` 目录可确认的大致链路：

```text
PaperDetail.vue
  -> api/papers.ts (qa / qa/stream / create-qa-index / qa-status / qa-diagnose)
  -> qa_router.py
  -> PaperQAService / PaperQAIndexBuilder / IndexJobManager
  -> LoadingService / ChunkingService
  -> GenerationService.compress_chunks_for_rerank()
  -> EmbeddingService
  -> VectorStoreService (Milvus)
  -> EnhancedRetrievalService
  -> RetrievalPipeline
  -> GenerationService
  -> 返回答案、sources、retrieval_debug、trace
```

### 4.3 Agent 主链路

```text
AgentSearch.vue
  -> api/agent.ts
  -> /api/agent/chat or /api/agent/chat/stream
  -> agent_router.py
  -> run_arxiv_search_agent / stream_arxiv_search_agent
  -> planner / executor / replanner / observer
  -> agent tool registry
  -> backend tools
  -> recommendation / paper_qa / arxiv 等 service
  -> 返回最终响应、步骤事件、待确认动作、论文结果
```

### 4.4 推荐主链路

```text
Recommendations.vue / Profile / like-dislike actions
  -> api/papers.ts
  -> user_router.py / paper_router.py
  -> RecommendationService
  -> DatabaseService + MemoryService + EmbeddingService + VectorStoreService
  -> 生成兴趣向量 / 候选召回 / 排序 / 推荐解释
```

## 5. 当前已确认的系统边界

### 5.1 明确属于前端

- 页面路由
- UI 交互
- 请求参数与响应归一化
- SSE 事件消费
- 会话视图、证据视图、偏好按钮状态同步

### 5.2 明确属于后端 router

- API 暴露
- 请求校验与参数整理
- 依赖注入
- 异常转 HTTP 状态码

### 5.3 明确属于 service

- 业务流程实现
- 存储读写编排
- 文档解析与索引构建
- 检索与生成主逻辑
- 推荐与记忆逻辑

### 5.4 明确属于 agent

- 多步计划
- 工具编排
- 对话级状态
- 流式步骤事件
- 待确认动作与恢复

### 5.5 明确属于 tools

- Agent 可调用能力的 schema 化与统一返回格式

### 5.6 明确属于 storage / provider

- SQLite / OAI SQLite：关系数据和业务状态
- Milvus：向量和 chunk 检索
- arXiv API / OAI：外部学术数据源
- DashScope / OpenAI / DeepSeek / 本地 HF：模型能力提供者

## 6. 不确定项与 TODO

- `README.md` 当前在本地读取时存在明显编码显示异常；虽然仍能确认项目主题、目录和基础运行方式，但若要把 README 当作长期事实源，建议后续确认其文件编码是否统一为 UTF-8
- `agent_graph` 前端路由当前重定向到 `/`，但仓库中仍保留了 `AgentGraph.vue`；这意味着“是否废弃该页面”在代码层面没有完全收口，`TODO`
- `paper_router.py` 中的论文搜索能力在前端 `api/papers.ts` 里存在 `/papers/search` 调用，但本轮入口扫描未在 `paper_router.py` 前半段看到对应实现；需要进一步核对该接口是否位于文件后半段、已删除、或当前未使用，`TODO`
- `GenerationService` 支持的模型族与真实部署默认值不是一回事；当前只能确认“代码支持哪些 provider 和路由逻辑”，无法只靠仓库静态代码确认“当前部署默认使用哪个组合”，`TODO`
- `retrieval` 目录中 query rewrite、HyDE、route retriever、LLM rerank 的默认开启策略较多，若要做性能/质量调优，建议单独补一份“检索链路详解”，本篇只保留系统级边界，`TODO`

## 7. 作为维护者建议优先阅读的文件顺序

如果你是新维护者，建议按“入口 -> API 面 -> 依赖装配 -> 核心业务 -> 底层能力”的顺序读：

1. `README.md`
   - 先建立项目目标、目录和运行方式的整体认知
2. `backend/main.py`
   - 看清 FastAPI 入口、路由挂载顺序、preload/lazy 生命周期
3. `backend/dependencies.py`
   - 这是后端真实装配中心，决定了 service 如何组合
4. `new_frontend/src/router/index.ts`
   - 先理解前端有哪些主页面
5. `new_frontend/src/api/request.ts`
   - 看清前端统一 API 基座
6. `new_frontend/src/api/agent.ts`
   - 理解 Agent SSE 协议和前端消费方式
7. `new_frontend/src/api/papers.ts`
   - 理解论文、QA、推荐、笔记、诊断等主要接口面
8. `backend/routers/agent_router.py`
   - Agent HTTP 暴露层
9. `backend/routers/qa_router.py`
   - 论文 QA / 会话 / 笔记主入口
10. `backend/routers/user_router.py`
   - 用户偏好、画像、推荐入口
11. `backend/routers/paper_router.py`
   - 论文元数据、看板、同步状态入口
12. `backend/agents/arxiv_search_agent/__init__.py`
   - 先看 Agent 对外暴露了哪些公共能力
13. `backend/agents/arxiv_search_agent/service.py`
   - 看 Agent 真正的运行入口
14. `backend/agents/arxiv_search_agent/planner.py`
   - 看它如何把用户目标转换成执行计划
15. `backend/agents/arxiv_search_agent/plan_executor.py`
   - 看计划如何落地执行
16. `backend/tools/tool_registry.py`
   - 看 Agent 实际可调用哪些后端标准化工具
17. `backend/services/paper_qa/paper_qa_service.py`
   - 看单篇论文问答主流程
18. `backend/services/paper_qa/paper_qa_index_builder.py`
   - 看 QA 索引如何从 PDF 落到向量库
19. `backend/services/retrieval/enhanced_retrieval_service.py`
   - 看检索链路如何做 query planning、召回、融合、重排、trace
20. `backend/services/recommendation/recommendation_service.py`
   - 看推荐链路主编排
21. `backend/services/storage/database_service.py`
   - 看业务状态落在哪些 SQLite 表
22. `backend/services/storage/vector_store_service.py`
   - 看 Milvus schema 和向量写入方式
23. `backend/utils/config.py`
   - 最后统一回看可配置项，理解系统的运行时开关

## 8. 一句话结论

这个仓库的核心结构不是“前端直连若干零散后端接口”，而是一个分层相对清晰的 RAG 系统：前端负责交互，router 暴露 API，service 承载业务能力，agent 负责多步编排，tools 负责给 agent 提供稳定调用面，SQLite/Milvus/模型 provider 则构成底层能力基座。
