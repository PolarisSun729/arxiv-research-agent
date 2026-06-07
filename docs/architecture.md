 # 架构说明

 本文档用于回答两个核心问题：

 1. 这个项目由哪些子系统组成；
 2. 用户一次检索、问答、推荐或 Agent 请求，是如何在系统里流转的。

 配置项与环境变量请参考 [`docs/configuration.md`](./configuration.md)，快速启动请参考 [`README.md`](../README.md)。

---

 ## 1. 项目定位

 `rag-project01-framework` 是一个面向 arXiv 论文场景的全栈 RAG 系统，当前覆盖以下主要能力：

 - arXiv 论文检索
 - 单篇论文详情查看
 - 论文级问答与对话会话
 - 基于 chunk 的证据召回与调试
 - 用户偏好记录与研究画像
 - 个性化推荐
 - Agent 驱动的自然语言检索工作流
 - 本地 arXiv OAI 元数据同步与本地数据管理

 从仓库结构上看，这是一个典型的“前端 + API 后端 + 本地持久化 + 模型能力 + 向量检索”组合系统，而不是单一的脚本项目。

---

 ## 2. 系统整体结构

 ### 2.1 分层视图

 ```mermaid
 flowchart LR
     U[用户 / 浏览器]
     F[Vue 3 Frontend\nnew_frontend]
     V[Vite Dev Server\n/api 代理]
     B[FastAPI Backend\nbackend/main.py]

     R1[arxiv_router]
     R2[paper_router]
     R3[qa_router]
     R4[user_router]
     R5[agent_router]
     R6[chunk_router]

     S1[arXiv / 搜索服务]
     S2[RAG / Retrieval / QA 服务]
     S3[Recommendation / Memory 服务]
     S4[Agent Graph]
     S5[Document Loading / Chunking]

     D1[(SQLite)]
     D2[(Milvus)]
     D3[本地文件资产\nPDF / chunks / traces / results]
     D4[arXiv OAI 本地库]
     E1[外部模型服务\nEmbedding / Rerank / LLM]
     E2[arXiv API / OAI]

     U --> F
     F --> V
     V --> B

     B --> R1
     B --> R2
     B --> R3
     B --> R4
     B --> R5
     B --> R6

     R1 --> S1
     R2 --> S1
     R2 --> S3
     R3 --> S2
     R4 --> S3
     R5 --> S4
     R6 --> S5

     S1 --> D4
     S1 --> E2
     S2 --> D1
     S2 --> D2
     S2 --> D3
     S2 --> E1
     S3 --> D1
     S3 --> D2
     S3 --> D4
     S3 --> E1
     S4 --> S1
     S4 --> S2
     S4 --> S3
     S5 --> D3
     S5 --> E1
 ```

 ### 2.2 一句话理解每一层

 - **前端层**：负责页面展示、用户交互、调用 `/api/*` 接口。
 - **路由层**：负责把 HTTP 请求分发到对应业务能力。
 - **服务层**：负责检索、问答、推荐、Agent、解析、索引等核心逻辑。
 - **存储层**：SQLite、Milvus 与本地文件目录共同组成持久化基础。
 - **外部依赖层**：arXiv、OAI、Embedding、Rerank、LLM 等外部服务。

---

 ## 3. 前端架构

 前端位于 `new_frontend/`，技术栈为：

 - Vue 3
 - TypeScript
 - Vite
 - Pinia
 - Element Plus

 ### 3.1 前端职责

 前端主要负责：

 - 检索条件输入与结果展示
 - 论文详情与论文 QA 页面
 - 推荐列表与用户画像页面
 - Agent 检索交互
 - chunk 调试查看

 ### 3.2 页面路由

 当前主要页面来自 `new_frontend/src/router/index.ts`：

 - `/`：Dashboard
 - `/search`：论文检索
 - `/paper/:id`：论文详情与论文级问答
 - `/recommendations`：推荐列表
 - `/profile`：研究画像
 - `/agent-search`：Agent 检索
 - `/labeled`：已标注论文
 - `/chunks`：chunk 查看器

 ### 3.3 前后端联调方式

 本地开发中，`new_frontend/vite.config.ts` 将 `/api` 请求代理到：

 - `http://127.0.0.1:8001`

 因此前端本地开发链路如下：

 ```text
 Browser -> Vite Dev Server -> /api proxy -> FastAPI Backend
 ```

 这意味着：

 - 前端不直接感知后端真实服务地址；
 - 后端统一暴露 `/api/*` 路径；
 - 本地联调时无需单独处理跨域细节。

---

 ## 4. 后端架构

 后端入口是 `backend/main.py`，使用 FastAPI 提供统一 API 服务。

 ### 4.1 后端入口职责

 `backend/main.py` 主要负责：

 - 创建 FastAPI 应用
 - 注册 CORS 中间件
 - 挂载各业务路由
 - 根据 `load_mode` 控制服务预热方式

 当前挂载的路由包括：

 - `arxiv_router`：`/api/arxiv/*`
 - `agent_router`：`/api/agent/*`
 - `user_router`：`/api/user/*`
 - `paper_router`：`/api/*` 下的论文与统计接口
 - `qa_router`：`/api/paper/{arxiv_id}/*`
 - `chunk_router`：`/api/chunks/*`

 ### 4.2 后端加载模式

 后端支持两种加载模式：

 - `lazy`：按需初始化，启动快，首次请求更慢
 - `preload`：启动阶段预热，启动慢，但首次请求更稳定

 默认值来自 `backend/utils/config.py`，当前默认偏向 `preload`。

---

 ## 5. 核心子系统说明

 ### 5.1 arXiv 检索子系统

 入口：`backend/routers/arxiv_router.py`

 提供的能力包括：

 - 原始 query 检索
 - 结构化字段检索
 - 分类与字段列表获取
 - PDF 下载
 - 检索并保存本地数据

 这个子系统是“论文发现入口”，既可以直接查 arXiv，也可以结合本地 OAI 数据进行更稳定的数据支撑。

---

 ### 5.2 论文管理子系统

 入口：`backend/routers/paper_router.py`

 主要负责：

 - Dashboard 聚合统计
 - 同步状态查询
 - 单篇论文新增 / 查询 / 删除
 - 论文分类查询

 其中 `add_paper` 不只是写数据库，还会：

 1. 生成摘要 embedding；
 2. 写入向量库；
 3. 再将论文元数据写入数据库。

 所以它本质上承担的是“**论文入库并变成可检索对象**”的职责。

---

 ### 5.3 论文 QA / RAG 子系统

 入口：`backend/routers/qa_router.py`

 主要负责：

 - QA 索引构建
 - QA 状态查询
 - 检索 trace 导出
 - 聊天会话管理
 - 论文笔记管理
 - 同步 / 流式问答

 这是当前最核心的 RAG 入口之一。它的设计目标不是简单“问一个问题返回一个答案”，而是支持：

 - 论文粒度索引构建
 - chunk 级证据召回
 - 历史对话上下文
 - 调试用 trace 导出
 - 前端页面上的连续问答体验

---

 ### 5.4 推荐与用户画像子系统

 入口：`backend/routers/user_router.py`

 主要负责：

 - like / dislike 反馈
 - 论文行为记录
 - 用户研究画像读写
 - 兴趣向量生成
 - 个性化推荐

 这个子系统的定位不是“附属功能”，而是项目的重要差异化能力。它让系统不仅能回答某篇论文的问题，还能逐步理解用户偏好并生成更个性化的候选论文。

---

 ### 5.5 Agent 检索子系统

 入口：

 - 路由：`backend/routers/agent_router.py`
 - 工作流图：`backend/agents/arxiv_search_agent/graph.py`

 提供的能力包括：

 - 同步 Agent 对话
 - 流式 Agent 对话
 - Agent 图结构导出

 这个子系统和普通检索接口的区别在于：

 - 它先理解用户意图；
 - 再决定走搜索、论文阅读、偏好更新还是确认分支；
 - 最后统一生成回复。

 从代码结构看，这里采用的是 **LangGraph 状态图编排**，更接近一个可调试、可扩展的多节点流程，而不是单次 prompt 调用。

---

 ### 5.6 chunk 调试子系统

 入口：`backend/routers/chunk_router.py`

 主要负责：

 - 列出本地 chunk JSON 文件
 - 返回具体 chunk 文件内容

 这个接口本质上偏“可观测性 / 调试支撑”，用于验证：

 - 文档切分是否正确
 - 页码与章节路径是否合理
 - 某篇论文的 chunk 结果是否已落盘

---

 ## 6. 数据层关系

 当前系统的数据不是只落在一种介质中，而是分散在三类存储中。

 ### 6.1 SQLite

 主要承载结构化数据与业务状态，例如：

 - 用户偏好与行为
 - 推荐相关状态
 - 论文元数据
 - QA 会话 / 消息 / 笔记 / 任务状态
 - OAI 同步状态的部分引用信息

 配置入口见：

 - `SQLITE_DATABASE_PATH`
 - `OAI_SQLITE_DATABASE_PATH`

 ### 6.2 Milvus

 主要承载向量检索相关数据，例如：

 - 论文摘要 embedding
 - chunk embedding
 - 推荐或召回依赖的语义向量

 对系统来说，Milvus 是“高召回语义检索能力”的关键基础设施。

 ### 6.3 本地文件系统

 主要承载中间产物、调试产物与文档资产，例如：

 - PDF / Docling 解析资产
 - chunk JSON
 - 搜索结果缓存
 - 生成结果
 - retrieval traces
 - OAI 同步状态文件

 因此，这个项目的本地磁盘不是临时实现细节，而是架构的一部分。

---

 ## 7. RAG 主链路

 下面描述“用户针对某篇论文发起问答”时的大致主链路。

 ```mermaid
 sequenceDiagram
     participant U as User
     participant F as Frontend
     participant Q as qa_router
     participant P as Paper QA Service
     participant R as Enhanced Retrieval
     participant V as Milvus
     participant DB as SQLite
     participant L as LLM / Rerank

     U->>F: 在论文详情页输入问题
     F->>Q: POST /api/paper/{arxiv_id}/...
     Q->>P: 创建或读取 QA 上下文
     P->>R: 发起增强检索
     R->>V: 向量召回候选 chunks
     R->>DB: 读取会话/元信息/调试状态
     R->>L: query rewrite / rerank / answer generation
     R-->>P: 返回最终上下文与调试信息
     P-->>Q: 生成回答、sources、trace
     Q-->>F: 返回同步结果或流式输出
     F-->>U: 展示答案与证据片段
 ```

 ### 7.1 主链路的关键阶段

 1. **问题输入**：用户在论文详情页发起问题。
 2. **会话归一化**：后端根据 `user_id`、`session_id` 等组装上下文。
 3. **检索增强**：进行 query rewrite、候选召回、rerank、上下文拼装。
 4. **上下文生成**：从向量库和本地元数据中选出最终证据片段。
 5. **答案生成**：由 LLM 根据上下文合成回答。
 6. **可观测性落盘**：将 trace、sources、会话消息等保存，便于复盘。

 ### 7.2 为什么要保留 trace

 对论文 QA 场景来说，用户往往不仅想知道“答案是什么”，还想知道：

 - 证据来自哪一页、哪一段；
 - 为什么召回的是这些 chunk；
 - 为什么某次问答效果变差。

 因此 trace 与调试信息是架构内建能力，而不是额外附加功能。

---

 ## 8. Agent 检索链路

 Agent 工作流定义在 `backend/agents/arxiv_search_agent/graph.py` 中。

 当前主图已经收敛为两个 LangGraph 节点，具体业务步骤由 `run_agent_turn` 内的 PlanExecutor 负责：

 - `parse_search_request`
 - `run_agent_turn`

 ### 8.1 Agent 流程图

 ```mermaid
 flowchart TD
     A[parse_search_request]
     B[run_agent_turn]
     C[PlanExecutor: plan / tool / observe / replan]
     D[LangGraph interrupt]
     E[Command resume]
     F[END]

     A --> B
     B --> C
     C -->|needs tool approval| D
     E --> C
     C -->|answered / failed / fallback| F
 ```

 ### 8.2 Agent 与普通检索接口的区别

 普通检索接口更像“用户给条件，系统返回结果”；
 Agent 检索则更像“用户表达意图，系统决定采取哪种动作”。

 因此 Agent 的价值主要体现在：

 - 支持更自然的输入方式
 - 支持基于 LangGraph interrupt/resume 的工具确认续接
 - 支持搜索失败后的放宽检索重试
 - 支持个性化排序与注释增强
 - 支持将搜索、论文阅读、偏好更新统一进一张图里

---

 ## 9. 推荐系统链路

 推荐相关接口主要位于 `backend/routers/user_router.py`。

 推荐子系统依赖以下几类输入信号：

 - 用户显式反馈：like / dislike
 - 用户行为信号：浏览、收藏、加入会话等 action
 - 用户研究画像：主题偏好、类别偏好、回答风格
 - 论文语义向量：用于做兴趣建模和语义相似度计算
 - 本地论文池：来自 OAI 或本地入库数据

 ### 9.1 推荐链路概览

 ```mermaid
 flowchart LR
     A[用户 like / dislike / action]
     B[user_router]
     C[Recommendation Service]
     D[(SQLite 用户行为/偏好)]
     E[(Milvus 论文向量)]
     F[Memory / Research Profile]
     G[候选召回与打分]
     H[推荐结果]

     A --> B --> C
     C --> D
     C --> E
     C --> F
     D --> G
     E --> G
     F --> G
     G --> H
 ```

 ### 9.2 推荐系统的核心思想

 推荐不是单一相似度检索，而是多因素组合：

 - 语义相似性
 - 类别偏好
 - 时效性
 - 负反馈惩罚
 - 兴趣簇多样性

 从 `backend/utils/config.py` 可见，推荐链路已经暴露了较多聚类与权重参数，说明它是一个具备进一步调优空间的独立子系统。

---

 ## 10. 本地数据目录与持久化边界

 当前项目的数据目录大致如下：

 | 路径 | 作用 | 是否建议持久化 |
 | --- | --- | --- |
 | `backend/06-database/` | SQLite 数据库 | 是 |
 | `backend/03-vector-store/` | 向量存储相关文件 | 是 |
 | `backend/03-docling-assets/` | Docling 解析资产 | 是 |
 | `backend/04-search-results/` | 检索结果输出 | 视情况而定 |
 | `backend/05-generation-results/` | 生成结果输出 | 视情况而定 |
 | `backend/06-daily-arxiv-paper/` | arXiv 日更数据 | 建议持久化 |
 | `backend/07-arxiv-tools/` | 同步脚本与状态文件 | 脚本本身是代码，状态文件建议保留 |
 | `backend/01-loaded-docs/` | chunk 调试产物 | 调试环境建议保留 |
 | `temp/retrieval-traces/` | 检索 trace | 调试环境建议保留 |

 ### 10.1 持久化边界建议

 可以把这些目录分成三类：

 #### 必须保留

 - SQLite 数据库
 - 向量存储数据
 - Docling 解析资产
 - OAI 数据库与同步状态

 这些数据直接影响：

 - 能否继续提供检索/问答/推荐能力
 - 是否需要重新构建索引
 - 是否会丢失用户行为与偏好数据

 #### 建议保留

 - 日更论文数据
 - chunk 调试产物
 - retrieval traces

 这些数据不一定决定系统是否能启动，但对问题排查非常有价值。

 #### 可重建 / 可清理

 - 部分搜索结果输出
 - 部分生成结果输出
 - 临时调试文件

 是否清理，取决于你是否仍需要复盘历史运行过程。

---

 ## 11. 配置如何影响架构行为

 当前很多架构行为都由 `backend/utils/config.py` 驱动，典型包括：

 - `BACKEND_SERVICE_LOAD_MODE`：影响服务预热方式
 - `MILVUS_URI`：影响向量库连接
 - `SQLITE_DATABASE_PATH` / `OAI_SQLITE_DATABASE_PATH`：影响本地数据库位置
 - `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL`：影响向量化能力
 - `RERANK_PROVIDER`：影响重排策略
 - `ENABLE_SHORT_TERM_MEMORY` 等：影响会话与记忆链路
 - `ARXIV_OAI_*`：影响 OAI 同步行为

 所以从架构角度，这个项目不是“代码写死逻辑”，而是“**代码 + 配置共同决定运行形态**”。

---

 ## 12. 当前架构特点总结

 ### 12.1 优点

 当前架构已经具备比较明确的模块边界：

 - 前端与后端分离
 - 路由层相对轻量
 - Agent 独立成状态图
 - RAG、推荐、用户画像各自成子系统
 - 本地调试可观测性较强

 ### 12.2 当前需要持续关注的复杂点

 随着能力增多，后续维护时要重点关注：

 - SQLite、Milvus、本地文件三种持久化的一致性
 - QA、推荐、Agent 三条链路对模型配置的耦合
 - 本地同步数据与线上回源数据的边界
 - trace、chunk、生成结果等调试资产的清理策略
 - preload 模式下的启动成本

---

 ## 13. 给新维护者的快速理解顺序

 如果你是第一次接手这个项目，建议按下面顺序阅读：

 1. `README.md`：先理解项目目标和启动方式
 2. `docs/configuration.md`：理解运行依赖和关键环境变量
 3. `backend/main.py`：理解后端总入口和路由装配
 4. `new_frontend/src/router/index.ts`：理解前端页面能力边界
 5. `backend/routers/qa_router.py`：理解论文 QA 主链路入口
 6. `backend/routers/user_router.py`：理解推荐与画像能力
 7. `backend/agents/arxiv_search_agent/graph.py`：理解 Agent 工作流
