 # Backend Test Plan

 本文档是当前 `backend/` 的系统化测试总规划。

 约束说明：
 - 本阶段只产出测试规划，不新增任何测试文件。
 - 本阶段不修改业务逻辑。
 - 当前仓库里已有少量 `unittest` 风格测试，后续规划优先兼容现状，再逐步补齐更系统的测试分层。

 ---

 ## 0. 扫描范围与结论摘要

 本次规划覆盖以下后端区域：

 - 应用入口与依赖装配：`main.py`、`dependencies.py`
 - FastAPI 路由：`routers/`
 - 核心服务：`services/`
 - Agent：`agents/arxiv_search_agent/`
 - 工具层：`tools/`
 - 配置与通用工具：`utils/`
 - 当前已有测试：`backend/tests/`

 当前后端的高风险测试面主要集中在：

 1. FastAPI 路由参数校验与错误码映射
 2. SQLite / Milvus / 本地文件三类存储的一致性
 3. RAG 检索链路中的 query rewrite / recall / rerank / generation 串联
 4. QA 索引构建流程中的 PDF 下载、加载、切块、向量写入、任务状态更新
 5. Recommendation 与 Memory 对用户状态的读写闭环
 6. LangGraph Agent 的分支路由、工具调用、流式输出、记忆落库

 当前已有测试文件：

 - `backend/tests/test_agent_planning_stage1.py`
 - `backend/tests/test_tool_node_stage2.py`
 - `backend/tests/test_search_tool_protocol_stage2.py`
 - `backend/tests/test_paper_reference_resolver.py`
 - `backend/tests/test_search_node_personalized_rerank.py`

 它们主要覆盖 Agent 规划、工具协议、论文引用解析、个性化重排失败保护，说明现有测试重心偏 Agent，而对 router / service / RAG 主链路覆盖明显不足。

 ---

 ## 1. 后端模块清单

 ### 1.1 应用入口与依赖装配

 - `backend/main.py`
 - `backend/dependencies.py`
 - `backend/utils/config.py`

 ### 1.2 HTTP 路由层

 - `backend/routers/arxiv_router.py`
 - `backend/routers/agent_router.py`
 - `backend/routers/paper_router.py`
 - `backend/routers/qa_router.py`
 - `backend/routers/user_router.py`
 - `backend/routers/chunk_router.py`
 - `backend/routers/qa_utils.py`

 ### 1.3 arXiv 相关服务

 - `backend/services/arxiv/arxiv_search_service.py`
 - `backend/services/arxiv/local_arxiv_service.py`
 - `backend/services/arxiv/arxiv_oai_service.py`
 - `backend/services/arxiv/arxiv_query_builder.py`

 ### 1.4 文档处理服务

 - `backend/services/document/loading_service.py`
 - `backend/services/document/parsing_service.py`
 - `backend/services/document/chunking_service.py`

 ### 1.5 Embedding / LLM 服务

 - `backend/services/embedding/embedding_service.py`
 - `backend/services/llm/generation_service.py`

 ### 1.6 Retrieval / RAG 服务

 - `backend/services/retrieval/search_service.py`
 - `backend/services/retrieval/query_planner.py`
 - `backend/services/retrieval/route_retriever.py`
 - `backend/services/retrieval/rerank_service.py`
 - `backend/services/retrieval/enhanced_retrieval_service.py`

 ### 1.7 Paper QA 服务

 - `backend/services/paper_qa/paper_qa_service.py`
 - `backend/services/paper_qa/paper_qa_index_builder.py`
 - `backend/services/paper_qa/index_job_manager.py`

 ### 1.8 Recommendation / Memory / Intent 服务

 - `backend/services/recommendation/recommendation_service.py`
 - `backend/services/recommendation/candidate_recall_service.py`
 - `backend/services/recommendation/candidate_materializer.py`
 - `backend/services/recommendation/recommendation_ranker.py`
 - `backend/services/recommendation/interest_profile_service.py`
 - `backend/services/memory/memory_service.py`
 - `backend/services/memory/memory_models.py`
 - `backend/services/memory/memory_debug.py`
 - `backend/services/intent/intent_service.py`

 ### 1.9 存储服务

 - `backend/services/storage/database_service.py`
 - `backend/services/storage/vector_store_service.py`

 ### 1.10 Agent 与工具层

 - `backend/agents/arxiv_search_agent/service.py`
 - `backend/agents/arxiv_search_agent/graph.py`
 - `backend/agents/arxiv_search_agent/state.py`
 - `backend/agents/arxiv_search_agent/schemas.py`
 - `backend/agents/arxiv_search_agent/node/*.py`
 - `backend/agents/arxiv_search_agent/utils/*.py`
 - `backend/tools/tool_registry.py`
 - `backend/tools/arxiv_tools.py`
 - `backend/tools/paper_qa_tools.py`
 - `backend/tools/recommendation_tools.py`
 - `backend/tools/schemas.py`
 - `backend/tools/tool_result.py`

 ### 1.11 运维脚本与本地资产边界

 - `backend/07-arxiv-tools/*`
 - `backend/01-loaded-docs/`
 - `backend/03-vector-store/`
 - `backend/05-generation-results/`
 - `temp/retrieval-traces/`（由服务写出）

 ---

 ## 2. 每个模块的核心类和函数

 ### 2.1 应用入口与依赖装配

 - `main.py`
   - `create_app`
   - lifespan preload 分支
 - `dependencies.py`
   - `get_database_service`
   - `get_oai_database_service`
   - `get_embedding_service`
   - `get_vector_store_service`
   - `get_generation_service`
   - `get_memory_service`
   - `get_enhanced_retrieval_service`
   - `get_arxiv_service`
   - `get_paper_qa_index_builder`
   - `get_index_job_manager`
   - `get_recommendation_service`
   - `get_paper_qa_service`
   - `normalize_service_load_mode`
   - `iter_service_getters`
   - `warm_up_services`

 ### 2.2 路由层

 - `arxiv_router.py`
   - `arxiv_search`
   - `arxiv_get_fields`
   - `arxiv_get_categories`
   - `arxiv_download`
   - `arxiv_search_and_save`
 - `agent_router.py`
   - `agent_chat_endpoint`
   - `agent_chat_stream_endpoint`
   - `agent_graph_endpoint`
 - `paper_router.py`
   - `get_dashboard_stats`
   - `get_sync_status`
   - `add_paper`
   - `get_paper`
   - `delete_paper`
   - `get_all_papers`
   - `search_papers_by_category`
 - `qa_router.py`
   - 请求模型：`QaRequest`、`CreatePaperChatSessionRequest`、`PaperNoteRequest`、`UpdatePaperNoteRequest`
   - helper：`_normalize_user_id`、`_serialize_chat_session`、`_serialize_chat_message`、`_serialize_qa_index_job`、`_serialize_paper_note`、`_build_notes_markdown`
   - endpoint：`get_paper_qa_status`、`diagnose_paper_qa`、`download_latest_qa_trace`、`create_paper_qa_index`、`get_latest_paper_qa_index_job`、`get_paper_qa_index_job`、`list_paper_chat_sessions`、`get_recent_paper_chat_session`、`create_paper_chat_session`、`get_paper_chat_session`、`get_paper_chat_messages`、`clear_paper_chat_session`、`delete_paper_chat_session`、`list_paper_notes`、`create_paper_note`、`update_paper_note`、`delete_paper_note`、`export_paper_notes_markdown`、`qa_paper`、`qa_paper_stream`
 - `user_router.py`
   - 请求模型：`PaperActionRequest`、`ResearchProfileRequest`
   - endpoint：`upsert_user_preferences`、`get_user_preferences`、`like_paper`、`dislike_paper`、`record_paper_action`、`remove_paper_action`、`get_user_paper_actions`、`get_user_research_profile`、`upsert_user_research_profile`、`patch_user_research_profile`、`remove_like`、`remove_dislike`、`generate_user_interest_vector`、`get_user_interest_vector`、`recommend_papers`
 - `chunk_router.py`
   - `list_chunk_files`
   - `get_chunk_file`
 - `qa_utils.py`
   - `sanitize_trace_slug`
   - `get_latest_retrieval_trace`
   - `build_qa_diagnostic`

 ### 2.3 arXiv 服务

 - `arxiv_search_service.py`
   - `ArxivSearchService`
   - `RateLimitError`
   - `SearchField`
 - `local_arxiv_service.py`
   - `LocalArxivService`
 - `arxiv_oai_service.py`
   - `ArxivOaiSyncStats`
   - `ArxivOaiDatabaseService`
   - `ArxivOaiSyncService`
 - `arxiv_query_builder.py`
   - `ArxivSearchValidationError`
   - `normalize_text_value`
   - `quote_arxiv_text`
   - `build_arxiv_field_clause`
   - `combine_arxiv_clauses`
   - `validate_arxiv_search_request`
   - `build_arxiv_raw_query`
   - `build_arxiv_query_from_structured_params`

 ### 2.4 文档处理服务

 - `loading_service.py`
   - `LoadingService`
 - `parsing_service.py`
   - `ParsingService`
 - `chunking_service.py`
   - `ChunkingService`

 ### 2.5 Embedding / LLM

 - `embedding_service.py`
   - `EmbeddingProvider`
   - `EmbeddingConfig`
   - `EmbeddingService`
   - `EmbeddingFactory`
 - `generation_service.py`
   - `GenerationService`
   - 重点关注其生成、路由、压缩、流式相关方法

 ### 2.6 Retrieval / RAG

 - `search_service.py`
   - `SearchService`
 - `query_planner.py`
   - `QueryPlanner`
 - `route_retriever.py`
   - `RouteRetriever`
 - `rerank_service.py`
   - `RerankService`
 - `enhanced_retrieval_service.py`
   - `RetrievalOptions`
   - `QueryProfile`
   - `EnhancedRetrievalService`

 ### 2.7 Paper QA

 - `paper_qa_service.py`
   - `PaperQAService`
 - `paper_qa_index_builder.py`
   - `PaperQAIndexBuilder`
 - `index_job_manager.py`
   - `IndexJobManager`

 ### 2.8 Recommendation / Memory / Intent

 - `recommendation_service.py`
   - `RecommendationService`
 - `candidate_recall_service.py`
   - `CandidateRecallService`
 - `candidate_materializer.py`
   - `CandidateMaterializer`
 - `recommendation_ranker.py`
   - `RecommendationRanker`
 - `interest_profile_service.py`
   - `InterestProfileService`
 - `memory_service.py`
   - `MemoryService`
 - `memory_models.py`
   - `PreferenceSummary`
   - `PaperChatHistory`
   - `BackendMemorySnapshot`
   - `AgentSessionMemory`
   - `MemoryDebugPayload`
 - `memory_debug.py`
   - `build_memory_debug_payload`
 - `intent_service.py`
   - `_normalize_text`
   - `_tokenize`
   - `_dedupe`
   - `IntentProfile`
   - `IntentService`

 ### 2.9 存储服务

 - `database_service.py`
   - `DatabaseService`
 - `vector_store_service.py`
   - `normalize_collection_name`
   - `is_valid_collection_name`
   - `VectorDBConfig`
   - `VectorStoreService`

 ### 2.10 Agent 与工具层

 - `service.py`
   - `_inject_user_memory_context`
   - `_load_agent_request_context`
   - `_persist_agent_session_memory`
   - `run_arxiv_search_agent`
   - `stream_arxiv_search_agent`
 - `graph.py`
   - `route_after_parse`
   - `build_arxiv_search_graph`
   - `export_arxiv_search_graph_mermaid`
 - `node/*.py`
   - `parse_search_request`
   - `run_agent_turn`
   - `handle_paper_reading_request`
   - `execute_tool`
 - `plan_executor.py`
   - `PlanExecutor`
   - `run_agent_turn`
   - `run_agent_turn_in_graph`
   - `Command(resume=...)`
 - `utils/*.py`
   - `paper_reference_resolver`
   - `search_spec_builder`
   - `state_utils`
   - `result_utils`
   - `text_utils`
 - `tool_registry.py`
   - `ToolSpec`
   - `_validate_arguments`
   - `get_tool_registry`
   - `get_tool_names`
   - `invoke_tool`
 - `arxiv_tools.py`
   - `search_arxiv_raw`
   - `search_arxiv_structured`
   - `get_paper_metadata`
 - `paper_qa_tools.py`
   - `check_paper_qa_index`
   - `build_paper_qa_index`
   - `answer_paper_question`
 - `recommendation_tools.py`
   - `recommend_papers`
   - `record_paper_preference`
 - `tool_result.py`
   - `make_tool_result`
   - `make_tool_trace`
   - `make_tool_error`

 ---

 ## 3. 每个模块的外部依赖

 ### 3.1 Web / 框架

 - `fastapi`
 - `starlette`
 - `uvicorn`

 ### 3.2 数据与存储

 - `sqlite3`
 - `pymilvus` / `MilvusClient`
 - 本地文件系统

 ### 3.3 网络与上游服务

 - arXiv API
 - arXiv OAI-PMH
 - `requests`
 - `feedparser`
 - 可选代理配置

 ### 3.4 模型与推理

 - `openai`
 - DashScope / Qwen 兼容接口
 - DeepSeek 兼容接口
 - HuggingFace / `transformers`
 - `torch`
 - `numpy`
 - `sentence-transformers`
 - 可选 cross-encoder rerank

 ### 3.5 文档处理

 - `PyMuPDF`
 - `docling`
 - PDF 文件 IO

 ### 3.6 Agent / 编排

 - `langgraph`
 - SSE / `StreamingResponse`
 - 内部工具注册表

 ### 3.7 其他重要运行时边界

 - 环境变量与配置文件
 - `lru_cache` 单例缓存
 - 后台线程（QA index job）
 - trace / markdown / chunk JSON 文件

 ---

 ## 4. 每个模块推荐的测试类型

 ### 4.1 应用入口与依赖装配

 - 单元测试
   - `normalize_service_load_mode`
   - `iter_service_getters`
 - 组件测试
   - `create_app` 路由挂载完整性
   - preload / lazy 分支
 - 契约测试
   - startup 时是否调用预热逻辑

 ### 4.2 路由层

 - API/router 测试
   - 参数校验
   - 状态码
   - 依赖覆盖
   - 错误映射
 - 流式接口测试
   - SSE event shape
   - `StreamingResponse` 内容结构
 - 文件接口测试
   - `FileResponse`
   - markdown export

 ### 4.3 arXiv 服务

 - 纯单元测试
   - query builder / validation
 - service mock 测试
   - search / download / save 流程
 - 网络隔离测试
   - API 返回异常、超时、rate limit
 - OAI DB 测试
   - 可使用临时 SQLite 验证 OAI 查询逻辑

 ### 4.4 文档处理服务

 - 单元测试
   - chunk 切分规则
   - metadata 填充
 - 文件型组件测试
   - 本地 PDF / JSON artifact 路径
 - mock-heavy 测试
   - 替代真实 Docling / PyMuPDF 解析过程

 ### 4.5 Embedding / LLM

 - 单元测试
   - provider 选择
   - payload 构造
   - fallback 分支
 - mock service 测试
   - API client 调用与错误包装
 - 非联网组件测试
   - cache / normalization / text building

 ### 4.6 Retrieval / RAG

 - 单元测试
   - query planning
   - route fusion
   - rerank 开关与阈值
 - 组件测试
   - fake vector search -> rerank -> final context selection
 - trace 导出测试
   - markdown / json trace 内容结构

 ### 4.7 Paper QA

 - 单元测试
   - 状态组合、序列化、上下文构造
 - 组件测试
   - build index 同步链路
   - answer question 链路
 - 后台任务测试
   - `IndexJobManager` 提交与状态推进
 - 有限集成测试
   - 临时 SQLite + fake vector store + fake services

 ### 4.8 Recommendation / Memory / Intent

 - 单元测试
   - score / penalty / preference merge / heuristic intent
 - 组件测试
   - interest vector 生成与推荐排序
 - repository-style 测试
   - memory/profile/chat/note/action 在 SQLite 中的读写

 ### 4.9 存储服务

 - repository 测试
   - `DatabaseService` 使用临时 SQLite
 - adapter 测试
   - `VectorStoreService` 全 mock Milvus client
 - 序列化测试
   - JSON 字段、时间字段、集合名合法性

 ### 4.10 Agent 与工具层

 - 纯单元测试
   - parse / state / resolver / tool schema
 - graph 测试
   - 路由分支、空结果重试、确认流
 - 工具协议测试
   - registry validate -> invoke -> trace
 - 流式测试
   - agent SSE event 序列

 ---

 ## 5. 每个 FastAPI endpoint 的测试清单

 说明：以下只列测试要点，不在本阶段落地测试文件。

 ### 5.1 arXiv endpoints

 #### `POST /api/arxiv/search`
 - 原始 query 成功
 - 结构化字段搜索成功
 - 原始 query 与结构化字段优先级
 - `max_results/start/sort_by/sort_order` 校验
 - 非法 query 返回 400
 - service 抛异常返回 500

 #### `GET /api/arxiv/fields`
 - 成功返回字段列表
 - service 异常返回 500

 #### `GET /api/arxiv/categories`
 - 成功返回分类列表
 - service 异常返回 500

 #### `POST /api/arxiv/download`
 - 下载成功返回 filepath
 - service 异常返回 500

 #### `POST /api/arxiv/search-and-save`
 - 仅搜索成功
 - 搜索并下载 PDF 成功
 - service 异常返回 500

 ### 5.2 agent endpoints

 #### `POST /api/agent/chat`
 - 正常对话成功
 - 请求模型校验失败
 - 内部 agent 抛异常时返回 500

 #### `POST /api/agent/chat/stream`
 - 返回 `StreamingResponse`
 - 事件顺序符合预期
 - agent 内部错误事件能暴露给前端

 #### `GET /api/agent/graph`
 - 返回 graph 数据
 - graph 导出失败时的错误映射

 ### 5.3 paper endpoints

 #### `GET /api/stats`
 - 统计聚合成功
 - sync 元数据文件不存在时兜底
 - OAI / main DB 任一失败时返回 500

 #### `GET /api/sync-status`
 - 正常读取 state/meta
 - JSON 损坏时兜底
 - 文件缺失时默认值

 #### `POST /api/paper`
 - embedding + vector insert + DB add 全成功
 - embedding 失败
 - vector insert 失败
 - DB 写入失败
 - 返回 payload 中含 vector metadata

 #### `GET /api/paper/{arxiv_id}`
 - 本地 DB 命中
 - DB 未命中后回源成功
 - 最终未找到返回 404
 - 回源失败返回 500

 #### `DELETE /api/paper/{arxiv_id}`
 - 删除成功
 - 未找到的处理
 - DB 异常

 #### `GET /api/papers`
 - 返回全量列表
 - DB 异常

 #### `GET /api/papers/category/{category}`
 - 分类命中
 - 空结果
 - DB 异常

 ### 5.4 QA endpoints

 #### `GET /api/paper/{arxiv_id}/qa-status`
 - 已建索引
 - 未建索引
 - service 异常

 #### `GET /api/paper/{arxiv_id}/qa-diagnose`
 - 诊断信息成功
 - `sample_limit` 边界
 - vector store / DB 异常

 #### `GET /api/paper/{arxiv_id}/qa-trace/latest`
 - 默认最新 md 下载成功
 - 指定 trace_name 成功
 - `format=json` 成功
 - 非法 format 返回 400
 - 路径穿越输入拒绝
 - 文件不存在返回 404

 #### `POST /api/paper/{arxiv_id}/create-qa-index`
 - `sync=true` 同步成功
 - 异步提交成功
 - 非法 loading method
 - builder / job manager 异常

 #### `GET /api/paper/{arxiv_id}/qa-index-jobs/latest`
 - 返回最新任务
 - 没有任务返回 404

 #### `GET /api/paper/{arxiv_id}/qa-index-jobs/{job_id}`
 - 返回指定任务
 - 任务存在但 arxiv_id 不匹配返回 404

 #### `GET /api/paper/{arxiv_id}/chat-sessions`
 - 正常分页返回
 - `limit` 边界

 #### `GET /api/paper/{arxiv_id}/chat-sessions/recent`
 - 有最近会话
 - 无会话时空结果结构

 #### `POST /api/paper/{arxiv_id}/chat-sessions`
 - 创建成功
 - 默认 user_id 生效
 - DB 创建失败

 #### `GET /api/paper/{arxiv_id}/chat-sessions/{session_id}`
 - 成功读取
 - 会话不属于当前 paper 返回 404

 #### `GET /api/paper/{arxiv_id}/chat-sessions/{session_id}/messages`
 - 成功返回消息列表
 - session 不存在或 paper 不匹配返回 404

 #### `POST /api/paper/{arxiv_id}/chat-sessions/{session_id}/clear`
 - 清空成功并返回刷新后的 session
 - session 不存在返回 404

 #### `DELETE /api/paper/{arxiv_id}/chat-sessions/{session_id}`
 - 删除成功
 - session 不存在返回 404

 #### `GET /api/paper/{arxiv_id}/notes`
 - 正常列出
 - note_type 过滤

 #### `POST /api/paper/{arxiv_id}/notes`
 - 普通创建成功
 - 通过 `session_id + source_turn_id` 回查 source message
 - `include_in_profile=true` 时调用 memory 更新
 - DB 异常

 #### `PATCH /api/paper/{arxiv_id}/notes/{note_id}`
 - 正常更新
 - 不存在返回 404
 - `include_in_profile=true` 时重新同步画像

 #### `DELETE /api/paper/{arxiv_id}/notes/{note_id}`
 - 删除成功
 - 不存在返回 404

 #### `GET /api/paper/{arxiv_id}/notes/export`
 - 返回 markdown 流
 - 文件名 slug 正确
 - notes 为空时仍能导出

 #### `POST /api/paper/{arxiv_id}/qa`
 - 正常问答成功
 - 请求体缺字段失败
 - service 异常返回 500

 #### `POST /api/paper/{arxiv_id}/qa/stream`
 - 返回 meta / delta / done 事件
 - contextualized question 生效
 - generation 过程中异常时发送 error 事件
 - sources 与 retrieval debug 进入输出

 ### 5.5 user endpoints

 #### `POST /api/user/preferences`
 - 返回指定用户偏好
 - DB 异常

 #### `GET /api/user/preferences/{user_id}`
 - 返回用户偏好
 - DB 异常

 #### `POST /api/user/like-paper`
 - 成功记录正反馈
 - 传 paper payload 时透传
 - service 异常

 #### `POST /api/user/dislike-paper`
 - 成功记录负反馈
 - service 异常

 #### `POST /api/user/paper-action`
 - 成功记录通用行为
 - metadata 透传
 - service 异常

 #### `DELETE /api/user/paper-action`
 - 删除成功
 - 未找到返回 404

 #### `GET /api/user/paper-actions/{user_id}`
 - 返回 actions 和 action_map
 - 可按 action_type 过滤

 #### `GET /api/user/research-profile/{user_id}`
 - 返回画像
 - service 异常

 #### `PUT /api/user/research-profile`
 - 完整 upsert 成功
 - `exclude_none` 生效

 #### `PATCH /api/user/research-profile`
 - 局部 patch 成功
 - 空 patch 行为

 #### `DELETE /api/user/like-paper`
 - 移除成功
 - 失败路径

 #### `DELETE /api/user/dislike-paper`
 - 移除成功
 - 失败路径

 #### `POST /api/user/generate-interest-vector`
 - 生成成功
 - service 异常

 #### `GET /api/user/interest-vector`
 - 查询成功
 - 不存在返回 404

 #### `POST /api/user/recommend-papers`
 - 推荐成功
 - top_n / max_age_months 透传
 - service 异常

 ### 5.6 chunk endpoints

 #### `GET /api/chunks/files`
 - 目录存在时正常列出
 - 目录不存在返回空列表
 - 仅返回 JSON 文件

 #### `GET /api/chunks/file/{filename}`
 - 读取成功
 - 文件不存在返回 404
 - 非法 JSON / 读取异常返回 500

 ---

 ## 6. 每个业务流程的集成测试清单

 ### 6.1 应用启动流程

 - `create_app()` 正常挂载全部 router
 - `load_mode=lazy` 不触发重服务初始化
 - `load_mode=preload` 触发 `warm_up_services()` 和 tool/category cache 预热

 ### 6.2 arXiv 搜索流程

 - router -> query builder -> local service 成功链路
 - router -> query builder -> remote service 成功链路
 - 结构化搜索 + 分类过滤 + 分页
 - 上游失败 / 速率限制 / 非法 query 链路

 ### 6.3 论文入库流程

 - `POST /api/paper` -> embedding -> vector insert -> DB add
 - vector insert 成功但 DB 写失败时的行为确认
 - 同一篇论文重复写入策略确认

 ### 6.4 QA 索引构建流程

 - paper metadata 命中主 DB
 - 主 DB 未命中 -> OAI DB fallback
 - OAI DB 未命中 -> arXiv API fallback
 - 下载 PDF -> load -> chunk -> embed -> vector write -> DB status success
 - 任一阶段失败后 job 状态写为 failed

 ### 6.5 论文问答流程

 - resolve session -> contextualize question -> retrieval -> generation -> persist messages
 - 多轮对话带 `session_id`
 - source payload 与 retrieval debug 落库
 - QA trace 导出可下载

 ### 6.6 用户反馈与推荐流程

 - like/dislike/action -> DB 落库
 - generate interest vector -> 向量保存
 - recommend papers -> candidate recall -> rank -> return
 - negative feedback 对排序的影响

 ### 6.7 笔记与画像联动流程

 - 创建 note 并 `include_in_profile=true`
 - 更新 note 后再次同步 profile
 - 删除 note 不影响已有聊天消息完整性

 ### 6.8 Agent 对话流程

 - `chat` 正常走搜索分支
 - 搜索为空时放宽条件重试
 - preference action 分支
 - paper reading / paper QA 分支
 - pending confirmation 分支
 - tool registry 调用成功 / 参数校验失败 / tool not found
 - streaming path 输出步骤事件与最终事件

 ---

 ## 7. RAG 检索链路的专项测试清单

 RAG 专项测试聚焦 `PaperQAService + EnhancedRetrievalService + GenerationService + VectorStoreService`。

 ### 7.1 查询理解与改写

 - 原始问题直接检索
 - 开启 `enable_query_rewrite`
 - 开启 `enable_hyde`
 - 会话上下文参与 contextualization
 - 中英文混合问题

 ### 7.2 召回路径

 - semantic recall
 - keyword recall
 - 多 route fusion
 - top_k 边界
 - 空召回

 ### 7.3 rerank 与上下文裁剪

 - `enable_llm_rerank=true`
 - rerank 失败回退原始顺序
 - rerank 后 context 数量截断
 - chunk compression / summarization 生效

 ### 7.4 source payload 与证据质量

 - page number / section path / parent chunk id 完整
 - sources 顺序与最终上下文顺序一致
 - source 去重
 - 上下文过长时的截断策略

 ### 7.5 trace 与可观测性

 - 生成 md trace
 - 生成 json trace
 - trace 目录分桶按 `arxiv_id` slug
 - trace 内容包含 query / routes / selected chunks / final context

 ### 7.6 失败保护

 - embedding service 异常
 - vector search 异常
 - rerank service 异常
 - generation service 异常
 - retrieval 返回空但接口仍能给出可诊断错误

 ---

 ## 8. RAG golden set 评测方案

 ### 8.1 目标

 为论文级 QA 建立稳定的回归评测集，防止 query rewrite、rerank、chunking、prompt、provider 切换后效果无感退化。

 ### 8.2 golden set 数据结构建议

 每条样本至少包含：

 - `case_id`
 - `arxiv_id`
 - `paper_title`
 - `question`
 - `question_type`
   - factual
   - method
   - experiment
   - limitation
   - comparison
   - citation / evidence locating
 - `expected_answer_points`
 - `expected_source_constraints`
   - 页码范围
   - section 名称
   - 必须命中的 chunk 关键词
 - `difficulty`
 - `language`

 ### 8.3 样本覆盖范围

 - 每类问题至少 10-20 条
 - 中英文问题都要有
 - 短问题 / 长问题都要有
 - 单轮问题 / 带上下文追问都要有
 - 已索引论文与新建索引论文都要有

 ### 8.4 评测维度

 - Retrieval hit@k
 - evidence precision
 - evidence coverage
 - answer groundedness
 - answer completeness
 - hallucination rate
 - no-answer correctness（检索不到时是否谨慎）

 ### 8.5 判分机制

 推荐分三层：

 1. 自动指标
   - hit@k
   - 是否命中 expected section/page
   - source 数量与去重
 2. 规则指标
   - answer 中是否覆盖关键点
   - 是否包含明显编造信息
 3. 抽样人工评审
   - 每次改动后抽样 20-30 条高风险题

 ### 8.6 执行策略

 - P0 回归：小型 smoke golden set，10-20 条
 - P1 回归：主 golden set，50-100 条
 - P2 回归：扩展集，100+ 条，含长文、边界 case、多轮 case

 ### 8.7 通过门槛建议

 - smoke golden set：不得出现 blocker 级退化
 - 主 golden set：
   - hit@5 >= 85%
   - groundedness >= 90%
   - hallucination rate <= 5%
 - 扩展集：以趋势监控为主，不作为单次阻塞门槛

 ---

 ## 9. 需要 fake/mock 的服务列表

 ### 9.1 必须默认 fake/mock 的外部服务

 - Milvus / `VectorStoreService` 的真实 client
 - Embedding provider
 - Rerank provider
 - LLM provider
 - arXiv API
 - arXiv OAI 网络接口
 - PDF 下载网络请求
 - Docling / PyMuPDF 的重解析路径（大多数测试中）

 ### 9.2 建议 mock 的内部重服务

 - `GenerationService`
 - `EmbeddingService`
 - `RecommendationService`
 - `PaperQAService`
 - `EnhancedRetrievalService`
 - `ArxivSearchService`
 - `IndexJobManager`

 ### 9.3 可保留真实对象但隔离 IO 的服务

 - `DatabaseService`：允许临时 SQLite
 - `MemoryService`：当其底层 DB 指向临时 SQLite 时可部分走真实实现
 - `IntentService`：可优先走真实 heuristic 分支，LLM 分支 mock

 ---

 ## 10. 测试目录结构设计

 规划目录如下，后续逐步建设：

 ```text
 backend/tests/
   unit/
     test_dependencies.py
     test_config_runtime.py
     routers/
       test_qa_utils.py
     services/
       arxiv/
       document/
       embedding/
       llm/
       retrieval/
       paper_qa/
       recommendation/
       memory/
       storage/
       intent/
     agents/
       arxiv_search_agent/
     tools/

   api/
     test_arxiv_router.py
     test_agent_router.py
     test_paper_router.py
     test_qa_router.py
     test_user_router.py
     test_chunk_router.py

   integration/
     test_app_startup.py
     test_paper_ingestion_flow.py
     test_qa_index_build_flow.py
     test_qa_answer_flow.py
     test_recommendation_flow.py
     test_agent_chat_flow.py

   golden/
     data/
       smoke_golden_set.jsonl
       main_golden_set.jsonl
     test_rag_golden_smoke.py
     test_rag_golden_main.py

   fixtures/
     sample_papers/
     sample_chunks/
     sample_traces/
     sample_arxiv_responses/
     sample_oai_xml/
     sample_notes/
     sample_sessions/

   helpers/
     fake_services.py
     fake_vector_store.py
     fake_generation.py
     fake_embedding.py
     fake_arxiv.py
     temp_db.py
     temp_files.py
 ```

 兼容现状建议：

 - 现有 `backend/tests/test_*.py` 暂不迁移，后续新增测试优先放入分层目录。
 - 若后续继续沿用 `unittest`，目录结构依然成立。
 - 若未来切换到 `pytest`，目录无需再改，只需增加 fixture 与 runner 配置。

 ---

 ## 11. 测试优先级 P0 / P1 / P2

 ### P0：必须最先补齐

 目标：保证主业务链路不崩。

 - `dependencies.py` 关键 getter 与 preload/lazy 行为
 - `arxiv_query_builder.py` 的 query 组装与校验
 - `paper_router.add_paper`
 - `qa_router.create-qa-index`
 - `qa_router.qa`
 - `qa_router.qa/stream`
 - `qa_router.qa-trace/latest`
 - `user_router.like/dislike/recommend-papers`
 - `DatabaseService` 核心 repository 能力
 - `VectorStoreService` 的 collection name / search / insert 适配层
 - `PaperQAService` 基本问答链路
 - `PaperQAIndexBuilder` 阶段推进
 - `EnhancedRetrievalService` 基础 recall+rereank 选择逻辑
 - `Agent` 的搜索主分支、工具主分支、流式输出主分支

 ### P1：第二阶段补齐

 目标：提高回归稳定性与边界场景覆盖。

 - 全部 router 的成功 / 404 / 500 路径
 - 笔记、会话、画像联动
 - recommendation 排序与惩罚逻辑
 - intent heuristics 与 LLM fallback
 - chunk export / note export / sync status 文件边界
 - OAI DB fallback 与 arXiv fallback
 - agent pending confirmation / preference action / paper reading 分支

 ### P2：增强型测试

 目标：持续优化效果与可观测性。

 - RAG golden set 全量回归
 - 更细粒度的 trace 内容断言
 - 多语言、多轮会话、长文本边界
 - OAI sync 流程与脚本侧校验
 - provider-specific 行为差异测试

 ---

 ## 12. 每个阶段的验收命令

 说明：以下命令是后续测试实现完成后的验收建议。当前仓库已有测试以 `unittest` 为主，因此先以 `unittest` 作为基线。

 ### 阶段 A：现有测试与 Agent 基础回归

 ```powershell
 cd backend
 python -m unittest discover -s tests -p "test_*.py"
 ```

 ### 阶段 B：P0 单元 + API 回归

 ```powershell
 cd backend
 python -m unittest discover -s tests/unit -p "test_*.py"
 python -m unittest discover -s tests/api -p "test_*.py"
 ```

 ### 阶段 C：P0/P1 集成回归

 ```powershell
 cd backend
 python -m unittest discover -s tests/integration -p "test_*.py"
 ```

 ### 阶段 D：golden set smoke 回归

 ```powershell
 cd backend
 python -m unittest tests.golden.test_rag_golden_smoke
 ```

 ### 阶段 E：全量回归

 ```powershell
 cd backend
 python -m unittest discover -s tests -p "test_*.py"
 ```

 ### 阶段 F：覆盖率验收（后续引入 coverage 工具后执行）

 ```powershell
 cd backend
 python -m coverage run -m unittest discover -s tests -p "test_*.py"
 python -m coverage report -m
 ```

 ---

 ## 13. 覆盖率目标

 说明：覆盖率目标按“可维护且有价值”原则设定，不追求无意义刷数。

 ### 总体目标

 - P0 阶段：backend 新增测试覆盖到关键主链路，整体语句覆盖率目标 `>= 55%`
 - P1 阶段：整体语句覆盖率目标 `>= 70%`
 - P2 阶段：整体语句覆盖率目标 `>= 80%`

 ### 分模块目标

 - `routers/`：`>= 80%`
 - `services/arxiv/arxiv_query_builder.py`：`>= 90%`
 - `services/storage/database_service.py`：`>= 75%`
 - `services/retrieval/`：`>= 70%`
 - `services/paper_qa/`：`>= 70%`
 - `services/recommendation/`：`>= 70%`
 - `agents/arxiv_search_agent/`：`>= 80%`
 - `tools/`：`>= 85%`

 ### 特别说明

 以下模块不以语句覆盖率为唯一目标，更看重高价值场景覆盖：

 - `generation_service.py`
 - `embedding_service.py`
 - `loading_service.py`
 - `chunking_service.py`
 - `arxiv_oai_service.py`

 原因：这些模块外部依赖重、分支复杂、包含大量 provider/file/network 边界，更适合以关键路径与失败保护用例为主。

 ---

 ## 14. 哪些测试不能连接真实 Milvus、Embedding、Rerank、LLM、arXiv 网络

 默认原则：除明确标记为“手工验收 / 外部契约 / 联调 smoke”的测试外，自动化测试一律不能连接真实外部服务。

 ### 14.1 禁止连接真实外部服务的测试类别

 - 所有 `unit/` 测试
 - 所有 `api/` 测试
 - 绝大多数 `integration/` 测试
 - 所有 `golden/` 自动回归测试
 - CI 中跑的全部测试

 ### 14.2 明确禁止真实连接的模块/场景

 - `VectorStoreService` 相关测试不能连真实 Milvus
 - `EmbeddingService` 相关测试不能打真实 embedding provider
 - `RerankService` / generation compression 测试不能打真实 rerank provider
 - `GenerationService` / QA / Agent / Recommendation 中所有 LLM 调用不能打真实 LLM
 - `ArxivSearchService` / `ArxivOaiSyncService` / router arXiv endpoints 测试不能打真实 arXiv 网络
 - `PaperQAIndexBuilder` 中 PDF 下载测试不能访问真实网络

 ### 14.3 唯一允许真实连接的场景

 仅允许单独维护的“手工联调 smoke”脚本或不进 CI 的本地验证步骤中使用真实服务，且必须显式标记：

 - `manual only`
 - `not for CI`
 - `requires external services`

 这些不属于本测试规划的自动化主线。

 ---

 ## 15. 哪些测试允许使用临时 SQLite

 允许，而且推荐使用临时 SQLite 的测试类别如下。

 ### 15.1 强烈推荐使用临时 SQLite 的测试

 - `DatabaseService` repository 测试
 - `MemoryService` 读写测试
 - `user_router` 中 preferences / actions / likes / dislikes / interest vector 的 API 测试
 - `qa_router` 中 chat sessions / messages / notes / jobs / qa-status 的 API 测试
 - `paper_router` 中 stats / paper CRUD 的部分集成测试
 - `PaperQAService` 中会话、消息、笔记落库相关测试
 - `RecommendationService` 中用户偏好、兴趣向量、推荐结果落库相关测试
 - `Agent` memory persistence 测试

 ### 15.2 可以使用临时 SQLite，但仍需 mock 其它外部依赖的测试

 - `PaperQAIndexBuilder`：DB 真实、Milvus/Embedding/LLM/PDF 下载 mock
 - `EnhancedRetrievalService`：DB 真实、向量检索与生成 mock
 - `qa_router.qa` / `qa_router.qa/stream`：DB 真实、QA service 或 generation mock
 - `recommend-papers`：DB 真实、candidate recall / vector / embedding mock

 ### 15.3 不建议仅靠临时 SQLite 的测试

 以下模块即使接临时 SQLite，仍无法构成有效验证，必须搭配 fake/mock：

 - `VectorStoreService`
 - `EmbeddingService`
 - `GenerationService`
 - `ArxivSearchService`
 - `ArxivOaiSyncService`
 - `LoadingService` / `ChunkingService` 的真实重解析路径

 ---

 ## 16. 推荐实施顺序

 1. 先补 `unit/` 中的 query builder、qa_utils、tool_registry、agent utils
 2. 再补 `api/` 中的 paper / qa / user 三大主业务 router
 3. 再补 `DatabaseService` 与 `PaperQAService` 的临时 SQLite 组件测试
 4. 再补 `EnhancedRetrievalService` 与 `PaperQAIndexBuilder` 的 fake 集成测试
 5. 最后建立 smoke golden set 与主 golden set

 这样可以最快把最容易出事故的链路纳入回归。

 ---

 ## 17. 当前阶段不做的事

 - 不在本阶段新增 pytest / unittest 文件
 - 不在本阶段引入真实外部服务联调自动化
 - 不在本阶段修改生产代码以迎合测试
 - 不在本阶段重构现有目录，仅先确定目标结构与优先级

