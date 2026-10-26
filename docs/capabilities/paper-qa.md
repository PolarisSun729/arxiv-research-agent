# 论文 QA

## 责任边界

论文 QA 以单个 `arxiv_id` 为边界，分两种模式：详细模式（`mode=research`，缺省值）先建立该论文可检索的资产，再依据当前论文的索引、会话和用户记忆回答问题；快速模式（`mode=fast`）不建索引，把整篇论文交给模型直接作答，见下文[快速问答](#快速问答)。它不能把候选论文列表直接作为最终 QA 上下文；多论文选择属于 Agent 的目标解析阶段，完成目标解析后 QA 只处理已解析的一篇论文。

Router 位于 [`backend/routers/qa_router.py`](../../backend/routers/qa_router.py)，负责协议转换、异步任务提交、会话序列化和 trace 下载。核心业务分别位于 [`PaperQAService`](../../backend/services/paper_qa/paper_qa_service.py)、[`PaperQAIndexBuilder`](../../backend/services/paper_qa/paper_qa_index_builder.py) 与 [`IndexJobManager`](../../backend/services/paper_qa/index_job_manager.py)。

## 索引构建

```mermaid
flowchart LR
    A[论文元数据] --> B[下载或定位 PDF]
    B --> C[加载文档]
    C --> D[切分与表格结构化]
    D --> E[保存 chunk 与检索索引]
    E --> F[生成 embedding]
    F --> G[写入向量库]
    G --> H[校验并激活索引版本]
```

`PaperQAIndexBuilder.build_qa_index()` 负责上述生命周期以及失败标记和清理。执行中的阶段记录、完成后的 `mark_index_success()` 与失败后的 `mark_index_failed()` 必须使用同一条索引版本语义；不要在 Router 中拼接这些阶段。

构建产物包括 chunk、检索索引、稀疏索引、Embedding 与向量集合。文件或集合存在不等于索引可用，真源是 `PaperQAIndexStore` 的活动构建版本和状态。删除或重建资产时必须经过 Builder 的 active-artifact guard，避免误删正在被问答使用的版本。

## 异步索引任务

`IndexJobManager.submit_job()` 为常规入口创建或复用任务，后台 worker 通过 `run_next_job()` 以数据库 lease 领取并执行。QA Router 在查询任务前调用 stale 自愈逻辑，避免前端永远看到 pending/running。

维护任务状态时应遵守：

- 由 store 负责 idempotency key、占用和状态写入。
- `retryable`、`stale`、`heartbeat_at`、`previous_job_id` 和 `recovery_action` 是恢复语义的一部分，不能只显示一个字符串状态。
- 同步构建只用于管理和诊断场景；不能把它作为普通请求的默认路径。

## 问答链路

`/qa` 与 `/qa/stream` 按请求的 `mode` 分流：`research` 走下述研究引擎，`fast` 走[快速问答](#快速问答)。两条路径共用会话存储和 `done` 字段契约，前端不需要按模式区分解析逻辑。

`PaperQAService.answer_question()` 是详细模式的入口。生产组合根始终注入 [`PaperEvidenceResearchService`](../../backend/services/paper_evidence_research/module.py)，同步和流式入口共用“准备会话 → 研究图 → 持久化结果”的执行流；详细模式内研究引擎是唯一路径，不再保留一次检索、一次生成的旧编排。

| 阶段 | 组件 | 责任 |
| --- | --- | --- |
| 会话与问题规范化 | `_prepare_research_context()`、[`session_service.py`](../../backend/services/paper_qa/session_service.py) | 通过 `load_conversation_state()` 读取真实会话；消歧失败退回原问题。无状态运行使用内部临时 ID，不冒充已保存会话。 |
| 取证决策与预算 | [`研究图`](../../backend/services/paper_evidence_research/graph.py)、[`action_gate.py`](../../backend/services/paper_evidence_research/action_gate.py) | LLM 提议动作，确定性门禁裁决预算、需求状态和完成条件。 |
| 检索与上下文 | [`NeedOrchestratedRetriever`](../../backend/services/paper_evidence_research/dependencies/need_orchestrated_retriever.py)、[`context_pack.py`](../../backend/services/paper_evidence_research/context_pack.py) | 解析单篇活动索引，经 `RetrievalPipeline.retrieve()` 取候选并在预算内选择证据。末轮关闭查询改写与 LLM rerank。 |
| 答案生成 | [`ResearchDraftGenerator`](../../backend/services/paper_evidence_research/dependencies/production_adapters.py) | 适配 `AnswerGenerator` 的真实请求接口，只消费选定证据。 |
| 主张提取与校验 | [`RuleClaimExtractor`](../../backend/services/paper_evidence_research/dependencies/rule_claim_extractor.py)、[`ResearchClaimVerifier`](../../backend/services/paper_evidence_research/dependencies/production_adapters.py) | 从可见答案提取含无引用句在内的事实主张，独立验证引用与支持关系；校验故障是执行失败。 |
| 持久化 | `persist_completed_turn()` 与 QA/chat stores | 原子地保留本轮消息、来源和调试快照。 |

每次回答应保留来源、上下文化后的问题和必要诊断快照，便于复现检索结论；不要只保存最终答案文本。

研究正常结束只有三种 `outcome`：`completed` 表示全部核心需求得到支持；`partial` 只保留可靠且有支持的部分核心回答；`abstained` 表示正常取证后仍无法支持核心结论。检索故障按需求保存在业务状态中；同一需求成功重试会清除故障。故障仍阻断取证且无法保留可靠有限回答时，返回运行错误，不能算作拒答。审计轨迹不驱动终态。

详细模式下 `/qa/stream` 发送 `meta`、`progress`（retrieval/draft/verification/completed）及唯一的 `done` 或 `error`。研究答案经过校验后由 `done.answer` 一次交付，不发送增量 `delta` 事件。`done` 保留会话、来源、`outcome`、`research_summary` 和 `usage`；`error` 使用 `AppError.to_payload()`，不返回底层异常文本。流式统计上下文在每次 `next()` 内绑定，避免工作线程切换导致 ContextVar 恢复失败。

## 快速问答

快速模式面向“读一篇论文、马上问”的场景：不切块、不建向量，整篇论文作为一次多模态输入交给模型，首字时间取决于模型而不是检索链路。代价是引用不经过逐句核对，前端在回答下方标注“快速模式 · 未逐句核对”。

| 阶段 | 组件 | 责任 |
| --- | --- | --- |
| 全文解析 | [`fulltext_parser.py`](../../backend/services/paper_fast_qa/fulltext_parser.py) | 用 pymupdf4llm（版面分析来自 pymupdf-layout）把 PDF 转成按阅读顺序排列的块：正文、标题、表格 Markdown，图片、公式和解析不出的表格截图保存为图片。块 ID 为 `p{页}-{序号}`，图片为 `fig-{序号}`。不依赖 arXiv HTML，任何文字版 PDF 都适用；扫描版没有可提取文字时报 `empty`。 |
| 缓存与排队 | [`PaperFulltextService`](../../backend/services/paper_fast_qa/fulltext_service.py) | 解析结果按 PDF 内容哈希与解析器版本缓存在 `backend/03-fulltext-cache/`，原子写入、自动清理旧版本。进程内队列限制并发解析数；提问触发的解析排在预解析前面，同一篇论文的请求合并；预解析队列满时返回 `busy`；失败后进入冷却期，避免反复解析坏文件。解析在隔离子进程中执行并有超时。 |
| 生成 | [`PaperFastQAService`](../../backend/services/paper_fast_qa/fast_answer_service.py) | 输入顺序固定为“指令 → 全文（图片插在原位）→ 最近几轮对话 → 本轮问题”，论文部分在多轮追问间保持不变，便于命中模型的上下文缓存。请求体超过百炼上限时先把图片缩小一次，仍超限或模型报输入过长时返回 `paper_too_long_for_fast_qa`，提示改用详细模式。 |
| 引用整理与持久化 | 同上 | 去掉指向不存在块的引用并记入调试快照；被引用的块作为来源返回，图片块带 `asset_url`，复用 `evidence-assets` 接口取图。持久化走与详细模式相同的 `persist_completed_turn()`。 |

详情页打开时调用 `POST /fast-qa/prepare` 预解析，`GET /fast-qa/status` 查询进度（`ready`/`queued`/`parsing`/`failed`/`not_parsed`）。提问时全文尚未就绪会在等待上限内继续等，期间发送 `progress(stage=parsing)`；超时或解析失败返回 `paper_fulltext_unavailable`。并发、队列、超时、等待、冷却、渲染 DPI 和历史轮数都在 `PAPER_FAST_QA_CONFIG` 中配置。

快速模式的流事件为 `meta`、`progress`（parsing/generating）、逐段 `delta` 和唯一的 `done` 或 `error`。调试快照的 `engine` 为 `fast_fulltext`，记录全文规模、请求体大小、token 用量（含缓存命中）、等待全文、首字和生成耗时。

## QA 观察与 Agent 修复

[`qa_observation.py`](../../backend/services/paper_qa/qa_observation.py) 的 `build_research_qa_observation()` 将研究三态映射为 `grounded`、`warning`、`insufficient_evidence`，保留终止原因和未解决主题；`build_qa_observation()` 是它内部复用的来源与阶段诊断构建函数。

研究图已拥有有界修复预算，外层 Agent 不应因为 `partial` 或正常 `abstained` 再重跑整次 QA。正常拒答没有引用属于预期；有限回答即使有引用也不能投影成完整回答。技术错误继续进入原有 failure/recovery 边界。[`repair_actions.py`](../../backend/services/paper_qa/repair_actions.py) 只为 Agent 的 Observer 与工具节点提供修复动作常量，不能反向覆盖研究终态。

生产与离线 runner 使用同一评测记录契约，记录完整答案、运行成败、研究轨迹和实际调用统计。格式、重新评分和基线口径见 [生成效果评测](evaluation.md)。

## 会话与画像

QA 会话和消息由 `PaperChatSessionStore`、`PaperChatMessageStore` 与 `PaperQATurnStore` 持久化。用户提问会写入 `qa_asked` profile event，作为画像构建的近期信号。

## 排障顺序

1. 查询 QA 索引和任务状态，确认活动版本存在且不是 stale/failed。
2. 检查 QA observation，区分无证据、召回不足、rerank 降级、上下文预算裁剪与生成/验证失败。
3. 在 debug 开启时读取受控 trace；trace 下载路径必须经过 slug 和文件名校验。
4. 再检查模型或向量服务依赖，避免将资产或状态问题误判为模型故障。

检索内部细节见 [检索与索引](retrieval-and-indexing.md)。
