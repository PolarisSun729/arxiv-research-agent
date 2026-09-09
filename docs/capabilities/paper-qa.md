# 论文 QA

## 责任边界

论文 QA 以单个 `arxiv_id` 为边界：先建立该论文可检索的资产，再依据当前论文的索引、会话和用户记忆回答问题。它不能把候选论文列表直接作为最终 QA 上下文；多论文选择属于 Agent 的目标解析阶段，完成目标解析后 QA 只处理已解析的一篇论文。

Router 位于 [`backend/routers/qa_router.py`](../../backend/routers/qa_router.py)，负责协议转换、异步任务提交、会话/笔记序列化和 trace 下载。核心业务分别位于 [`PaperQAService`](../../backend/services/paper_qa/paper_qa_service.py)、[`PaperQAIndexBuilder`](../../backend/services/paper_qa/paper_qa_index_builder.py) 与 [`IndexJobManager`](../../backend/services/paper_qa/index_job_manager.py)。

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

`PaperQAIndexBuilder.build_qa_index()` 负责上述生命周期以及失败标记和清理。构建前的 `prepare_rebuild()`、执行中的阶段记录、完成后的 `mark_index_success()` 与失败后的 `mark_index_failed()` 必须使用同一条索引版本语义；不要在 Router 中拼接这些阶段。

构建产物包括 chunk、检索索引、稀疏索引、Embedding 与向量集合。文件或集合存在不等于索引可用，真源是 `PaperQAIndexStore` 的活动构建版本和状态。删除或重建资产时必须经过 Builder 的 active-artifact guard，避免误删正在被问答使用的版本。

## 异步索引任务

`IndexJobManager.submit_job()` 为常规入口创建或复用任务，`run_job()` 负责占用和执行。QA Router 在查询任务前调用 stale 自愈逻辑，避免前端永远看到 pending/running。

维护任务状态时应遵守：

- 由 store 负责 idempotency key、占用和状态写入。
- `retryable`、`stale`、`heartbeat_at`、`previous_job_id` 和 `recovery_action` 是恢复语义的一部分，不能只显示一个字符串状态。
- 同步构建只用于管理和诊断场景；不能把它作为普通请求的默认路径。

## 问答链路

`PaperQAService.answer_question()` 是单篇 QA 入口。生产组合根始终注入 [`PaperEvidenceResearchService`](../../backend/services/paper_evidence_research/module.py)，同步和流式入口共用“准备会话 → 研究图 → 持久化结果”的执行流。旧组件仍供兼容调用和测试使用，生产入口不再使用旧的一次检索、一次生成编排。

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

`/qa/stream` 发送 `meta`、`progress`（retrieval/draft/verification/completed）及唯一的 `done` 或 `error`。研究答案经过校验后由 `done.answer` 一次交付，客户端不能假设一定收到 `delta`。`done` 保留会话、来源、`outcome`、`research_summary` 和 `usage`；`error` 使用 `AppError.to_payload()`，不返回底层异常文本。流式统计上下文在每次 `next()` 内绑定，避免工作线程切换导致 ContextVar 恢复失败。

## QA 观察与 Agent 修复

[`qa_observation.py`](../../backend/services/paper_qa/qa_observation.py) 的 `build_research_qa_observation()` 将研究三态映射为 `grounded`、`warning`、`insufficient_evidence`，保留终止原因和未解决主题。`build_qa_observation()` 仍用于兼容链的阶段诊断。

研究图已拥有有界修复预算，外层 Agent 不应因为 `partial` 或正常 `abstained` 再重跑整次 QA。正常拒答没有引用属于预期；有限回答即使有引用也不能投影成完整回答。技术错误继续进入原有 failure/recovery 边界。兼容路径的 [`repair_actions.py`](../../backend/services/paper_qa/repair_actions.py) 保留原语义，不能反向覆盖研究终态。

生产与离线 runner 使用同一评测记录契约，记录完整答案、运行成败、研究轨迹和实际调用统计。格式、重新评分和基线口径见 [生成效果评测](evaluation.md)。

## 会话、笔记与画像

QA 会话和消息由 `PaperChatSessionStore`、`PaperChatMessageStore` 与 `PaperQATurnStore` 持久化。阅读笔记由 `PaperNoteStore` 管理；当 `include_in_profile` 为真时，它会形成可被画像构建消费的信号。笔记更新或删除时应验证相应 profile event 是否同步失效，避免已删除信号持续影响推荐。

## 排障顺序

1. 查询 QA 索引和任务状态，确认活动版本存在且不是 stale/failed。
2. 检查 QA observation，区分无证据、召回不足、rerank 降级、上下文预算裁剪与生成/验证失败。
3. 在 debug 开启时读取受控 trace；trace 下载路径必须经过 slug 和文件名校验。
4. 再检查模型或向量服务依赖，避免将资产或状态问题误判为模型故障。

检索内部细节见 [检索与索引](retrieval-and-indexing.md)。
