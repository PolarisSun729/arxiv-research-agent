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

`PaperQAService.answer_question()` 是单篇 QA 入口。它先通过 `build_qa_context()` 组织会话、论文和用户记忆上下文，再调用检索服务、上下文包构建、答案生成和证据校验。各阶段责任如下：

| 阶段 | 组件 | 责任 |
| --- | --- | --- |
| 会话与问题规范化 | [`question_contextualizer.py`](../../backend/services/paper_qa/question_contextualizer.py)、[`session_service.py`](../../backend/services/paper_qa/session_service.py) | 把多轮问题还原为可检索的当前问题。 |
| 检索与上下文 | [`EnhancedRetrievalService`](../../backend/services/retrieval/enhanced_retrieval_service.py)、[`context_pack_builder.py`](../../backend/services/paper_qa/context_pack_builder.py) | 选择论文集合、获取候选证据并在预算内构造上下文。 |
| 答案生成 | [`answer_generator.py`](../../backend/services/paper_qa/answer_generator.py) | 基于受限上下文生成回答，不拥有检索策略。 |
| 证据校验 | [`evidence_verifier.py`](../../backend/services/paper_qa/evidence_verifier.py) | 评估回答与来源是否足以支撑结论。 |
| 持久化 | `persist_completed_turn()` 与 QA/chat stores | 原子地保留本轮消息、来源和调试快照。 |

每次回答应保留来源、上下文化后的问题和必要诊断快照，便于复现检索结论；不要只保存最终答案文本。

## QA 观察与 Agent 修复

[`qa_observation.py`](../../backend/services/paper_qa/qa_observation.py) 的 `build_qa_observation()` 将 query planning、各召回路由、融合、rerank、上下文、生成和验证结果压缩为结构化观察。它给 Agent 提供“为什么答案不足”的可消费原因，而不是让 Agent 重新解析原始日志。

[`repair_actions.py`](../../backend/services/paper_qa/repair_actions.py) 只描述可执行的 QA 修复动作。若修改观察 schema、质量阈值或修复动作，必须同步检查 Agent 的 failure/recovery 分支，保证低质量回答不会被当作成功终态。

## 会话、笔记与画像

QA 会话和消息由 `PaperChatSessionStore`、`PaperChatMessageStore` 与 `PaperQATurnStore` 持久化。阅读笔记由 `PaperNoteStore` 管理；当 `include_in_profile` 为真时，它会形成可被画像构建消费的信号。笔记更新或删除时应验证相应 profile event 是否同步失效，避免已删除信号持续影响推荐。

## 排障顺序

1. 查询 QA 索引和任务状态，确认活动版本存在且不是 stale/failed。
2. 检查 QA observation，区分无证据、召回不足、rerank 降级、上下文预算裁剪与生成/验证失败。
3. 在 debug 开启时读取受控 trace；trace 下载路径必须经过 slug 和文件名校验。
4. 再检查模型或向量服务依赖，避免将资产或状态问题误判为模型故障。

检索内部细节见 [检索与索引](retrieval-and-indexing.md)。
