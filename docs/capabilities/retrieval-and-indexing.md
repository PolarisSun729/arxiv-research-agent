# 检索与索引

## 责任分层

[`EnhancedRetrievalService`](../../backend/services/retrieval/enhanced_retrieval_service.py) 是调用方使用的薄门面：它解析运行选项并委托 [`RetrievalPipeline`](../../backend/services/retrieval/retrieval_pipeline.py)。检索策略、外部依赖降级和 trace 都应留在 retrieval 包内，Paper QA 和 Agent 不应自行拼接向量、BM25、rerank 或上下文预算步骤。

[`RetrievalRules`](../../backend/services/retrieval/retrieval_rules.py) 集中约束默认值、上限和规则选择；调用方传入的开关只能在规则允许的范围内生效。

## 检索流水线

`RetrievalPipeline.retrieve()` 的稳定顺序如下：

1. 解析 collection profile 和对应的 retrieval index，确定当前论文/集合有哪些可用资产。
2. 由 `QueryPlanner` 生成 intent profile、query profile、query views 和 rerank query；可选 query rewrite 在此处发生。
3. 由 `RouteRetriever` 运行向量、关键词/BM25、memory 与可选表格结构化路由，并记录各路由指标。
4. 由 `ResultFusionService` 去重并以加权 RRF 融合候选，形成稳定候选池。
5. 由 `RerankService` 对融合候选排序；失败、超时或关闭时保持融合顺序，而不是让整条 QA 失败。
6. 由上下文扩展和预算选择器生成最终 chunk 集合，预算层而非 rerank 层决定最终可进入 prompt 的上下文。
7. 仅在 debug 模式构造完整 debug payload 并导出 trace。

```mermaid
flowchart LR
    Q[question] --> P[QueryPlanner]
    P --> R[RouteRetriever]
    R --> F[Weighted RRF Fusion]
    F --> K[Rerank or fused fallback]
    K --> E[Context expansion]
    E --> B[Context budget selector]
    B --> C[final chunks]
```

## 路由、融合和 rerank

每条召回路由必须返回可追踪的候选及其来源，不能把不同路由的结果提前丢失。`ResultFusionService` 负责跨路由去重和融合；新增路由应同时补齐来源标识、融合权重和 trace 展示。

`RerankService.llm_rerank()` 是增强步骤，而非可用性前提。`RetrievalPipeline._rerank_or_passthrough()` 在 rerank 不可用时使用 `fallback_fused`，保留融合候选并将失败原因写入 debug/metrics。不要在调用方捕获 rerank 异常后返回空答案。

多 query view、生成问题索引、索引级 BM25 与 chunk 级回退均由运行选项和 `RetrievalRules` 协作控制。修改它们时应优先验证候选数量、来源分布和最终预算，而不是只观察最终模型回答。

## 上下文预算

最终上下文不是简单的 top-k。`ContextExpansionPreparer` 可以补充相邻/相关 chunk，`ContextBudgetSelector` 在字符数、软 token 限制、候选块数量和最终 top-k 之间做有界选择。任何提高召回数量或 rerank 候选数的改动，都必须验证预算层仍能控制 prompt 大小。

## 索引资产

[`PaperQAIndexBuilder`](../../backend/services/paper_qa/paper_qa_index_builder.py) 在构建时产出多类资产：

| 资产 | 构建职责 | 消费位置 |
| --- | --- | --- |
| chunk 文件 | 文档加载、切分与表格结构化 | 检索、来源展示和上下文扩展。 |
| retrieval index | `build_retrieval_indexes()` 与版本化保存 | collection profile、表格/索引级检索。 |
| sparse index | `save_sparse_index_artifact()` 与校验 | BM25/sparse 召回。 |
| embedding 文件与向量集合 | `create_chunk_embeddings()`、`index_embeddings_to_vector_store()` | 向量路由。 |
| 索引版本记录 | `PaperQAIndexStore` | 判断当前活动资产并支持重建/清理。 |

增加或删除一种资产时，必须同时更新 Builder、collection profile、检索路由、清理逻辑、版本校验和 QA 状态判断。仅写出文件但不写活动版本记录，会导致检索到旧资产或无法诊断。

## Trace 与调试

`TraceBuilder` 只在 debug 模式导出检索阶段、候选、rerank 和预算信息。普通 QA 响应不应包含本地文件路径、完整调试内容或外部提供者的原始响应。trace 下载由 QA Router 做 arxiv slug 和文件名限制，新增 trace 入口必须保留同等路径安全检查。

## 修改检查表

- 新增召回路由：检查可用资产判断、来源字段、融合、debug 和无资产时的降级。
- 调整 RRF 或 rerank：检查候选池大小、rerank 失败后是否仍可回答、最终上下文是否符合预算。
- 调整索引格式：检查旧资产识别、版本激活、清理保护和测试 fixture。
- 调整 trace：检查只有 debug 模式落盘，且对外下载不会接受路径穿越输入。

QA 上层行为见 [论文 QA](paper-qa.md)。
