# 001 定量评测体系与论文证据研究引擎扶正

- 状态: 生产接线与评测修复已实现；真实 golden 基线等待人工标注
- 日期: 2026-09-04
- 来源: 2026-09-04 grilling 会话共识 + 当日代码现状核对
- 关联词汇: `backend/CONTEXT.md`（"自修复增益"、"评估单位"已定义）

当前维护口径见 [论文 QA](../capabilities/paper-qa.md) 和 [生成效果评测](../capabilities/evaluation.md)。生产同步、流式与 runner 已共用正式研究依赖；旧组件仍保留给兼容入口与既有调用方，下面的整段删除计划尚不代表已完成。现有 12 条 smoke 仍缺三项人工标注，不能宣称已有真实效果基线。

## 1. 背景与目标

系统缺少定量评价指标。经讨论确认：指标体系无法建立在当前生产 QA 路径上，因为该路径把
"完整回答 / 有限回答 / 证据拒答"三种终态抹平成一段文本，没有结构化 outcome、修复轮次、
引用-主张绑定可以落盘。而 `paper_evidence_research` 子图（commit 008a89b）已经具备这些
语义，但它是代码孤岛（无生产调用方），6 个依赖没有真实实现。

因此本 PRD 覆盖两件事，且顺序不可颠倒：

1. **重构**：把 `paper_evidence_research` 扶正为唯一生产 QA 引擎，旧 QA 编排链迁移后删除。
2. **评测**：在新引擎的结构化输出上建确定性指标体系 + 离线评测运行器 + 回归报告。

顺序是"先重构、再评估"，不做 A/B 对比——生产路径终态语义本身是错的，拿它当基线只会把
错误固化成标准。重构完成后第一批 golden 跑分即"新基准"。

## 2. 共识决策记录（已拍板，不再讨论）

| # | 决策 | 内容 |
|---|------|------|
| D1 | 数据 | 用户手工标注 20-30 条 golden（格式 = 现有 jsonl + 新增 `expected_chunk_ids` / `answerable` / `gold_answer` 三字段）；放量后走线上 query 回流 + LLM 生成抽检，本期不做 |
| D2 | 埋点 | 补 eval record（outcome、citations、效率、app_version），终态不抹平 |
| D3 | 架构 | `paper_evidence_research` 扶正为唯一生产 QA 引擎；旧 QA 编排链迁移完成后整段删除；arxiv_search_agent 的搜索/推荐/画像等其他意图不动 |
| D4 | 依赖 | question_analyzer / decision_policy 采用 LLM 优先 + 规则兜底；首期 claim_extractor 使用独立规则提取可见主张。预算护栏在图骨架，LLM 输出过 schema；独立 claim_verifier 故障必须记运行错误，不能规则伪造 supported |
| D5 | 检索集成 | 研究循环 3 次检索预算，末次降级轻量模式；候选池去重 |
| D6 | 观测 | 检索候选（query + chunk_id + rank）走 trace 事件流（方案 A），`PaperEvidenceResearchResult` 对外契约不变 |
| D7 | 评测运行 | 离线脚本跑 golden，报告落盘 `backend/06-evaluation-result/`，文件名带时间戳 + commit hash；每 case n=3 取中位数并记录离散度；关键指标跌 5pt 报告头部标红，暂不进 CI fail |
| D8 | LLM 不确定性 | 不锁 temperature、不缓存固定输出（那是在测一个线上不存在的版本）；用 n=3 中位数 + 离散度上报消解 |

## 3. 设计输入核对（2026-09-04 历史快照）

与讨论时的认知有 3 处偏差，已按当前代码修正：

1. **旧 QA 链的形态变了**。`agents/` 目录已重组为 `agents/arxiv_search_agent/`（planner /
   recovery_chooser 现在服务于搜索、推荐、画像等意图）。生产 QA 路径实际是
   `routers/qa_router.py:726 POST /qa`（及 `:784 /qa/stream`）→
   `services/paper_qa/paper_qa_service.py:555 PaperQAService.answer_question()`：
   build_qa_context（EnhancedRetrievalService 检索）→ AnswerGenerator 生成 →
   EvidenceVerifier 校验 + apply_answer_guardrail → persist_completed_turn。
   **刀刃对象是这条编排链**，不是 planner/recovery。
2. **`paper_qa` 服务并非裸奔**。它已有 EvidenceVerifier、citation_contract、citation_boundary、
   qa_observation。缺的是：结构化三态 outcome、needs 账本、修复轮次落盘。扶正方案不变，
   且 claim_verifier 可以复用 evidence_verifier + citation_contract。
3. **`05-generation-results/` 不是空的**（有 7-8 月旧生成 dump），但 `06-evaluation-result/`
   确实是空的。评测产物落 `06-evaluation-result/` 不变。

其余确认：

- 研究图骨架完整：`prepare → decide → validate → search|draft|finalize|abstain|terminal`，
  draft 后 `extract_claims → verify_claims → project_coverage → decide` 回环；
  `contracts.py` 有 `ResearchLimits`（max_retrievals=3, max_draft_attempts=2, max_no_progress=1,
  max_invalid_actions=2）、三态 outcome、`ResearchSummary`（retrieval_count / draft_attempt_count /
  verification_count / supported_claim_count / citation_repair_count / termination_reason /
  unresolved_topics）；`state.trace_events` 贯穿全程。
- 6 个依赖只有 `tests/unit/services/paper_evidence_research/test_research_service.py` 里的
  scripted 假人，无任何生产实现。接口约定（duck typing）：
  - `question_analyzer.analyze(request) -> {research_question, evidence_needs}`（≥1 个 core need）
  - `decision_policy.decide(ResearchDecisionContext) -> action dict`
  - `retriever.retrieve(SearchPaperAction, state) -> {candidates: [EvidenceCandidate…], status}`
  - `draft_generator.generate(DraftGenerationRequest) -> {draft_id, answer, declared_claims, used_candidate_ids}`
  - `claim_extractor.extract(ClaimExtractionRequest) -> {claims: [AnswerClaim…]}`
  - `claim_verifier.verify(ClaimVerificationRequest) -> {assessments: [ClaimAssessment…]}`
- **方案 A 尚未落地**：`graph.py:189` 的 `retrieval_completed` trace 事件只有计数
  （new/duplicate_candidate_count），没有候选 chunk_id + rank。Phase 2 补。
- `RetrievalPipeline.retrieve(user_query, collection_name, paper_context, options)`
  （`services/retrieval/retrieval_pipeline.py:55`）存在但生产未接（docstring 自称
  "PaperQAService 未来可直接调用的检索入口"），生产仍走 `EnhancedRetrievalService`。
  RequestTrace 落盘在 `temp/backend-request-traces/{run_id}.json`。
- golden 集 `backend/tests/golden/data/smoke_golden_set.jsonl` 现有 12 条，schema 缺
  D1 的三个字段。
- `tools/paper_qa_tools.py:126 answer_paper_question` 与
  `agents/arxiv_search_agent/tool_adapters/paper_qa.py` 消费 `PaperQAService`，
  迁移时要同步透传 outcome。
- 工作树有与本 PRD 无关的未提交改动（arxiv_search_agent / recommendation / 前端，
  分支 codex/paper-target-resolution）。**开工前先提交或 stash 这批改动**，避免混入。

## 4. 实施计划（4 个独立提交，严格按序）

### Phase 1 — 三器官 LLM 实现 + 规则兜底（不接线，纯新增）

新目录 `backend/services/paper_evidence_research/dependencies/`：

| 新文件 | 职责 | LLM | 兜底 |
|--------|------|-----|------|
| `llm_question_analyzer.py` | 问题 → research_question + evidence_needs（≥1 core，结构化输出） | GenerationService JSON 输出 + pydantic 校验 | `template_question_analyzer.py`：按 main_intent 查表（method_flow/experiment_setup/… 固定 needs 模板） |
| `llm_decision_policy.py` | 从 `ResearchDecisionContext` 提议一个语义动作 | LLM JSON 输出 → `parse_research_action` | `rule_decision_policy.py`：纯查表状态机（open core need 且有检索额度 → search；额度尽 → draft；有 draft 且未验证 → finalize 前置校验……护栏在图层） |
| `rule_claim_extractor.py` | 草稿 → 原子主张 | 无（首期直接用规则版：切句 + 剥元话语） | — |
| `research_dependency_factory.py` | 组装 6 依赖 + checkpointer，产出 `PaperEvidenceResearchService` 实例 | — | — |

铁律落地方式：

- 预算护栏不动：`ResearchLimits` 仍在 request 里，`action_gate.py` / `_route_after_validation`
  继续裁决；LLM 决策只能在预算内选 search/draft/finalize/abstain。
- 降级即留痕：任何 LLM 输出 schema 不合法 → 记 `action_rejection_code` / 降级 trace 事件 →
  用规则版结果继续，图永不卡死。
- `claim_extractor` 首期只用规则版（切粗只影响校验粒度，不引入误判），LLM 版留接口后续替换。

单测：`backend/tests/unit/services/paper_evidence_research/test_llm_question_analyzer.py`、
`test_llm_decision_policy.py`（mock GenerationService：合法 JSON / 非法 JSON / 超时三类路径，
非法必须降级且产生 trace 事件）、`test_rule_claim_extractor.py`、
`test_research_dependency_factory.py`。

**验收**：用 scripted 假人换成 LLM 器官后，现有 `test_research_service.py` 全套图行为测试
仍绿（只换依赖不改图）。

### Phase 2 — NeedOrchestratedRetriever 适配器 + trace 分桶 + 候选入 trace（方案 A）

- 新文件 `backend/services/paper_evidence_research/dependencies/need_orchestrated_retriever.py`：
  - 包装 `RetrievalPipeline.retrieve`；`SearchPaperAction.target_need_id` + need.description
    + research_question → 检索 query；collection 取该 arxiv_id 的 QA 索引 collection。
  - **次数治理**：图在调用前已自增，`state.retrieval_count >= limits.max_retrievals` 的那一轮传轻量
    `RetrievalOptions`（跳过 query rewrite 与 LLM rerank），前几轮走全管线。
  - **候选映射**：检索结果 chunk → `EvidenceCandidate`（candidate_id=chunk_id，
    content/section_path/page_number/chunk_type 对齐现有 QA 索引 chunk 字段）。
  - **候选池去重**：回传前过滤 `state.evidence_candidates` 已持有的 chunk_id
    （`evidence_pool.merge_candidates` 的 duplicate_count 仍是第二道闸）。
- `graph.py:_search_node`：`retrieval_completed` trace 事件增加
  `candidates=[{"candidate_id", "rank", "score"}]` + `query` 字段（D6，对外契约不动）。
- trace 分桶：研究全程复用 `RequestTrace(run_id=research_run_id)`，检索适配器每次
  retrieve 向 trace 追加 `retrieval_round` 事件（含各阶段候选摘要），文件名即
  research_run_id，天然按 run 分桶；QA 层不再按 arxiv_id 覆盖写。

单测：轻量模式触发条件、chunk→candidate 映射、去重过滤、trace 事件 schema；
集成冒烟一次（2401.00001 真实索引）。

**验收**：`research()` 一次调用的 trace 文件里能看到每一轮检索的 query 与带 rank 的候选
chunk_id；同一 chunk 不会在两轮中重复进入 evidence_pool。

### Phase 3 — 生产切换 + eval record 落盘 + 删旧链

切换（`paper_qa_service.py`）：

- `answer_question()` 编排体替换为：会话/上下文准备（保留 session_service、
  question_contextualizer 产出的 conversation_snapshot）→
  `PaperEvidenceResearchService.research(PaperEvidenceResearchRequest)` →
  结果映射 + `persist_completed_turn`。
- result 新增 `"outcome": "completed|partial|abstained"`、`"research_summary"`、
  `"citations"`（来自 VerifiedCitation）；`sources` 字段由 citations 投影生成，保持前端
  现有消费形态；`qa_observation` 兼容层保留（outcome 映射进 answer_quality）。
- `/qa/stream`：最小实现——同样调 research，按阶段 trace 事件推 SSE
  （research_started / draft_created / research_completed），行为向后兼容。
- 同步更新 `tools/paper_qa_tools.py:answer_paper_question` 与 agent tool adapter 透传 outcome。

eval record（D2）：

- 新目录 `backend/services/evaluation/`，`eval_record.py`：
  在 `persist_completed_turn` 旁边写一条 JSON 到
  `backend/06-evaluation-result/records/YYYY-MM-DD/{research_run_id}.json`：
  `record_id, timestamp, app_version(git commit hash), user_id, session_id, turn_id,
  run_status, paper_context{arxiv_id, indexes}, configuration{research_limits, engine},
  query{raw, rewritten, main_intent}, outcome, answer, termination_reason,
  citations[source_id, content, claim_ids], research_summary{…}, efficiency{latency_ms,
  llm_calls, input_tokens, output_tokens, total_tokens, retrieval_count, draft_attempt_count}, trace_ref, trace_events, error`。
  - 运行失败必须 `run_status=error, outcome=null`；拒答属于正常研究终态。
  - 写失败只告警，不影响问答主链路（与 `_write_qa_trace` 同策略）。
- llm_calls/tokens：在 `GenerationService` 的 SDK 边界与远程 rerank 的 HTTP 边界计数，包含失败请求；未报告用量记 null。线上和 runner 使用同一请求级统计上下文。

删除（切换稳定后同一提交或紧随提交）：

- `paper_qa/answer_generator.py` 编排逻辑、`paper_qa/repair_actions.py` 及其单测
  （先核对 `tools/paper_qa_tools.py` 对 repair 策略映射的依赖，若 agent 工具仍需，
  把映射搬到工具侧后删服务侧实现）。
- `EnhancedRetrievalService` 若仅剩 QA 调用方则一并退市（RetrievalPipeline 成为唯一入口）；
  若搜索意图仍在用则保留，只解绑 QA。
- 旧 QA 编排相关单测改写或删除；`test_paper_qa_service.py` 按新编排重写关键路径。

**验收**：`POST /qa` 返回三态 outcome；拒答类问题返回 `abstained` 而非编造答案；
eval record 每轮落盘；全量测试绿；旧编排代码无残留引用。

### Phase 4 — 评测运行器 + 指标模块 + 回归报告

新目录 `backend/services/evaluation/`：

| 文件 | 内容 |
|------|------|
| `metrics_retrieval.py` | Recall@k / Hit@k / MRR：golden `expected_chunk_ids` × trace 事件流里的各轮候选（D6 方案 A 的数据） |
| `metrics_generation.py` | ①要点覆盖率（0-1 连续值，中英文别名归一——把现 smoke 测试里硬编码的 alias 表抽成数据文件）；②引用忠诚度：每个 citation 的 claim 是否 verdict=supported 且 chunk 真实存在；③三态正确率：`answerable` × `outcome` 交叉矩阵（该拒没拒 / 不该拒乱拒分开计）；④自修复增益：按 CONTEXT.md 定义，用 draft_attempt 间 supported_claim_count 差值计算 |
| `metrics_report.py` | 分桶报告（按 difficulty / main_intent），与上一份报告 diff，Recall@5 / 三态正确率 / 引用忠诚度跌幅 >5pt 在报告头部标红（D7） |
| `golden_runner.py` | CLI：`python -m services.evaluation.golden_runner --cases <jsonl> --repeat 3`；每 case 跑 3 次取中位数 + 记录离散度（离散度本身是指标：同 case 三次 outcome 不一致要显式上报）；报告落盘 `06-evaluation-result/reports/{ts}_{commit}.json` |
| `contracts.py` / `scoring.py` / `configuration.py` | 统一记录与重评分；运行失败不丢分母；基线比较校验实际预算、模型、索引与重复次数，版本缺失时禁止自动 diff |

pytest 使用真实请求/结果模型和研究图，仅替换外部 I/O，覆盖指标、runner、会话、SSE、失败统计与重新评分；完整真实 golden 评测不进 CI。

指标 → 数据源对照（Q9 结论，全部可在新引擎上直接计算）：

| 指标 | 来源 | 状态 |
|------|------|------|
| 三态正确率 / 终止原因分布 | 顶层 run_status/outcome + research_summary.termination_reason / unresolved_topics | 已接通；需人工 answerable |
| 自修复增益 | 各版 evidence_coverage_projected 的 supported_claim_count 与核心覆盖差值 | 已接通；同时记录有害修复 |
| 引用忠诚度 | citations[].source_id + claim_ids + 最终版 claims/assessments 的 verdict 和引用绑定 | 已接通；无引用不自动满分 |
| 效率 / 成本 | research_summary 计数 + 请求级真实 provider 调用与 usage | 已接通；缺用量为 null |
| Recall@k / MRR | event_type=retrieval_completed 的去重前排名候选 | 已接通；需人工 expected_chunk_ids |

**验收**：12 条现有 golden + 新字段能跑出完整报告；重复跑两次报告可 diff；
 golden 标注完成后输出第一份"新基准"报告。

补充口径：报告同时显示中位数和逐次均值，以及失败率、有效/缺测样本数。技术失败在适用指标中计 0；未标注指标为 null。只有标注和轨迹完整、配置稳定、每题至少三次且无运行失败的报告才是可用基线；自动 diff 还要求与上次基线的数据、指标、配置指纹一致。

## 5. 数据集任务（并行，人工，不阻塞 Phase 1-2）

- 用户按 D1 标注 20-30 条：围绕 2-3 篇已入库论文（2401.00001 起步），沿用现有 schema
  追加三字段；`answerable: false` 的负例占 ≥20%。
- 节奏：每次精读一篇新论文顺手标 5-10 条；放量后接线上 query 回流（另立 PRD）。
- `expected_chunk_ids` 标注需要 Phase 2 的 trace 里有 chunk_id——两者天然对齐。

## 6. 明确不做（本期边界）

- arxiv_search_agent 的搜索 / 推荐 / 画像 / 澄清意图不改。
- LLM-as-Judge（忠实度/相关性打分）不做，留作后续可选层。
- 前端 UI 对 outcome 的展示改造不做（API 先透出，前端另立任务）。
- 扩展评测集（LLM 生成 + 抽检）、线上行为信号（点击/反馈率）不做。

## 7. 风险与对策

| 风险 | 对策 |
|------|------|
| LLM 器官不稳定导致研究图行为抖动 | 规则兜底 + schema 校验降级（D4 铁律）；n=3 中位数消解评测噪声 |
| 3 次检索 × 全管线拖高延迟/成本 | 末次轻量模式（D5）；后续按 eval record 的效率数据再调 |
| 迁移期两套语义并存造成回归盲区 | Phase 3 一次性切换并删除，不设双跑期；切换提交里用 golden 集人工抽验 5-10 条 |
| `repair_actions` 被 agent 工具消费，删除牵连 | Phase 3 先核依赖再删，必要时映射逻辑搬到工具侧 |
| golden 标注进度慢，评测长期空转 | Phase 4 完成即可用现有 12 条 smoke 跑管线；标注与开发并行 |
| 工作树已有未提交改动 | 开工前提交/stash，本 PRD 分支不与其混搭 |
