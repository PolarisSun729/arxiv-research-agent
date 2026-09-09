# 生成效果评测

## 责任与入口

[`services/evaluation`](../../backend/services/evaluation/) 评估单篇论文研究结果的要点覆盖、证据支持、拒答、修复增益与成本。生产 QA 和 [`golden_runner.py`](../../backend/services/evaluation/golden_runner.py) 使用组合根 `get_paper_evidence_research_service()` 装配的同一研究引擎；评分函数只读取记录与标注，不调用模型。

研究图的独立主张校验属于生产能力。评测中的引用忠诚度是对该校验关系的确定性复核，不是额外的 LLM-as-Judge，也不能代替人工校准校验器的准确性。

## Golden 标注

JSONL 每行一个 [`GoldenCase`](../../backend/services/evaluation/contracts.py)。`case_id` 必须唯一，`arxiv_id` 指向已建立 QA 索引的论文，`question` 非空。以下示意中的文本和 ID 必须替换为人工核实的内容：

```json
{"case_id":"method-01","arxiv_id":"论文ID","question":"该方法如何决定是否检索？","difficulty":"medium","main_intent":"method_flow","language":"zh","expected_answer_points":["人工确认的关键要点"],"expected_chunk_ids":["真实证据source_id"],"answerable":true,"gold_answer":"人工核对论文后的参考答案"}
```

| 字段 | 标注含义 |
| --- | --- |
| `expected_answer_points` | 需要覆盖的完整要点或短语。可回答题至少一项；共享中英文别名保存在 [`answer_point_aliases.json`](../../backend/services/evaluation/answer_point_aliases.json)。 |
| `expected_chunk_ids` | 当前固定索引中的相关证据 ID，应与答案的 `source_id` 和检索轨迹的 `candidate_id` 对齐。缺字段表示未标注；空数组表示人工确认没有可用相关证据。 |
| `answerable` | 严格布尔值。`false` 表示论文自身不足以回答；索引或服务故障不能标成不可回答。 |
| `gold_answer` | 人工参考答案；可回答题不能为空。拒答题记录论文缺少哪些核心证据。当前不做参考答案语义相似度评分。 |
| `difficulty` / `main_intent` | 报告分桶标签，缺失时为 `unknown`。 |

现有 [`smoke_golden_set.jsonl`](../../backend/tests/golden/data/smoke_golden_set.jsonl) 的 12 条样本缺少 `expected_chunk_ids`、`answerable`、`gold_answer`，只能验证管线，不能建立真实召回和拒答基线。建议人工标注 20–30 条、覆盖 2–3 篇论文，至少 20% 为不可回答题。不得根据当前模型答案自动补齐“真值”。

## 记录与轨迹

[`EvaluationRecord`](../../backend/services/evaluation/contracts.py) 是线上落盘、离线运行及重新评分共用的顶层结构，版本为 `paper_evidence_eval_v1`：

- `run_status` 为 `success` 或 `error`；只有成功运行有 `outcome`。失败必须有 `error`，且 `outcome=null`。
- `answer`、`citations`、`research_summary`、`termination_reason` 都在顶层；终止原因来自真实 `result.research_summary`，没有 `record.result.answer` 包装层。
- `trace_events` 使用 `event_type`。`retrieval_completed` 保存去重前排名的候选 ID；`claim_verification_completed` 保存该版主张与 verdict；`evidence_coverage_projected` 保存每版可靠支持和核心覆盖。
- `configuration` 保存实际请求预算和模型、检索配置；普通轮次和末轮的有效参数复用管线自身解析器，包含 top-k 裁剪、上下文预算及开关覆盖。`paper_context.indexes` 保存本次实际访问的 collection、活动 build/version、embedding 模型和可用的稀疏索引源哈希。运行 ID、时间和凭据不进入配置指纹。
- `efficiency` 保存延迟、检索/草稿轮次及真实 LLM 请求统计；`trace_ref` 关联研究运行。

记录默认在 `backend/06-evaluation-result/records/YYYY-MM-DD/`，报告默认在 `backend/06-evaluation-result/reports/`，不依赖启动目录。运行 ID 会归一化为安全文件名。评测记录或轨迹写失败只记录服务端警告，不破坏已完成的问答；会话持久化失败则属于运行失败。

完整轨迹是私有审计产物。公开 SSE 只返回阶段计数和研究摘要，不能把完整校验数据当作普通进度事件。

## 指标口径

| 指标 | 计算与适用条件 |
| --- | --- |
| `recall@k` / `hit@k` | 对各轮原始排名的 top-k 取并集，与人工相关证据集合比较；同一 ID 多次命中只计一次。不是把多轮列表拼接后截断。没有相关证据标注时为 `null`。 |
| `mrr` | 相关证据在任一轮的最佳真实 rank 的倒数；未命中为 0。保留原 rank，不在过滤候选后重新编号。 |
| `key_coverage_rate` | 被答案完整短语或共享别名命中的要点比例；英文使用词界，中文使用完整短语。它是词面覆盖，不证明语义正确或证据充分。 |
| `citation_fidelity` | 每个 `source_id` 真实存在，绑定的所有主张在最终校验版均为 `supported`，支持证据与原引用集合一致，且不依赖尚未验证的纯视觉内容，才算忠实引用。完整/有限回答没有引用得 0；正常拒答不适用，记 `null`。 |
| `three_state_accuracy` | `answerable=true` 时 completed/partial 计正确拒答决策，`false` 时仅 abstained 计正确；执行错误为 0。二值可回答性标签不能证明 partial 已完整覆盖，因此报告同时保留三态交叉矩阵。 |
| `self_repair_gain` | 最后一版减初稿的已验证支持主张数，可为负；没有校验快照时为 `null`，只有一稿时为 0。不是修复次数占比。 |
| `core_coverage_gain` / `harmful_repair` | 核心需求满足比例的前后差值；发生多稿修复且可靠支持或核心覆盖下降时标记有害修复。须结合需求账本变化和成本解读。 |
| 延迟与成本 | `latency_ms` 为墙钟耗时；`llm_calls` 包含实际生成、规划、验证、远程 rerank 请求及失败请求。provider 未报告 token 时为 `null`；只报告总量时不捏造输入/输出拆分。 |

模型调用统计在 provider 边界计数，以请求级 ContextVar 隔离并发，检索线程复制调用上下文。`observed_*_tokens` 仅表示已返回的部分用量，不能当成整轮总成本。这里的 token/call 统计不包含 embedding、存储、PDF 处理费用，也不换算货币。

## 运行与重新评分

以下命令从 `backend/` 执行。数据校验完全离线，不初始化模型：

```powershell
python -m services.evaluation.golden_runner --cases tests/golden/data/smoke_golden_set.jsonl --validate-only
python -m services.evaluation.golden_runner --cases tests/golden/data/smoke_golden_set.jsonl --validate-only --allow-unlabeled
```

第一条会因为现有 12 条缺标注而返回非零；第二条明确允许未标注诊断。正式运行需准备人工标注 JSONL 和真实索引、模型环境，再执行 `python -m services.evaluation.golden_runner --cases <标注文件> --repeat 3`。不带 `--validate-only` 会运行真实引擎；`--allow-unlabeled` 只放宽标注要求，不会切换为离线模型。

[`score_record()`](../../backend/services/evaluation/scoring.py) 对首次评分和重新评分使用相同失败分母。下面是 Python 调用示例，重新评分不重新运行研究图：

```python
import json
from pathlib import Path
from services.evaluation.scoring import score_record

record = json.loads(Path("已保存的记录.json").read_text(encoding="utf-8"))
case = json.loads(Path("单条人工标注.json").read_text(encoding="utf-8"))
scored = score_record(case, record)
print(json.dumps(scored, ensure_ascii=False, indent=2))
```

## 报告与基线

每题默认独立运行 3 次，保留全部答案、原始终态、逐次指标、中位数、标准差、最小/最大值及缺测数。`overall_metrics` 按题汇总中位数；`overall_run_metrics` 公开逐次均值，避免“两次成功、一次失败”的中位数掩盖故障。`run_summary` 单列有效运行、失败运行、失败率和终态波动，`metric_counts` 单列已评分、未评分、成功及失败样本数。

已标注的执行失败在适用质量指标中占分母并计 0，缺标注/不适用指标为 `null`。执行故障不能变成正确拒答，也不能用空指标把困难样本从分母删除。CLI 遇运行失败返回非零，指标退化只告警。

`baseline_ready` 要求标注完整、无执行失败或缺失轨迹、每题至少 3 次且配置完整稳定。只有数据集哈希、指标版本、重复次数、实际预算、模型配置和索引指纹一致，且上一份报告是有效基线时才生成自动 diff；缺少版本或运行中切换索引都会禁止比较。模型名应固定到可复现版本；供应商若在同名别名背后更新权重，配置指纹无法识别这种外部变化。

`recall@5`、`three_state_accuracy`、`citation_fidelity` 下降超过 5 个百分点进入 `red_flags`。报告还提供 difficulty/main_intent 分桶、可回答性 × completed/partial/abstained/error 矩阵和终止原因分布。

## 离线回归

```powershell
python -m pytest tests/unit/services/evaluation tests/unit/services/paper_evidence_research tests/integration/test_research_qa_contract.py tests/unit/services/test_llm_call_metrics.py
python smoke_test_research_stream.py
```

这些测试使用真实请求/结果模型、研究图、临时会话存储与受控外部 I/O，验证同步、SSE、记录重评分、引用支持、故障分母和成本统计，不生成真实 golden 基线。完整测试入口见 [测试指南](../operations/testing.md)。
