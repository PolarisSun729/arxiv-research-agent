# PlanExecutor 执行链路说明（拆分前行为基准）

> 本文档对应「等价拆分前」阶段，目标是**冻结 PlanExecutor 当前的运行行为**，
> 为后续把执行、观察、确认、重规划拆成独立 LangGraph 节点提供一份可对照的基准。
>
> 范围：`backend/agents/arxiv_search_agent/plan_executor.py`。
> 它**只描述现状**，不提出改造方案；任何后续重构都应保持本文档描述的外部行为不变。

## 1. 一次 Agent turn 的整体状态流转

一次成功的 turn 大致经过：

```
build_executable_plan(state)            # planner 门面产出可信 ExecutablePlan
        │
        ▼
PlanExecutor.execute(plan, state)       # 初始化 PlanRuntime（step_status=pending、清空 outputs/trace/计数）
        │
        ▼
_execute_runtime(runtime, state)        # 主循环：反复 execute_next_step 直到终态
        │
        ├── select_next_step            # 选出下一个可执行 step（只看依赖/条件/前置，不调工具）
        │
        ├── _execute_step               # 绑定输入 → 确认门 → 调工具 → 校验 → 观察 → (replan)
        │
        ▼
_build_turn_result(runtime)             # 收束成 AgentTurnResult（success/waiting_confirmation/...）
```

`PlanRuntime` 是执行期的可变真源（step 状态、outputs、trace、retry/replan 计数、pending_confirmation、
approved_step_ids 等）。每次单步执行后通过 `_sync_runtime_state` 把它投影成 `AgentRuntimeState`
（一等、可序列化的执行现场）回写到 `AgentState.runtime_state` 和 `AgentState.plan_runtime`。

### 两类入口

| 入口 | allow_interrupt | 确认链路 |
| --- | --- | --- |
| `run_agent_turn` / `execute` / `execute_runtime` | `False` | 命中确认时直接返回 `waiting_confirmation`，不调用 LangGraph `interrupt` |
| `run_agent_turn_in_graph` / `_execute_runtime(..., allow_interrupt=True)` | `True` | 命中确认时调用 `interrupt()` 暂停，由前端 resume payload 决定 approve/reject |

此外存在一组**显式单步边界方法**，是面向「未来 LangGraph 多节点」准备的，目前与大循环并存：
`select_next_step`、`execute_current_step_tool`（只调工具）、`observe_current_step`（只观察）、
`replan_after_observation`（只重规划）、`finalize_runtime`（收束）。它们消费/产出 `StepExecutionResult`。

> ⚠️ 重复逻辑提示：`_execute_step`（大循环用）与
> `execute_current_step_tool` + `observe_current_step` + `replan_after_observation`（显式节点用）
> 实现了**两套等价的执行/观察/重规划链路**。拆分时这是首要的去重目标，但当前两条路径都必须保持现有行为。

## 2. 七类关键行为边界

### 2.1 普通工具步骤如何执行
`_execute_step` / `execute_current_step_tool`：
1. `step_status = running`，记录 `started_at`。
2. `_resolve_input_bindings` 绑定输入（见 2.2）。
3. 副作用复用检查（见 2.5）。
4. 确认门检查（见 2.3）。
5. `step_started` trace → 进入 `max_attempts` 重试循环调用 `_invoke_step_tool`。
6. `_invoke_named_tool`：取 contract → `_augment_tool_input` 注入 state 上下文 → `_validate_tool_input`（Pydantic）→ `adapter.execute` → `_validate_tool_output`。
7. 工具成功后 `project_tool_result` 归一化输出 → `duplicate_output_key` 守卫 → postcondition 校验（见 2.6）→ Observer 观察（见 2.6）。
8. 观察 success/partial_success：`_record_step_output` 写 outputs，`step_status=success`，`step_succeeded` trace。

### 2.2 工具输入如何从上下文绑定
`_resolve_input_bindings` 按 `StepInputBinding.source_type` 取值：
`state` / `context` / `goal` / `search_spec` / `step_output` / `literal`。

- `_extract_state_value` 在边界收敛别名：`user_request` / `original_question` / `question` → `state.message`。
- 取值为空（`None/""/[]/{}`）且 `required=True` 时，立即返回 `missing_input`，**不执行工具、不进入确认门**。
- `_augment_tool_input` 在模型校验前为特定工具注入 state 上下文（如 `normalize_request`/`build_arxiv_search_spec` 兜底 `search_spec`、`message`、`intent`；`analyze_ambiguity` 注入 `context/user_id/search_spec/pending_action/goal`；`request_confirmation` 注入 `pending_state` 等）。

### 2.3 什么时候需要人工确认
`_needs_confirmation`：仅当 `step.confirmation_policy.requires_confirmation == True` 时才可能需要确认；
满足以下任一**批准来源**则跳过确认门（按优先级）：
1. `runtime.approved_step_ids` / `state.plan_runtime` / `state.runtime_state` 中的 approved_step_ids；
2. `state.context["approved_step_ids"]`；
3. 业务 checkpoint（`_checkpoint_approved_step_ids_for_state`）——**真源**，且只接受 `status=="running"`、`pending_confirmation` 为空、`plan_id` 匹配的记录（见 2.4）。

`state.pending_action` 是前端**展示镜像**，绝不能作为批准真源（仅打日志）。

命中确认时走 `_handle_confirmation_gate`：写 `waiting_confirmation` 状态、`pending_confirmation`、
`confirmation_created` + `confirmation_requested` trace。`allow_interrupt=False` 时直接收束；
`allow_interrupt=True` 时调用 `interrupt()` 等待 resume。

### 2.4 用户确认后如何恢复现场
- **inline resume**（`_handle_confirmation_gate` 内 interrupt 返回）：`_normalize_confirmation_resume_payload`
  把 payload 归一成 `approve`/`reject`（**未知值一律按 reject**，防止误执行副作用）。
- **节点重入 resume**（`execute_current_step_tool` 开头 `_should_resume_confirmation_for_step`）：
  LangGraph resume 会从节点函数开头重跑，因此先 `_resume_pending_confirmation` 消费同一确认请求，
  把批准态落到 runtime/context/runtime_state 三处真源，再继续——避免二次弹窗。
- approve 收尾统一走 `_consume_confirmation_approval`：写 `approved_step_ids`（针对**目标 step**，桥接确认时是后续副作用 step）、清空 pending_confirmation、重置 turn_status、
  写 `pending_action(status=approved)`、`confirmation_approved` + `confirmation_consumed` trace。
- **桥接步**（`request_confirmation`）：approve 后桥接步本身置 `success`，真正的副作用 step 保持 `pending`，由下一轮 `select_next_step` 借 approved_step_ids 跳过确认门后执行。
- **目标论文确认**（`paper_target_confirmation`）：approve 时 `_materialize_paper_target_confirmation`
  只在 pending confirmation 保存的候选集合内匹配（`_match_confirmed_candidate`），**禁止 resume 时重新解析自然语言**；
  校验 `pending_action_id`，匹配失败则按 reject 处理。

### 2.5 哪些工具有副作用，如何避免重复执行
`_can_reuse_side_effect_output`：当 `side_effect_level ∈ {persistent_write, external_call}` 且
`output_key` 已存在非空输出时，视为本轮可复用结果——置 `success`、写 `step_reused_output` trace、
**不再调用后端工具**。这条规则在 `_execute_step` 和 `execute_current_step_tool` 两处都有，保证 resume/局部 replan 不会重复外部调用或重复持久写入。

### 2.6 工具失败 / postcondition 失败 / observer 失败时如何进入 replan
失败分三层，语义不同：

| 失败类型 | 触发点 | runtime.error 前缀 | 后续 |
| --- | --- | --- | --- |
| 缺必填输入 | `_resolve_input_bindings` | `missing_input:` | `step_failed`，recovery=ask_clarification，**不进 replan** |
| 工具执行异常 | `_invoke_step_tool` 抛异常 | `tool_execution_failed:` | 重试耗尽后 `step_failed` |
| 重复输出键 | `duplicate_output_key` 守卫 | `duplicate_output_key:` | `step_failed`，recovery=abort_with_error |
| postcondition 失败 | `_evaluate_condition(postconditions)` | `postcondition_failed:` | `step_failed`，recovery=abort_with_error |
| 结构化工具错误 / 质量不足 | Observer 返回非 success/partial_success | （走 replan） | `needs_replan=True` → `_handle_observation_replan` |

`_handle_observation_replan` 调用 `Replanner.replan`：
- 成功 patch：替换 plan、合并 replan 计数与 trace、当前 step 置 success（低质量输出不污染 final_answer）、`step_replanned` trace。
- 无法 patch：当前 step 置 failed、`record_recovery_fallback`、`replan_fallback` trace，turn 收束为 `fallback`。

> `auto_replan=False`（显式节点路径）时，`_execute_step` 只记录观察结果并暂停本步，把 replan 留给独立的 `replan_after_observation`。

### 2.7 replan 终止条件（不会无限循环）
`Replanner` 三道上限（`replanner.py`）：
- `MAX_REASON_REPLANS = 2`（同一 reason_key）
- `MAX_STEP_REPLANS = 2`（同一 step）
- `MAX_PLAN_REPLANS = 5`（整轮总次数）

任一超限即 `replan_limit_exceeded` trace + 保守 fallback，**保证 replan 有限终止**。
Paper QA 修复链复用同一上限，超限时 fallback 决策里 `max_repair_limit_triggered=True`。

## 3. 终态与外部契约

`_build_turn_result` 把 runtime 收束成 `AgentTurnResult.status`：

| 条件 | status |
| --- | --- |
| `runtime.pending_confirmation` 非空 | `waiting_confirmation` |
| 存在 failed step 且无 final_answer | `failed` |
| `goal_type == "unclear"` | `need_clarification` |
| `goal_type == "unsupported"` | `fallback` |
| 其余 | `success` |

**拆分必须保持不变的外部接口：**
- `run_agent_turn` / `run_agent_turn_in_graph` / `execute` / `execute_runtime` 的签名与返回 `AgentTurnResult`；
- `AgentTurnResult` / `StepExecutionResult` / `ConfirmationRequest` 的字段形态（前端协议、resume payload 依赖）；
- 确认 payload（`interrupt(confirmation_request.model_dump())`）的结构；
- step 状态机取值：`pending/running/success/failed/skipped/waiting_confirmation`。

后续拆分只做**内部结构整理**：不改前端协议、不改路由返回格式、不改 Agent 对外调用方式。

## 4. Trace 事件清单（调试可观测的真源）

执行链路通过 `_append_trace` 写入 `ExecutionTrace`，关键事件：

```
step_started / step_succeeded / step_failed / step_observed
step_reused_output            # 副作用复用
condition_skipped / precondition_failed / dependency_blocked
confirmation_created / confirmation_requested
confirmation_approved / confirmation_rejected / confirmation_consumed
paper_target_confirmation_invalid
plan_replanned / replan_fallback / replan_limit_exceeded
paper_qa_quality_decision     # Paper QA 质量闭环
```

每次执行「走了哪条路径」可由这串 trace 完整回放。第 3 步阶段会在 `step_started` / `step_succeeded` 等事件补充
归一化的「执行决策摘要」字段（current step / tool / 是否确认 / 是否 resume / 是否 retry / 是否 replan /
是否复用副作用 / 最终状态），详见 `runtime_state` 投影与新增的 `execution_path` 摘要字段。


## 5. 执行状态机与终态可区分性（第 3 步收紧）

### 5.1 状态语义

一次 Agent turn 经过的执行阶段（由 `runtime.step_status` 的单步状态与 `runtime.turn_status` 的整轮终态共同表达）：

| 阶段 | 载体 | 含义 |
|---|---|---|
| planning | planner 门面 | 构造并校验 `ExecutablePlan`，执行器只消费可信计划 |
| executing | step_status=`running` | 内核正在执行单步工具调用 |
| waiting_confirmation | step_status=`waiting_confirmation` / turn_status=`waiting_confirmation` | 副作用工具前暂停，等待用户确认 |
| observing | `step_observed` trace | Observer 判定结果质量 |
| replanning | `plan_replanned` trace | 质量不足触发重规划 patch plan |
| completed | turn_status=`success` | 正常完成，`final_answer` 可用 |
| failed | turn_status=`failed` | 输入缺失 / 工具失败 / postcondition 失败 / 恢复失败 |
| fallback | turn_status=`fallback` | replan 上限耗尽或不可恢复，收口为兜底答案 |
| cancelled | step_status=`skipped` + `confirmation_rejected` trace | 用户拒绝确认，原副作用步骤被安全跳过 |

### 5.2 终态可区分性

`AgentTurnResult.status` 是对外可区分的终态枚举：

- **正常完成** → `success`，`final_answer` 存在。
- **等待用户确认** → `waiting_confirmation`，`pending_confirmation` 非空。
- **用户取消** → 整轮 `success`/`fallback`（视后续步骤），但 `state.pending_action["status"]=="rejected"` 且对应 step `skipped`、有 `confirmation_rejected` trace。
- **工具失败** → `failed`，`error` 形如 `missing_input:` / `tool_execution_failed:` / `postcondition_failed:`。
- **replan 后失败** → `fallback`，`error` 形如 `fallback:<step>:<reason>`，`recovery_strategy.type=="fallback_answer"`，含 `replan_fallback` / `replan_limit_exceeded` trace。
- **状态恢复失败** → `failed`，`error=="observation_missing_step_output"` / `replan_missing_observation`。

`state.debug["execution_path"].final_status` 提供归一化只读摘要：`success / replanned_success / waiting_confirmation / failed / need_clarification / in_progress`。

## 6. 四类高风险状态的一致性约束（第 3 步）

### 6.1 confirmation 生命周期

- 一个 `ConfirmationRequest` 只绑定一个 plan_id / step_id / 候选集合 / 恢复点（pending_action_id）。
- 没有 pending confirmation 时不接受确认输入；service 层 `checkpoint_manager.validate_resume` 校验 step_id / tool_name / pending_action_id 一致。
- 用户选择不在候选集合中（`_match_confirmed_candidate` 返回 None）→ 拒绝，不执行。
- 确认成功 → `_consume_confirmation_approval` / `_materialize_paper_target_confirmation` 清空 pending、写 approved_step_ids。
- 用户拒绝 → `_consume_confirmation_rejection` 把 step 标 `skipped`、清 pending、设 `skip_step` recovery，原副作用工具不再执行。

### 6.2 checkpoint 恢复一致性

`_checkpoint_approved_step_ids_for_state` 放行旧批准态前必须同时满足：

1. `status == "running"`；
2. `pending_confirmation` 已清空；
3. `step_id` 在 `approved_step_ids` 内；
4. **plan_id 一致**（`checkpoint_plan_id == current_plan_id`）；
5. **goal 类型一致**（`checkpoint_goal_type == current_goal_type`，第 3 步新增）。

任意一项不满足 → 返回空批准集，本轮重新进入确认门，旧 checkpoint 绝不静默恢复到新 plan。

### 6.3 side effect 幂等

- `side_effect_level ∈ {persistent_write, external_call}` 的工具，已有非空 `output_key` 输出时，`_can_reuse_side_effect_output` 在 resume / retry / replan 后复用结果，不重复调用。
- 复用入口唯一（第 2 步已收敛到 `execute_current_step_tool` 内核），旧路径只作薄包装委托。

### 6.4 replan 边界

- 三级上限：`MAX_REASON_REPLANS=2` / `MAX_STEP_REPLANS=2` / `MAX_PLAN_REPLANS=5`（`replanner.py`）。
- 每次 replan 在 `plan_replanned` trace 记录 reason / failure_category / observation_signal。
- replan patch plan 时不覆盖关键历史输出：`step.output_key != "final_answer"` 且 `output_key not in outputs` 才写入，已有成功输出受保护。
- 上限耗尽 → `replan_limit_exceeded` + `replan_fallback`，返回可解释 `fallback` 终态，不无限循环。
