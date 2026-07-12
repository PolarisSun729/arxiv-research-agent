# PlanExecutor 与 Agent Runtime 重构设计

## 1. 问题结论

`backend/agents/arxiv_search_agent/plan_executor.py` 已经成为 Agent Runtime 的中心耦合点。它不仅编排计划执行，还直接承担输入绑定、工具契约校验、确认恢复、checkpoint 批准态恢复、多份运行时状态同步、trace、replan 和业务 fallback。

问题的核心不是文件行数，而是状态所有权、持久化事务、执行策略和流程编排没有清晰边界。现有测试覆盖了不少行为，但大量测试依赖完整 `PlanExecutor` 或其私有方法，无法为各子系统提供独立契约。

## 2. 重构目标

保留 `PlanExecutor` 作为 LangGraph 节点调用的薄编排门面，但进行 clean break：不保留旧确认字段、旧 resume payload、bridge step 或 checkpoint 兼容恢复逻辑。

重构完成后，`PlanExecutor` 只负责推进以下生命周期：

```text
选择 step
-> 解析条件与输入绑定
-> 执行守卫
-> 工具调用
-> observation
-> recovery decision
-> 统一投影状态
```

## 3. 核心架构决策

### 3.1 状态写入所有权

输入绑定器、工具执行器、确认处理器和 recovery policy 只返回结构化结果，不直接修改 `AgentState`、`PlanRuntime` 或 context。

只有 `PlanExecutor` 可以安排 transition 的应用，只有 `RuntimeStateProjector` 可以修改共享运行时状态。

### 3.2 双 checkpoint 分工

保留双 checkpoint，但禁止相互猜测或兼容回退：

- LangGraph checkpoint 只恢复图执行位置和框架内部状态。
- Agent runtime checkpoint 负责用户与会话归属、交互状态、原子消费和业务生命周期。

resume 必须精确匹配身份并同时通过两层 checkpoint 校验。删除默认用户兜底、按 thread 模糊扫描和旧状态修复逻辑。

### 3.3 统一交互协议

删除 `pending_action`、`pending_confirmation`、debug 确认镜像以及旧 resume payload，改用唯一的 `AgentInteraction`：

```python
class AgentInteraction:
    interaction_id: str
    kind: Literal["target_selection", "side_effect_approval"]
    status: Literal["pending", "resolved", "cancelled", "expired"]
    payload: TargetSelectionPayload | SideEffectApprovalPayload
    expires_at: datetime
```

API 恢复请求只提交 `interaction_id`、decision 和交互响应。前端提交的工具名、step ID 或完整参数不能作为服务端权威数据。

### 3.4 删除确认 bridge step

删除 `request_confirmation`。确认是目标副作用 step 的执行前置守卫，不是业务计划步骤。

执行顺序改为：输入绑定完成后，对最终参数计算 fingerprint；需要批准时创建交互；恢复后重新进入同一目标 step，并通过参数绑定的授权凭证放行。

### 3.5 区分目标选择和副作用批准

论文目标消歧使用 `target_selection`，只解决输入绑定问题，不产生副作用授权。

副作用工具使用 `side_effect_approval`。即使用户已经选择目标论文，执行后续副作用前仍需针对最终工具参数单独批准。

### 3.6 参数绑定的一次性授权

删除 `approved_step_ids`，改用 `ApprovalGrant`：

```python
class ApprovalGrant:
    plan_id: str
    step_id: str
    tool_name: str
    arguments_fingerprint: str
    approved_at: datetime
    confirmation_id: str
```

授权必须精确匹配 plan、step、工具和最终参数 fingerprint。replan、参数变化或 plan 变化都会使旧授权失效。

### 3.7 副作用调用生命周期

授权在调用副作用工具前消费，避免外部操作成功但本地超时后被自动重复调用。

消费授权和创建 `SideEffectInvocation(prepared)` 必须在同一数据库事务中完成。调用状态为：

```text
prepared -> invoking -> succeeded | failed | indeterminate
```

进程在 `prepared` 或 `invoking` 阶段崩溃时，默认进入 `indeterminate`，禁止通用执行器自动重放。只有工具提供稳定 idempotency key 或状态查询能力时，专用 reconciliation policy 才能继续处理。

### 3.8 工具执行策略

确认、重试和恢复能力由工具契约声明，不在执行器中按工具名硬编码：

```python
class ToolExecutionPolicy:
    effect: Literal["read_only", "external_read", "side_effect"]
    confirmation: Literal["never", "always", "conditional"]
    retry: Literal["safe", "idempotent_only", "never"]
    reconciliation: Literal["none", "query_status", "idempotency_key"]
```

### 3.9 Observation 与 recovery

`ObservationService` 只将工具结果转换为结构化 observation。`RecoveryPolicy` 使用确定性规则返回 continue、retry、replan、fallback、finalize 或 fail。

只有明确选择 replan 后，`Replanner` 才能调用 LLM。`PlanExecutor` 和 `FallbackPolicy` 不得隐式调用 LLM。Paper QA 等业务修复逻辑通过专用 policy 扩展，不进入通用执行器。

### 3.10 Replan 授权边界

每次 replan 生成新的 `plan_id` 和 `plan_version`。旧 plan 的 pending interaction 和未消费 grant 全部作废。

已经进入 prepared、invoking 或 indeterminate 的副作用调用不能撤销，只能保留执行事实并进入 reconciliation 流程。

### 3.11 持久化边界

Runtime checkpoint 只保存可恢复执行快照和当前 pending interaction。

以下实体独立持久化：

- `approval_grants`
- `side_effect_invocations`

`PlanExecutor` 不直接访问 store，由领域服务负责授权消费、invocation 创建和事务一致性。

### 3.12 结构化领域事件

业务审计使用稳定的结构化事件，例如：

```text
interaction_requested
interaction_resolved
approval_granted
approval_consumed
approval_revoked
tool_invocation_prepared
tool_invocation_started
tool_invocation_succeeded
tool_invocation_failed
tool_invocation_indeterminate
step_observed
plan_replanned
plan_completed
plan_failed
```

debug 和普通 trace 只用于诊断，不参与业务恢复，也不能被前端用于推断业务状态。

## 4. 建议模块边界

```text
plan_executor.py                 # 薄编排门面
execution/scheduler.py           # StepScheduler
execution/bindings.py            # InputBindingResolver、StepConditionEvaluator
execution/tools.py               # ToolExecutionService、ToolContractValidator
execution/interactions.py        # InteractionCoordinator、TargetSelectionHandler
execution/approvals.py           # ExecutionGuard、ApprovalGrantService
execution/side_effects.py        # SideEffectInvocationService、ReconciliationPolicy
execution/observation.py         # ObservationService
execution/recovery.py            # RecoveryPolicy、FallbackPolicy
execution/state.py               # RuntimeStateProjector、ExecutionTransition
execution/traces.py              # 结构化领域事件记录
```

## 5. 迁移顺序

1. 建立现有行为基线和端到端恢复测试。
2. 抽离条件求值、输入绑定和纯校验逻辑。
3. 抽离工具执行和工具策略契约。
4. 引入 `ExecutionTransition` 与 `RuntimeStateProjector`，收口所有共享状态写入。
5. 引入统一 `AgentInteraction`，拆分 target selection 与 side-effect approval。
6. 删除 bridge step、旧确认字段和旧 resume payload。
7. 引入 `ApprovalGrant`、`SideEffectInvocation` 及事务性 store。
8. 抽离 observation、recovery、replan 和 fallback policy。
9. 更新前端、SSE、API、测试和文档。
10. 使旧 schema 的未完成 checkpoint 失效，删除所有兼容代码。

不建立长期 feature flag 或新旧双轨。开发可以分阶段提交，但最终以后端、存储、前端和测试的一次原子切换交付。

## 6. 测试策略

- 编排契约测试只验证 `PlanExecutor` 的组件调用顺序。
- 组件单元测试覆盖绑定、校验、交互、授权、invocation、replan 和状态投影。
- 恢复集成测试使用真实或接近真实的双 checkpoint 与持久化 store。
- 端到端测试固定 SSE、interaction payload、最终响应和错误码。
- 重构完成后删除直接调用 `PlanExecutor` 私有方法的测试。

必须覆盖：正常批准、重复批准、参数变化、replan、LangGraph checkpoint 缺失、崩溃窗口、目标选择与批准隔离、旧 schema 拒绝。

## 7. Definition of Done

- `PlanExecutor` 仅承担 step 生命周期编排，且不直接依赖 store。
- `PlanExecutor` 不直接修改共享 runtime state。
- 执行器中没有具体工具名、Paper QA 或确认恢复的业务分支。
- 仓库中不存在旧确认字段、bridge step 和旧 resume payload 的有效引用。
- 唯一交互协议为 `AgentInteraction`。
- 副作用批准使用参数绑定、一次性的 `ApprovalGrant`。
- grant 消费与 invocation 创建在同一事务中完成。
- replan 会撤销旧 interaction 和未消费 grant。
- 旧 checkpoint schema 被明确拒绝，不存在兼容恢复 fallback。
- 前端不读取 debug 或 trace 推断业务状态。
- 测试围绕组件契约和可观察行为，不调用执行器私有方法。
- 新增状态迁移、LLM 调用、规则兜底、异常处理和 trace 逻辑包含必要、可维护的中文注释。

## 8. 面试官问题与标准回答

### 面试官问题

为什么不能只把 `plan_executor.py` 按函数分到几个文件？

### 标准回答

因为当前问题是状态、事务和策略所有权混乱，而不只是文件过长。如果多个新模块仍然共同修改 `PlanRuntime`、`AgentState`、context 和 checkpoint，拆文件只会形成分布式 God Object。必须先规定 transition 返回值和唯一状态投影入口。

### 面试官问题

为什么 LangGraph checkpoint 和业务 runtime checkpoint 不能合并？

### 标准回答

LangGraph checkpoint 恢复图执行位置和框架内部状态；业务 checkpoint 负责用户归属、交互生命周期、原子消费与防重。两者的一致性要求不同，因此应严格分工，并在 resume 前同时校验，而不是互相替代。

### 面试官问题

为什么要删除 `request_confirmation` bridge step？

### 标准回答

确认不是业务计划步骤，而是副作用 step 的执行前置守卫。bridge step 会产生 bridge ID 与 target step ID 的映射，使批准容易记录到错误对象并导致重复确认。守卫直接绑定目标 step 和最终参数后，状态机更简单且授权语义更准确。

### 面试官问题

为什么 `approved_step_ids` 不够安全？

### 标准回答

step ID 无法证明用户批准的是哪组最终参数。replan、输入重新绑定或 step ID 复用后，旧批准可能放行不同操作。参数绑定的 `ApprovalGrant` 将 plan、step、工具和参数 fingerprint 组合成明确授权边界。

### 面试官问题

为什么授权要在工具调用前消费？

### 标准回答

副作用工具可能已经在外部成功，但本地因超时没有收到结果。如果成功后才消费授权，自动重试可能重复产生副作用。调用前消费保证一次批准最多触发一次调用；结果未知时进入 indeterminate，而不是静默重试。

### 面试官问题

为什么论文目标选择不能等同于副作用批准？

### 标准回答

目标选择解决“输入指向谁”，副作用批准解决“是否允许执行某个具体操作”。选择论文并不代表用户批准解析、建索引或其他副作用。两者必须使用不同 interaction 类型和状态迁移。

### 面试官问题

为什么不迁移旧的未完成 checkpoint？

### 标准回答

旧 checkpoint 缺少参数 fingerprint、grant 和 invocation 状态，而且多份确认镜像可能冲突。迁移无法证明旧批准对应新模型中的哪次具体操作。通过 schema version 明确失效，比引入长期兼容和潜在越权执行更可靠。
