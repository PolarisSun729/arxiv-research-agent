# 确认循环问题修复说明

## 问题描述

用户点击"确认执行"按钮后，系统没有继续执行后续步骤，而是再次弹出相同的确认请求，形成无限循环。

## 问题根因

### 执行流程

1. **创建确认请求**：`parse_and_index_paper` 步骤需要确认，系统创建 `pending_confirmation` 并保存到数据库
2. **用户点击确认**：前端发送 resume 请求，后端调用 `consume_pending_confirmation` 原子地清空数据库中的确认状态
3. **恢复执行**：LangGraph 从 checkpoint 恢复，执行器清空内存中的 `runtime.pending_confirmation` 和 `state.runtime_state.pending_confirmation`
4. **持久化节点状态**：每个节点执行后，`_persist_runtime_checkpoint_node` 调用 `persist_state` 保存当前状态
5. **问题发生**：`persist_state` 从 `state.plan_runtime.pending_confirmation` 读取到**未清空的旧确认请求**，重新写入数据库
6. **再次进入确认门**：下次执行时，从数据库读取到 `pending_confirmation`，又创建了相同的确认请求

### 核心原因

**三处状态不同步**：
- `runtime.pending_confirmation` ✅ 被清空
- `state.runtime_state.pending_confirmation` ✅ 被清空  
- `state.plan_runtime.pending_confirmation` ❌ **没有被清空**

导致 `persist_state` 从 `plan_runtime` 读取到残留的确认状态，覆盖了之前消费时的清空操作。

## 修复方案

在 `_clear_confirmation_runtime_state` 方法中，**同时清空三处的 `pending_confirmation`**：

```python
def _clear_confirmation_runtime_state(self, ...) -> None:
    # 清空 runtime
    runtime.pending_confirmation = None
    
    # 清空 runtime_state
    if state.runtime_state is not None:
        state.runtime_state.pending_confirmation = None
    
    # 清空 plan_runtime（新增）
    if state.plan_runtime is not None:
        state.plan_runtime.pending_confirmation = None
```

## 关键日志

### 问题日志

```
14:21:26 - confirmation_consumed: decision=approve  ✅ 确认已消费
14:21:28 - confirmation required: runtime_approved=[] context_approved=[] checkpoint_approved=[]  ❌ 又判断需要确认
14:21:28 - confirmation_created: step_id=parse_and_index_paper  ❌ 再次创建确认
```

### 修复后预期

```
14:21:26 - confirmation_consumed: decision=approve  ✅ 确认已消费
14:21:28 - confirmation bypassed: source=runtime_approved_step_ids  ✅ 批准态生效，跳过确认
14:21:28 - tool_started: tool_name=parse_and_index_paper  ✅ 继续执行工具
```

## 文件修改

- `backend/agents/arxiv_search_agent/plan_executor.py` (第 2254-2282 行)
  - 修改 `_clear_confirmation_runtime_state` 方法
  - 新增对 `state.plan_runtime.pending_confirmation` 的清空

## 测试验证

1. 发起需要确认的操作（如解析论文）
2. 点击"确认执行"
3. 验证不再出现重复确认请求
4. 验证后续工具正常执行

## 架构改进建议

当前系统维护了三份执行现场：
- `PlanRuntime runtime` - 执行器内部运行态
- `AgentRuntimeState state.runtime_state` - 跨节点序列化快照
- `PlanRuntime state.plan_runtime` - 旧版兼容保留

**建议**：统一为单一真源（`runtime_state`），避免多份状态同步问题。

## 相关代码

- `plan_executor.py`: 执行器逻辑
- `runtime_checkpoint.py`: checkpoint 持久化
- `service.py`: stream 模式入口
- `database_service.py`: `consume_agent_runtime_pending_confirmation` 原子消费

## 日期

2026-06-16
