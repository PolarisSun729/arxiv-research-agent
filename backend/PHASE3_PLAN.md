# 阶段三（P2）类型安全提升详细计划

## 防御性代码分析

### 🔍 当前防御性代码分类

#### 类别 A：**必须保留** - 处理外部不可控数据
```python
# 1. trace_events 内容（来自运行时，格式可能变化）
event.get("event_type")  # ✅ 保留
event.get("configuration")  # ✅ 保留
event.get("index_snapshot")  # ✅ 保留
isinstance(snapshot, dict)  # ✅ 保留

# 2. Git 命令执行（外部进程）
try:
    result = subprocess.run(...)
except (OSError, subprocess.SubprocessError):  # ✅ 保留
```

#### 类别 B：**可以删除** - Pydantic 已验证的字段
```python
# 调用方保证非空
events = list(trace_events)  # ❌ 删除 - trace_events 已是 list
summary = result.research_summary.model_dump(mode="json")  # 已验证

# result 已经是 PaperEvidenceResearchResult，不会为 None
if result:  # ❌ 删除条件判断
```

#### 类别 C：**可以删除** - 冗余的类型转换
```python
str(main_intent).strip().lower()  # ❌ 如果已知是 str，不需要 str()
main_intent if main_intent else "unknown"  # ❌ 可简化为 main_intent or "unknown"
```

---

## Phase 3.1: Pydantic 模型（1-2小时）

### Step 3.1.1: 创建类型模型

**文件**: `backend/services/evaluation/types.py`

```python
"""评测系统类型定义"""

from typing import Any
from pydantic import BaseModel, Field, field_validator


class LLMUsage(BaseModel):
    """LLM 调用统计
    
    从 LLMCallStats.to_dict() 输出格式定义
    """
    llm_calls: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    usage_reported_calls: int | None = None
    observed_input_tokens: int | None = None
    observed_output_tokens: int | None = None
    observed_total_tokens: int | None = None
    models: list[str] = Field(default_factory=list)
    task_calls: dict[str, int] = Field(default_factory=dict)
    
    @field_validator("llm_calls")
    @classmethod
    def validate_calls(cls, v: int) -> int:
        if v < 0:
            raise ValueError("llm_calls must be non-negative")
        return v
    
    class Config:
        extra = "allow"  # 允许未来扩展字段


class EvalError(BaseModel):
    """评测错误信息"""
    code: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    error_type: str | None = None
    
    @field_validator("code", "stage")
    @classmethod
    def reject_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("字段不能为空白")
        return v.strip()


class EvalRecordInput(BaseModel):
    """评测记录构造输入 - 成功场景
    
    用于参数验证和文档化
    """
    request: Any  # PaperEvidenceResearchRequest
    result: Any   # PaperEvidenceResearchResult
    trace_events: list[dict[str, Any]] = Field(min_length=1)
    turn_id: str = ""
    latency_ms: float = Field(ge=0)
    llm_usage: LLMUsage
    
    @field_validator("latency_ms")
    @classmethod
    def validate_latency(cls, v: float) -> float:
        if v < 0:
            raise ValueError("latency_ms must be non-negative")
        return v


class EvalRecordErrorInput(BaseModel):
    """评测记录构造输入 - 失败场景"""
    request: Any  # PaperEvidenceResearchRequest
    error: EvalError
    trace_events: list[dict[str, Any]] = Field(min_length=1)
    latency_ms: float = Field(ge=0)
    llm_usage: LLMUsage | None = None
```

### Step 3.1.2: 更新函数签名（向后兼容）

**文件**: `services/evaluation/eval_record.py`

```python
from .types import LLMUsage, EvalError

def build_success_eval_record(
    *,
    request: PaperEvidenceResearchRequest,
    result: PaperEvidenceResearchResult,
    trace_events: list[dict[str, Any]],
    turn_id: str = "",
    latency_ms: float,
    llm_usage: dict[str, Any] | LLMUsage,  # 兼容两种输入
) -> dict[str, Any]:
    """构造成功执行的评测记录
    
    Args:
        request: 研究请求
        result: 研究结果
        trace_events: 轨迹事件列表（至少1个事件）
        turn_id: 对话轮次ID
        latency_ms: 延迟毫秒数（非负）
        llm_usage: LLM使用统计（dict或LLMUsage）
    
    Returns:
        评测记录字典
    
    Raises:
        ValueError: 参数验证失败
    """
    # 入口验证和归一化
    if not trace_events:
        raise ValueError("trace_events cannot be empty")
    if latency_ms < 0:
        raise ValueError("latency_ms must be non-negative")
    
    # 归一化 llm_usage
    usage = LLMUsage(**llm_usage) if isinstance(llm_usage, dict) else llm_usage
    usage_dict = usage.model_dump()
    
    # 内部逻辑 - 减少防御性代码
    configuration = _extract_configuration_from_trace(trace_events, request)
    summary = result.research_summary.model_dump(mode="json")
    
    return EvaluationRecord(
        record_id=f"eval-{request.research_run_id}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        app_version=_get_git_commit_hash(),
        run_status="success",
        user_id=request.user_id,
        session_id=request.session_id,
        turn_id=turn_id,
        paper_context=_build_paper_context_from_trace(trace_events, request.arxiv_id),
        configuration=configuration,
        query={
            "raw": request.original_question,
            "rewritten": request.original_question,
            "main_intent": _extract_main_intent_from_trace(trace_events),
        },
        outcome=result.outcome,
        answer=result.answer,
        termination_reason=result.research_summary.termination_reason,
        citations=[c.model_dump(mode="json") for c in result.citations],
        research_summary=summary,
        efficiency={
            "latency_ms": round(latency_ms, 1),
            **usage_dict,
            "retrieval_count": summary.get("retrieval_count"),
            "draft_attempt_count": summary.get("draft_attempt_count"),
        },
        trace_ref=request.research_run_id,
        trace_events=trace_events,  # 直接使用，不再 list() 包装
        error=None,
    ).model_dump(mode="json")


def build_error_eval_record(
    *,
    request: PaperEvidenceResearchRequest,
    error: dict[str, Any] | EvalError,
    trace_events: list[dict[str, Any]],
    latency_ms: float,
    llm_usage: dict[str, Any] | LLMUsage | None = None,
) -> dict[str, Any]:
    """构造失败执行的评测记录
    
    Args:
        request: 研究请求
        error: 错误信息（dict或EvalError）
        trace_events: 轨迹事件列表（至少1个事件）
        latency_ms: 延迟毫秒数（非负）
        llm_usage: LLM使用统计（可选）
    
    Returns:
        评测记录字典
    
    Raises:
        ValueError: 参数验证失败
    """
    # 入口验证
    if not trace_events:
        raise ValueError("trace_events cannot be empty")
    if latency_ms < 0:
        raise ValueError("latency_ms must be non-negative")
    
    # 归一化
    error_obj = EvalError(**error) if isinstance(error, dict) else error
    usage = LLMUsage(**llm_usage) if isinstance(llm_usage, dict) else llm_usage if llm_usage else None
    
    configuration = _extract_configuration_from_trace(trace_events, request)
    
    return EvaluationRecord(
        record_id=f"eval-{request.research_run_id}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        app_version=_get_git_commit_hash(),
        run_status="error",
        user_id=request.user_id,
        session_id=request.session_id,
        turn_id="",
        paper_context=_build_paper_context_from_trace(trace_events, request.arxiv_id),
        configuration=configuration,
        query={
            "raw": request.original_question,
            "rewritten": request.original_question,
            "main_intent": _extract_main_intent_from_trace(trace_events),
        },
        outcome=None,
        answer="",
        termination_reason=None,
        citations=[],
        research_summary={},
        efficiency={
            "latency_ms": round(latency_ms, 1),
            **(usage.model_dump() if usage else {}),
        },
        trace_ref=request.research_run_id,
        trace_events=trace_events,
        error=error_obj.model_dump(),
    ).model_dump(mode="json")
```

---

## Phase 3.2: mypy 配置（2-3小时）

### Step 3.2.1: 创建 mypy 配置

**文件**: `backend/pyproject.toml` 或 `backend/mypy.ini`

```ini
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
warn_redundant_casts = true
warn_unused_ignores = true
check_untyped_defs = true
disallow_any_generics = false
disallow_untyped_defs = false  # 全局宽松
no_implicit_optional = true
strict_optional = true

# 评测模块严格检查
[[tool.mypy.overrides]]
module = "services.evaluation.*"
disallow_untyped_defs = true
disallow_incomplete_defs = true
warn_return_any = true

# 第三方库存根
[[tool.mypy.overrides]]
module = "pymilvus.*"
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "langsmith.*"
ignore_missing_imports = true
```

### Step 3.2.2: 修复类型问题

**优先修复的文件**:
1. `services/evaluation/types.py` ✅ 新文件，从头严格
2. `services/evaluation/eval_record.py` ⭐️ 核心
3. `services/evaluation/metrics_generation.py`
4. `services/evaluation/metrics_report.py`
5. `services/evaluation/configuration.py`

**运行**:
```bash
cd backend
mypy services/evaluation/
```

---

## Phase 3.3: 减少防御性代码（1-2小时）

### 原则
1. ✅ **保留**: 处理外部数据（trace_events 内容、subprocess 调用）
2. ✅ **保留**: 有明确业务含义的 fallback
3. ❌ **删除**: Pydantic 已验证的参数
4. ❌ **删除**: 冗余的类型转换和空值检查

### 具体删除清单

#### eval_record.py

```python
# ❌ 删除
events = list(trace_events)
# ✅ 改为
# 直接使用 trace_events（已验证非空列表）

# ❌ 删除
main_intent if main_intent else "unknown"
# ✅ 改为
main_intent or "unknown"

# ❌ 删除（如果已验证 main_intent 是 str）
str(main_intent).strip().lower()
# ✅ 改为
main_intent.strip().lower()
```

#### metrics_report.py

```python
# ❌ 删除（record 已经是 dict）
metrics = rec.get("metrics") or {}
# ✅ 改为
metrics = rec.get("metrics", {})

# ❌ 删除冗余的 or
value = metrics.get(metric_name) if metrics else 0.0
# ✅ 改为（metrics 保证是 dict）
value = metrics.get(metric_name)
```

#### _extract_main_intent

```python
# ❌ 删除冗余的 str() 转换
main_intent = str(config_event.get("main_intent"))
# ✅ 改为（如果已知是 str）
main_intent = config_event.get("main_intent", "")

# ❌ 删除冗余
return main_intent if main_intent else "unknown"
# ✅ 改为
return main_intent or "unknown"
```

### ✅ 必须保留的防御代码

```python
# ✅ 外部数据 - 保留
event.get("event_type")
event.get("configuration")
snapshot = event.get("index_snapshot")
if isinstance(snapshot, dict):  # 运行时类型检查

# ✅ 外部进程 - 保留
try:
    subprocess.run(...)
except (OSError, subprocess.SubprocessError):
    ...

# ✅ 业务逻辑 fallback - 保留
if config_event:
    return config_event["configuration"]
return {"research_limits": ..., "engine": {}}  # 最小契约
```

---

## 测试策略

### 单元测试更新
- 新增 `test_types.py` - 测试 Pydantic 模型验证
- 更新现有测试 - 确保新签名兼容

### 回归测试
```bash
# 运行完整测试套件
pytest backend/tests/unit/services/evaluation/ -v

# mypy 检查
mypy services/evaluation/
```

---

## 预期收益

### 代码质量
- ✅ **类型安全**: Pydantic + mypy 双重保障
- ✅ **代码简洁**: 减少 20-30 行防御代码
- ✅ **可读性**: 减少嵌套条件和 or/get 链
- ✅ **IDE 支持**: 更好的自动补全和错误提示

### 维护性
- ✅ **明确契约**: 输入验证在一处
- ✅ **快速失败**: 参数错误立即发现
- ✅ **文档化**: Pydantic 模型即文档

---

## 执行顺序

1. **Phase 3.1**: 创建 types.py + 更新 eval_record.py（1-2h）
2. **Phase 3.3**: 删除防御代码（与 3.1 同步进行）（1h）
3. **Phase 3.2**: 配置 mypy + 修复类型（2h）
4. **测试验证**: 回归测试（30min）

**总计**: 4-5 小时

---

## 风险控制

### 高风险点
1. **Pydantic 验证过严** - 可能拒绝合法输入
   - 缓解: 使用 `extra = "allow"`，逐步收紧
   
2. **删除必要的防御代码** - 可能引入运行时错误
   - 缓解: 严格遵循"保留外部数据检查"原则

### 回滚策略
- 每个 Phase 独立 commit
- 测试失败立即停止，不继续下一 Phase
