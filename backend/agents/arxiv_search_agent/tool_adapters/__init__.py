"""强类型 ToolAdapter 包。

各业务域 adapter 只负责把已经校验过的 InputModel 转成业务调用，并统一返回
ToolExecutionResult；工具选择、风险、确认和恢复策略仍由 ToolContract 维护。
"""

from .models import ToolError, ToolExecutionResult

__all__ = ["ToolError", "ToolExecutionResult"]
