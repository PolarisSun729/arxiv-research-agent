"""评测服务：记录评测数据、计算指标、生成报告。

Phase 3: eval_record 落盘
Phase 4: metrics 计算 + runner + 报告生成
"""

from __future__ import annotations

__all__ = ["write_eval_record"]

from .eval_record import write_eval_record
