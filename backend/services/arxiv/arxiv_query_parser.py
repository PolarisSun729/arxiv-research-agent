"""arXiv 查询解析器模块。

该模块负责解析 arXiv 查询字符串，将其转换为抽象语法树（AST），
用于后续的 SQL 编译和执行。支持本地 OAI 镜像的查询子集。
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from services.arxiv.local_oai_search_contract import (
    LOCAL_OAI_SUPPORTED_QUERY_SUBSET,
    UnsupportedLocalArxivQuery,
)

logger = logging.getLogger(__name__)

class ArxivQueryParser:
    """arXiv 查询解析器，将查询字符串解析为 AST。"""

    def parse(self, query: str) -> Optional[Dict[str, Any]]:
        """解析本地支持的 arXiv 查询子集；超出子集时显式失败而不是 Python 兜底。"""
        text = self._strip_query_outer_parentheses(str(query or "").strip())
        if not text:
            return None
        for operator, node_type in ((" ANDNOT ", "andnot"), (" AND ", "and"), (" OR ", "or")):
            if operator in text:
                parts = self._split_query_top_level(text, operator)
                if len(parts) > 1:
                    # 只有顶层真正发生拆分时才递归构树，避免括号内操作符造成死递归。
                    return {
                        "type": node_type,
                        "children": [self.parse(part) for part in parts],
                    }
        return self._parse_atomic_query(text, original_query=query)

    def _strip_query_outer_parentheses(self, query: str) -> str:
        """移除查询最外层成对括号；解析阶段需要尊重引号，避免 phrase 被误拆。"""
        text = query.strip()
        while text.startswith("(") and text.endswith(")"):
            depth = 0
            in_quote = False
            escaped = False
            balanced = True
            for index, char in enumerate(text):
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == '"':
                    # 解析本地子集时需要保留 phrase 内部原样内容，不能把引号里的括号当结构符。
                    in_quote = not in_quote
                    continue
                if in_quote:
                    continue
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0 and index != len(text) - 1:
                        balanced = False
                        break
            if balanced and depth == 0 and not in_quote:
                text = text[1:-1].strip()
            else:
                break
        return text

    def _split_query_top_level(self, query: str, token: str) -> List[str]:
        """按顶层布尔操作符切分查询；引号和括号内的操作符只作为普通文本处理。"""
        text = query.strip()
        parts: List[str] = []
        depth = 0
        in_quote = False
        escaped = False
        start = 0
        index = 0
        token_length = len(token)
        while index < len(text):
            char = text[index]
            if escaped:
                escaped = False
                index += 1
                continue
            if char == "\\":
                escaped = True
                index += 1
                continue
            if char == '"':
                in_quote = not in_quote
                index += 1
                continue
            if not in_quote:
                if char == "(":
                    depth += 1
                    index += 1
                    continue
                if char == ")":
                    depth = max(depth - 1, 0)
                    index += 1
                    continue
                if depth == 0 and text.startswith(token, index):
                    # 只有真正位于顶层的布尔操作符，才允许成为 AST 的拆分边界。
                    parts.append(text[start:index].strip())
                    index += token_length
                    start = index
                    continue
            index += 1
        parts.append(text[start:].strip())
        return [part for part in parts if part]

    def _parse_submitted_date_range(self, text: str, *, original_query: str) -> Dict[str, Any]:
        """把 submittedDate 范围子句解析成标准日期节点。"""
        match = re.fullmatch(r"submittedDate:\[(\d{12})\s+TO\s+(\d{12})\]", text.strip())
        if not match:
            raise UnsupportedLocalArxivQuery(
                "本地 OAI 镜像只支持 submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM] 日期范围。",
                query=original_query,
                reason="unsupported_submitted_date_syntax",
            )
        start_raw, end_raw = match.groups()
        # 统一编译成 SQLite 可直接比较的 datetime 字符串，减少后续节点类型分支复杂度。
        start_dt = datetime.strptime(start_raw, "%Y%m%d%H%M").strftime("%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(end_raw, "%Y%m%d%H%M").strftime("%Y-%m-%d %H:%M:%S")
        return {"type": "date", "start": start_dt, "end": end_dt}

    def _parse_atomic_query(self, text: str, *, original_query: str) -> Dict[str, Any]:
        """解析单个不可再拆分的本地 OAI 查询子句。"""
        text = self._strip_query_outer_parentheses(text.strip())
        if not text:
            return {"type": "empty"}
        if text.startswith("submittedDate:"):
            return self._parse_submitted_date_range(text, original_query=original_query)

        field = "all"
        raw_value = text
        if ":" in text:
            field, raw_value = text.split(":", 1)
            field = field.strip().lower()
            raw_value = raw_value.strip()

        aliases = {
            "title": "ti",
            "abstract": "abs",
            "authors": "au",
            "author": "au",
            "category": "cat",
        }
        # 兼容常见字段别名，但最终仍收敛到本地 OAI 明确支持的最小字段集合。
        field = aliases.get(field, field)
        supported_fields = {"id", "cat", "ti", "abs", "au", "all"}
        if field not in supported_fields:
            raise UnsupportedLocalArxivQuery(
                f"本地 OAI 镜像不支持字段 `{field}`，请改用支持的查询子集。",
                query=original_query,
                reason=f"unsupported_field:{field}",
            )

        if not raw_value:
            raise UnsupportedLocalArxivQuery(
                "本地 OAI 镜像不支持空字段查询。",
                query=original_query,
                reason="empty_field_query",
            )
        if "*" in raw_value or "?" in raw_value:
            raise UnsupportedLocalArxivQuery(
                "本地 OAI 镜像不支持通配符查询，避免低精度误召回。",
                query=original_query,
                reason="unsupported_wildcard_query",
            )

        # phrase 和普通 token 查询在 FTS 编译阶段有不同语义，这里先把标记保留下来。
        phrase = raw_value.startswith('"') and raw_value.endswith('"') and len(raw_value) >= 2
        value = raw_value[1:-1] if phrase else raw_value
        value = value.replace('\\"', '"').replace("\\\\", "\\").strip()

        if field == "id":
            return {"type": "id", "value": value}
        if field == "cat":
            return {"type": "category", "value": value}
        return {"type": "text", "field": field, "value": value, "phrase": phrase}
