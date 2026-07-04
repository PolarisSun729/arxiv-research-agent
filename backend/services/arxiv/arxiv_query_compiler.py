"""arXiv 查询 SQL 编译器模块。

该模块负责将查询 AST 编译为可执行的 SQLite SQL 语句，
支持 FTS5 全文检索和精确字段过滤。
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from services.arxiv.local_oai_search_contract import (
    LocalArxivSearchIndexUnavailable,
    UnsupportedLocalArxivQuery,
)

logger = logging.getLogger(__name__)


class ArxivQueryCompiler:
    """arXiv 查询编译器，将 AST 编译为 SQL。"""

    def compile(
        self,
        node: Optional[Dict[str, Any]],
        *,
        fts5_available: bool,
        original_query: str,
    ) -> Tuple[str, List[Any], bool]:
        """把查询 AST 编译成 SQL WHERE 片段，返回是否消费了 FTS 文本索引。

        返回值:
            Tuple[str, List[Any], bool]: (SQL WHERE 子句, 参数列表, 是否使用了 FTS)
        """
        if not node or node.get("type") == "empty":
            return "", [], False

        node_type = node.get("type")
        if node_type in {"and", "or"}:
            # 复合节点先递归编译子节点，再在 SQL 层按原布尔关系拼接。
            compiled = [
                self.compile(child, fts5_available=fts5_available, original_query=original_query)
                for child in node.get("children") or []
                if child
            ]
            clauses = [item[0] for item in compiled if item[0]]
            params = [param for item in compiled for param in item[1]]
            uses_fts = any(item[2] for item in compiled)
            if not clauses:
                return "", [], uses_fts
            joiner = " AND " if node_type == "and" else " OR "
            return "(" + joiner.join(clauses) + ")", params, uses_fts

        if node_type == "andnot":
            children = [child for child in node.get("children") or [] if child]
            if not children:
                return "", [], False
            # ANDNOT 只把第一个子句当正条件，后续子句全部编译成 NOT (...)。
            left_sql, left_params, left_fts = self.compile(
                children[0],
                fts5_available=fts5_available,
                original_query=original_query,
            )
            negative_parts = [
                self.compile(child, fts5_available=fts5_available, original_query=original_query)
                for child in children[1:]
            ]
            clauses = [left_sql] if left_sql else []
            params = list(left_params)
            uses_fts = left_fts
            for sql, item_params, item_fts in negative_parts:
                if sql:
                    clauses.append(f"NOT ({sql})")
                    params.extend(item_params)
                uses_fts = uses_fts or item_fts
            return "(" + " AND ".join(clauses) + ")", params, uses_fts

        if node_type == "id":
            # arXiv ID 本身就是主键级过滤条件，直接等值命中最精确。
            return "p.arxiv_id = ?", [node.get("value")], False
        if node_type == "category":
            return (
                "EXISTS (SELECT 1 FROM arxiv_oai_paper_categories c "
                "WHERE c.arxiv_id = p.arxiv_id AND c.category = ?)"
            ), [node.get("value")], False
        if node_type == "date":
            # 日期查询在多种时间字段之间做一致兜底，尽量贴近实际可用时间语义。
            return (
                "datetime(COALESCE(p.updated, p.created, p.oai_datestamp, p.fetched_at)) "
                "BETWEEN datetime(?) AND datetime(?)"
            ), [node.get("start"), node.get("end")], False
        if node_type == "text":
            if not fts5_available:
                raise LocalArxivSearchIndexUnavailable(
                    "本地 OAI 镜像文本索引不可用，无法执行 ti/abs/au/all 文本检索。",
                    query=original_query,
                    reason="fts5_unavailable",
                )
            field_map = {"ti": "title", "abs": "abstract", "au": "authors", "all": "all_text"}
            fts_field = field_map.get(str(node.get("field") or "all"), "all_text")
            # MATCH 语句始终只打到 FTS 表，避免退回 LIKE 造成语义和性能不可控。
            fts_query = self._build_fts_query(
                str(node.get("value") or ""),
                phrase=bool(node.get("phrase")),
                original_query=original_query,
            )
            return (
                "p.arxiv_id IN (SELECT arxiv_id FROM arxiv_oai_papers_fts "
                "WHERE arxiv_oai_papers_fts MATCH ?)"
            ), [f"{fts_field}:({fts_query})"], True

        raise UnsupportedLocalArxivQuery(
            "本地 OAI 镜像无法识别该查询节点。",
            query=original_query,
            reason=f"unsupported_node:{node_type}",
        )

    def _build_fts_query(self, value: str, *, phrase: bool, original_query: str) -> str:
        """把文本查询编译成 FTS5 语法；只使用 token/phrase，禁止子串模糊匹配。"""
        # 只抽取英文和数字 token，主动拒绝"空 token / 纯符号"输入，避免 MATCH 语法漂移。
        tokens = re.findall(r"[A-Za-z0-9]+", str(value or "").lower())
        if not tokens:
            raise UnsupportedLocalArxivQuery(
                "本地 OAI 镜像文本检索需要至少一个英文/数字 token。",
                query=original_query,
                reason="empty_fts_tokens",
            )
        if phrase:
            # phrase 查询保持原始顺序，适合 title/abstract 中的连续短语检索。
            return '"' + " ".join(tokens) + '"'
        # 非短语查询按 AND 拼接，显式要求所有 token 命中，保证高精度。
        return " AND ".join(tokens)
