import json
from typing import Any


class JsonFieldCodec:
    """提供 SQLite JSON 文本字段编解码，集中保留历史空串和普通字符串兼容语义。"""

    @staticmethod
    def _serialize_json_field(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        # SQLite 仍以 TEXT 保存 JSON；保留中文原文，方便排查画像和问答调试字段。
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _deserialize_json_field(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 历史字段可能存的是普通字符串；解析失败时原样返回，避免迁移期误清空旧值。
            return value

    @staticmethod
    def _deserialize_paper_db_value(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return ""
        if text.startswith("[") or text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # 论文 authors/categories 的旧数据可能是普通字符串；失败时保持原值供上层兜底。
                return value
        return value
