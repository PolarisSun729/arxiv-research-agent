"""arXiv OAI-PMH XML 解析工具模块。

该模块负责解析 arXiv OAI-PMH 接口返回的 XML 响应，提取论文元数据、
作者信息、分类等字段，并提供标准化和规范化工具函数。
"""

import json
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class ArxivOaiXmlParser:
    """arXiv OAI-PMH XML 解析器。"""

    def parse_paper_metadata(self, record: ET.Element, paper_elem: ET.Element) -> Optional[Dict[str, Any]]:
        """把单条 OAI-PMH record 解析成内部统一论文结构。"""
        try:
            raw_arxiv_id = self._extract_text(paper_elem, "id")
            arxiv_id = self.normalize_arxiv_id(raw_arxiv_id)
            title = self.normalize_whitespace(self._extract_text(paper_elem, "title"))
            abstract = self.normalize_whitespace(self._extract_text(paper_elem, "abstract"))
            created = self.normalize_whitespace(self._extract_text(paper_elem, "created"))
            updated = self.normalize_whitespace(
                self._extract_text(paper_elem, "updated") or self._extract_text(paper_elem, "updateDate")
            )
            categories_text = self.normalize_whitespace(self._extract_text(paper_elem, "categories"))
            categories = self.split_categories(categories_text)
            primary_category = self._extract_primary_category(paper_elem)
            authors = self._extract_authors(paper_elem)
            oai_datestamp = self.normalize_whitespace(self._extract_text(record, "datestamp"))

            # authors/categories 继续存成 JSON 字符串，兼容现有数据库结构和旧调用方读取习惯。
            return {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "authors": json.dumps(authors, ensure_ascii=False),
                "categories": json.dumps(categories, ensure_ascii=False),
                "categories_list": categories,
                "primary_category": primary_category,
                "created": created,
                "updated": updated,
                "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
                "oai_datestamp": oai_datestamp,
            }
        except Exception as exc:
            logger.error("Failed to parse OAI-PMH record: %s", exc)
            return None

    def _extract_authors(self, paper_elem: ET.Element) -> List[str]:
        """从 OAI 元数据中提取作者列表，兼容多种 author 结构。"""
        authors_parent = self._find_first_child(paper_elem, "authors")
        # 优先限定在 authors 节点下遍历；缺失时再退回整棵 paper 子树，兼容历史 XML 变体。
        search_root = authors_parent if authors_parent is not None else paper_elem
        authors: List[str] = []
        for node in search_root.iter():
            if self._local_name(node.tag) != "author":
                continue
            name_parts: List[str] = []
            forenames = self.normalize_whitespace(self._extract_text(node, "forenames"))
            keyname = self.normalize_whitespace(self._extract_text(node, "keyname"))
            full_name = self.normalize_whitespace(self._extract_text(node, "name"))
            if full_name:
                # 某些 XML 直接提供完整姓名；优先使用它，避免再拼接出重复空格或错序。
                authors.append(full_name)
                continue
            if forenames:
                name_parts.append(forenames)
            if keyname:
                name_parts.append(keyname)
            if name_parts:
                authors.append(" ".join(name_parts))
        return [author for author in authors if author]

    def _extract_primary_category(self, paper_elem: ET.Element) -> str:
        """提取主分类，兼容不同 OAI 扩展字段命名。"""
        primary_category = self._find_first_descendant(paper_elem, "primary_category")
        if primary_category is None:
            # 不同 arXiv/OAI 元数据版本里字段名可能不同，这里同时兼容 snake/camel 两种写法。
            primary_category = self._find_first_descendant(paper_elem, "primaryCategory")
        if primary_category is None:
            return ""
        term = primary_category.attrib.get("term", "").strip()
        if term:
            # term 属性通常比节点文本更规范，优先使用它作为主分类真值。
            return term
        return self.normalize_whitespace(primary_category.text or "")

    def extract_resumption_token(self, root: ET.Element) -> str:
        """从 OAI-PMH 响应中提取翻页续传 token。"""
        token_elem = self._find_first_descendant(root, "resumptionToken")
        if token_elem is None or token_elem.text is None:
            return ""
        # 空串由上层统一解释为"没有下一页"，避免把 XML 细节暴露到同步主循环。
        return token_elem.text.strip()

    def extract_oai_error(self, root: ET.Element) -> str:
        """提取 OAI-PMH 错误节点，拼成可直接记录的错误文本。"""
        error_elem = self._find_first_descendant(root, "error")
        if error_elem is None:
            return ""
        code = error_elem.attrib.get("code", "").strip()
        text = self.normalize_whitespace(error_elem.text or "")
        if code and text:
            # 把 code 和文本合并成单行字符串，方便日志与告警直接展示。
            return f"{code}: {text}"
        return code or text

    def split_categories(self, categories_text: str) -> List[str]:
        """把分类字符串拆成分类列表。"""
        if not categories_text:
            return []
        # OAI categories 常见为空格分隔，但也兼容逗号分隔的历史数据。
        raw_items = re.split(r"[\s,]+", categories_text.strip())
        return [item for item in (part.strip() for part in raw_items) if item]

    def normalize_arxiv_id(self, raw_value: str) -> str:
        """把 OAI / URL / 带版本号的原始 ID 规整成统一 arXiv ID。"""
        value = self.normalize_whitespace(raw_value)
        if not value:
            return ""
        # 去掉常见前缀和 URL 包装，保证后续主键、链接和查询都基于同一 ID 形态。
        value = re.sub(r"^oai:arXiv\.org:", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^arxiv:", "", value, flags=re.IGNORECASE)
        if "arxiv.org/abs/" in value.lower():
            value = value.rsplit("/", 1)[-1]
        value = value.split("?", 1)[0].strip()
        # 版本号不参与主键归一化，避免同一论文不同版本重复入库。
        value = re.sub(r"v\d+$", "", value)
        return value

    def normalize_whitespace(self, value: str) -> str:
        """折叠连续空白字符，得到稳定的展示和入库文本。"""
        if not value:
            return ""
        # 标题、摘要、作者名都可能带换行或多空格，统一规整后更利于检索和比较。
        return " ".join(value.split()).strip()

    def find_first_child(self, parent: ET.Element, local_name: str) -> Optional[ET.Element]:
        """在直接子节点里查找第一个指定本地名的元素。"""
        return self._find_first_child(parent, local_name)

    def find_first_element_child(self, parent: ET.Element) -> Optional[ET.Element]:
        """返回第一个真正的元素子节点，跳过注释等非元素节点。"""
        for child in list(parent):
            if isinstance(child.tag, str):
                return child
        # metadata 节点可能为空；这里显式返回 None 供上层统计缺失原因。
        return None

    def _find_first_child(self, parent: ET.Element, local_name: str) -> Optional[ET.Element]:
        """在直接子节点里查找第一个指定本地名的元素。"""
        for child in list(parent):
            if self._local_name(child.tag) == local_name:
                return child
        # 返回 None 让上层自行决定是"字段可缺省"还是"解析失败"。
        return None

    def _find_first_descendant(self, parent: ET.Element, local_name: str) -> Optional[ET.Element]:
        """在整棵子树里查找第一个指定本地名的元素。"""
        for node in parent.iter():
            if self._local_name(node.tag) == local_name:
                return node
        # 统一使用 None 表示未命中，减少辅助函数之间的异常分支。
        return None

    def _extract_text(self, parent: ET.Element, local_name: str) -> str:
        """提取第一个命中节点的文本内容；缺失时返回空串。"""
        element = self._find_first_descendant(parent, local_name)
        if element is None or element.text is None:
            return ""
        # 解析阶段直接做 strip，避免调用方反复处理首尾空白。
        return element.text.strip()

    def _local_name(self, tag: Any) -> str:
        """剥离 XML 命名空间前缀，返回纯本地标签名。"""
        if not isinstance(tag, str):
            return ""
        if "}" in tag:
            # ElementTree 会把命名空间编码进 `{namespace}tag` 形式，这里统一去掉前缀。
            return tag.rsplit("}", 1)[-1]
        return tag
