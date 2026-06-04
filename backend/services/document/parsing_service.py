"""文档解析服务模块。

该模块把页级结构数据进一步整理为不同粒度的内容视图，例如全文流、逐页
内容、按标题组织的章节内容以及文本/表格混合内容。它位于加载与切分之间，
用于把上游提取结果转成更适合下游消费的结构化表达。
"""

import logging
from typing import List
from datetime import datetime

logger = logging.getLogger(__name__)

class ParsingService:
    """
    PDF 文档解析服务。

    这里的“解析”更偏向结构化整理，而不是底层 PDF 提取：
    上游已经给出 page_map，本服务负责把页级文本重组为后续流程
    更容易消费的形态，例如整页内容、章节内容或文本/表格混合内容。
    """

    def parse_pdf(self, text: str, method: str, metadata: dict, page_map: list = None) -> dict:
        """
        使用指定策略把 page_map 组织成统一的结构化文档对象。

        参数:
            text (str): 保留的原始文本入参。当前实现主要依赖 page_map，
                该参数保留是为了和上层调用约定兼容。
            method (str): 解析方法，可选值包括 all_text、by_pages、
                by_titles、text_and_tables。
            metadata (dict): 文档元数据，至少应包含文件名等来源信息。
            page_map (list): 页级结构数据列表，每一项至少包含 page 和 text。

        返回:
            dict: 统一格式的解析结果，包含文档级 metadata 与结构化 content。

        异常:
            ValueError: 当 page_map 为空或 method 不受支持时抛出。
        """
        try:
            if not page_map:
                raise ValueError("Page map is required for parsing.")

            parsed_content = []
            total_pages = len(page_map)

            # 这里集中做方法分发，保证外部调用方始终拿到统一的数据壳。
            if method == "all_text":
                parsed_content = self._parse_all_text(page_map)
            elif method == "by_pages":
                parsed_content = self._parse_by_pages(page_map)
            elif method == "by_titles":
                parsed_content = self._parse_by_titles(page_map)
            elif method == "text_and_tables":
                parsed_content = self._parse_text_and_tables(page_map)
            else:
                raise ValueError(f"Unsupported parsing method: {method}")

            # 文档级元数据在这里统一补齐，避免每个策略重复处理公共字段。
            document_data = {
                "metadata": {
                    "filename": metadata.get("filename", ""),
                    "total_pages": total_pages,
                    "parsing_method": method,
                    "timestamp": datetime.now().isoformat()
                },
                "content": parsed_content
            }

            return document_data

        except Exception as e:
            logger.error(f"Error in parse_pdf: {str(e)}")
            raise

    def _parse_all_text(self, page_map: list) -> list:
        """
        以“全文顺序流”的方式返回所有页面文本。

        这里不会尝试识别章节或页面边界之外的结构，只是把每页内容按
        原顺序打平成一组 Text 记录，适合需要最大保真原文的场景。

        返回:
            list: 由文本节点组成的列表，每个节点保留来源页码。
        """
        # 这里保持最小加工，适合需要最大限度保留原文顺序的下游流程。
        return [{
            "type": "Text",
            "content": page["text"],
            "page": page["page"]
        } for page in page_map]

    def _parse_by_pages(self, page_map: list) -> list:
        """
        逐页返回内容，显式保留页面边界。

        返回:
            list: Page 结构列表，每项对应 PDF 中的一页。
        """
        parsed_content = []
        for page in page_map:
            # 逐页封装可以最大限度保留和原文页面的一一对应关系。
            parsed_content.append({
                "type": "Page",
                "page": page["page"],
                "content": page["text"]
            })
        return parsed_content

    def _parse_by_titles(self, page_map: list) -> list:
        """
        使用简化标题启发式把文档组织为章节。

        当前规则比较轻量：长度较短且全部大写的行会被视作标题。
        这不是通用标题检测器，但对学术 PDF、报告类文档足够实用。

        返回:
            list: 章节列表，每项包含标题、正文和定位页码。
        """
        parsed_content = []
        current_title = None
        current_content = []

        for page in page_map:
            lines = page["text"].split('\n')
            for line in lines:
                # 使用非常保守的标题规则，尽量减少把普通正文误识别为章节标题。
                if len(line.strip()) < 60 and line.isupper():
                    if current_title:
                        # 遇到新标题时，先把上一段章节落盘，保持章节顺序稳定。
                        parsed_content.append({
                            "type": "section",
                            "title": current_title,
                            "content": '\n'.join(current_content),
                            "page": page["page"]
                        })
                    current_title = line.strip()
                    current_content = []
                else:
                    current_content.append(line)

        # 循环结束后手动补上最后一个章节，避免尾部内容丢失。
        if current_title:
            parsed_content.append({
                "type": "section",
                "title": current_title,
                "content": '\n'.join(current_content),
                "page": page["page"]
            })

        return parsed_content

    def _parse_text_and_tables(self, page_map: list) -> list:
        """
        用简单规则区分文本页与表格页。

        当前版本不直接调用专业表格解析器，而是使用分隔符特征作为
        低成本启发式判断，适合先粗分类型，再交给下游更细的处理逻辑。

        返回:
            list: 每页一个节点，类型可能为 text 或 table。
        """
        parsed_content = []
        for page in page_map:
            # 这里只做轻量识别：命中常见表格分隔符就先按 table 归类。
            content = page["text"]
            if '|' in content or '\t' in content:
                parsed_content.append({
                    "type": "table",
                    "content": content,
                    "page": page["page"]
                })
            else:
                parsed_content.append({
                    "type": "text",
                    "content": content,
                    "page": page["page"]
                })
        return parsed_content 
