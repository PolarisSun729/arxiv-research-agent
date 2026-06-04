"""本地 arXiv 服务模块。

该模块提供一个面向业务层的轻量封装，使调用方可以继续沿用“本地 arXiv
服务”的接口语义，而底层实际读取来源已经切换为 SQLite 版的 OAI 数据库。
这样既兼容旧调用方式，也避免上层直接依赖数据库服务的内部实现细节。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class LocalArxivService:
    """
    本地 arXiv 检索服务。

    当前实现底层依赖 ``ArxivOaiDatabaseService``，因此这里更像一个兼容层：
    一方面保留旧代码熟悉的调用入口，另一方面把实际搜索能力委托给 SQLite
    OAI 数据库服务完成。
    """

    def __init__(
        self,
        data_path: str = "",
        db_path: Optional[str] = None,
        check_same_thread: Optional[bool] = None,
    ):
        """初始化本地 arXiv 服务。

        参数:
            data_path (str): 兼容旧版本保留的本地数据路径参数；当前不再实际使用。
            db_path (Optional[str]): SQLite 数据库路径；为空时使用默认配置。
            check_same_thread (Optional[bool]): SQLite 线程检查开关。

        返回:
            None
        """
        self.data_path = data_path
        from services.arxiv.arxiv_oai_service import ArxivOaiDatabaseService

        self.database_service = ArxivOaiDatabaseService(db_path=db_path, check_same_thread=check_same_thread)

    def search(
        self,
        search_query: str = "",
        id_list: Optional[List[str]] = None,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> Dict[str, Any]:
        """执行基础 arXiv 搜索，并直接复用数据库服务结果。

        参数:
            search_query (str): 原始查询字符串。
            id_list (Optional[List[str]]): 论文 ID 精确匹配列表。
            max_results (int): 最大返回条数。
            start (int): 分页起始偏移量。
            sort_by (str): 排序字段。
            sort_order (str): 排序方向。

        返回:
            Dict[str, Any]: 标准化搜索结果。
        """
        return self.database_service.search(
            search_query=search_query,
            id_list=id_list,
            max_results=max_results,
            start=start,
            sort_by=sort_by,
            sort_order=sort_order,
        )

    def search_papers(
        self,
        search_query: str = "",
        id_list: Optional[List[str]] = None,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        submitted_days_ago: Optional[int] = None,
    ) -> Dict[str, Any]:
        """执行论文搜索，并可附加最近提交天数过滤。

        参数:
            search_query (str): 查询字符串。
            id_list (Optional[List[str]]): 论文 ID 列表。
            max_results (int): 最大返回条数。
            start (int): 分页偏移量。
            sort_by (str): 排序字段。
            sort_order (str): 排序方向。
            submitted_days_ago (Optional[int]): 仅保留最近若干天提交的论文。

        返回:
            Dict[str, Any]: 搜索结果字典。
        """
        return self.database_service.search_papers(
            search_query=search_query,
            id_list=id_list,
            max_results=max_results,
            start=start,
            sort_by=sort_by,
            sort_order=sort_order,
            submitted_days_ago=submitted_days_ago,
        )

    def search_by_author(self, author: str, **kwargs) -> Dict[str, Any]:
        """按作者字段发起高级搜索。"""
        return self.search_advanced(author=author, **kwargs)

    def search_by_title(self, title: str, **kwargs) -> Dict[str, Any]:
        """按标题字段发起高级搜索。"""
        return self.search_advanced(title=title, **kwargs)

    def search_by_category(self, category: str, **kwargs) -> Dict[str, Any]:
        """按学科分类字段发起高级搜索。"""
        return self.search_advanced(category=category, **kwargs)

    def search_by_abstract(self, abstract: str, **kwargs) -> Dict[str, Any]:
        """按摘要字段发起高级搜索。"""
        return self.search_advanced(abstract=abstract, **kwargs)

    def search_advanced(
        self,
        title: Optional[str] = None,
        author: Optional[str] = None,
        abstract: Optional[str] = None,
        category: Optional[str] = None,
        comment: Optional[str] = None,
        journal_ref: Optional[str] = None,
        report_number: Optional[str] = None,
        operator: str = "AND",
        id_list: Optional[List[str]] = None,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        submitted_days_ago: Optional[int] = None,
    ) -> Dict[str, Any]:
        """执行多字段高级搜索。

        参数:
            title (Optional[str]): 标题查询词。
            author (Optional[str]): 作者查询词。
            abstract (Optional[str]): 摘要查询词。
            category (Optional[str]): 分类查询词。
            comment (Optional[str]): 备注查询词。
            journal_ref (Optional[str]): 期刊引用查询词。
            report_number (Optional[str]): 报告编号查询词。
            operator (str): 多字段组合逻辑操作符。
            id_list (Optional[List[str]]): 精确匹配 ID 列表。
            max_results (int): 最大返回条数。
            start (int): 分页偏移量。
            sort_by (str): 排序字段。
            sort_order (str): 排序方向。
            submitted_days_ago (Optional[int]): 最近提交天数过滤。

        返回:
            Dict[str, Any]: 高级搜索结果。
        """
        return self.database_service.search_advanced(
            title=title,
            author=author,
            abstract=abstract,
            category=category,
            comment=comment,
            journal_ref=journal_ref,
            report_number=report_number,
            operator=operator,
            id_list=id_list,
            max_results=max_results,
            start=start,
            sort_by=sort_by,
            sort_order=sort_order,
            submitted_days_ago=submitted_days_ago,
        )

    def get_available_fields(self) -> List[Dict[str, str]]:
        """返回前端可展示的 arXiv 常用搜索字段清单。

        返回:
            List[Dict[str, str]]: 字段前缀、名称与说明列表。
        """
        return [
            {"prefix": "ti", "field": "Title", "description": "搜索论文标题"},
            {"prefix": "au", "field": "Author", "description": "搜索作者姓名"},
            {"prefix": "abs", "field": "Abstract", "description": "搜索摘要"},
            {"prefix": "cat", "field": "Subject Category", "description": "搜索学科分类"},
            {"prefix": "all", "field": "All Fields", "description": "搜索所有字段"},
        ]

    def get_subject_categories(self) -> List[Dict[str, str]]:
        """返回预置的常见 arXiv 学科分类列表。

        返回:
            List[Dict[str, str]]: 分类代码与名称列表。
        """
        return [
            {"code": "cs.AI", "name": "Artificial Intelligence"},
            {"code": "cs.AR", "name": "Hardware Architecture"},
            {"code": "cs.CC", "name": "Computational Complexity"},
            {"code": "cs.CE", "name": "Computational Engineering, Finance, and Science"},
            {"code": "cs.CG", "name": "Computational Geometry"},
            {"code": "cs.CL", "name": "Computation and Language"},
            {"code": "cs.CR", "name": "Cryptography and Security"},
            {"code": "cs.CV", "name": "Computer Vision and Pattern Recognition"},
            {"code": "cs.CY", "name": "Computers and Society"},
            {"code": "cs.DB", "name": "Databases"},
            {"code": "cs.DC", "name": "Distributed, Parallel, and Cluster Computing"},
            {"code": "cs.DL", "name": "Digital Libraries"},
            {"code": "cs.DM", "name": "Discrete Mathematics"},
            {"code": "cs.DS", "name": "Data Structures and Algorithms"},
            {"code": "cs.ET", "name": "Emerging Technologies"},
            {"code": "cs.FL", "name": "Formal Languages and Automata Theory"},
            {"code": "cs.GL", "name": "General Literature"},
            {"code": "cs.GR", "name": "Graphics"},
            {"code": "cs.GT", "name": "Computer Science and Game Theory"},
            {"code": "cs.HC", "name": "Human-Computer Interaction"},
            {"code": "cs.IR", "name": "Information Retrieval"},
            {"code": "cs.IT", "name": "Information Theory"},
            {"code": "cs.LG", "name": "Machine Learning"},
            {"code": "cs.LO", "name": "Logic in Computer Science"},
            {"code": "cs.MA", "name": "Multiagent Systems"},
            {"code": "cs.MM", "name": "Multimedia"},
            {"code": "cs.MS", "name": "Mathematical Software"},
            {"code": "cs.NA", "name": "Numerical Analysis"},
            {"code": "cs.NE", "name": "Neural and Evolutionary Computing"},
            {"code": "cs.NI", "name": "Networking"},
            {"code": "cs.OH", "name": "Other Computer Science"},
            {"code": "cs.OS", "name": "Operating Systems"},
            {"code": "cs.PF", "name": "Performance"},
            {"code": "cs.PL", "name": "Programming Languages"},
            {"code": "cs.RO", "name": "Robotics"},
            {"code": "cs.SC", "name": "Symbolic Computation"},
            {"code": "cs.SD", "name": "Sound"},
            {"code": "cs.SE", "name": "Software Engineering"},
            {"code": "cs.SI", "name": "Social and Information Networks"},
            {"code": "cs.SY", "name": "Systems and Control"},
            {"code": "stat.ML", "name": "Machine Learning"},
            {"code": "physics.quant-ph", "name": "Quantum Physics"},
            {"code": "hep-th", "name": "High Energy Physics - Theory"},
            {"code": "hep-ph", "name": "High Energy Physics - Phenomenology"},
            {"code": "astro-ph", "name": "Astrophysics"},
        ]
