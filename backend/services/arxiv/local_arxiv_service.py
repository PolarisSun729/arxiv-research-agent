from __future__ import annotations

from typing import Any, Dict, List, Optional


class LocalArxivService:
    """
    Local arXiv service backed by backend/06-database/arxiv_oai.db.

    The constructor keeps a data_path argument only for backward compatibility;
    it is ignored because local reads now come from the SQLite OAI database.
    """

    def __init__(
        self,
        data_path: str = "",
        db_path: Optional[str] = None,
        check_same_thread: Optional[bool] = None,
    ):
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
        return self.search_advanced(author=author, **kwargs)

    def search_by_title(self, title: str, **kwargs) -> Dict[str, Any]:
        return self.search_advanced(title=title, **kwargs)

    def search_by_category(self, category: str, **kwargs) -> Dict[str, Any]:
        return self.search_advanced(category=category, **kwargs)

    def search_by_abstract(self, abstract: str, **kwargs) -> Dict[str, Any]:
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
        return [
            {"prefix": "ti", "field": "Title", "description": "搜索论文标题"},
            {"prefix": "au", "field": "Author", "description": "搜索作者姓名"},
            {"prefix": "abs", "field": "Abstract", "description": "搜索摘要"},
            {"prefix": "cat", "field": "Subject Category", "description": "搜索学科分类"},
            {"prefix": "all", "field": "All Fields", "description": "搜索所有字段"},
        ]

    def get_subject_categories(self) -> List[Dict[str, str]]:
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
