from typing import List, Dict, Any, Optional, Union
import logging
import json
import os
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class LocalArxivService:
    """
    本地 arXiv 数据集服务类
    读取 Kaggle 下载的 arXiv JSON 数据集，提供搜索功能
    数据集格式: https://www.kaggle.com/datasets/Cornell-University/arxiv
    """
    
    def __init__(self, data_path: str = "../07-local-arxiv/arxiv-metadata-oai-snapshot.json"):
        """
        初始化本地 arXiv 服务
        
        Args:
            data_path (str): JSON 数据集文件路径
        """
        self.data_path = data_path
        self.data = []
        self._load_data()
    
    def _load_data(self):
        """
        加载 JSON 数据集
        Kaggle arXiv 数据集每行是一个 JSON 对象
        """
        if not os.path.exists(self.data_path):
            logger.warning(f"Dataset file not found: {self.data_path}")
            logger.warning("Please download the dataset from: https://www.kaggle.com/datasets/Cornell-University/arxiv")
            return
        
        logger.info(f"Loading arXiv dataset from: {self.data_path}")
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            paper = json.loads(line)
                            self.data.append(paper)
                        except json.JSONDecodeError:
                            continue
            
            logger.info(f"Loaded {len(self.data)} papers from dataset")
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
    
    def _parse_categories(self, categories_str: str) -> List[str]:
        """
        解析分类字符串，支持空格分隔的多个分类
        
        Args:
            categories_str (str): 分类字符串，如 "hep-ph cs.AI"
            
        Returns:
            List[str]: 分类列表
        """
        return [cat.strip() for cat in categories_str.split() if cat.strip()]
    
    def _build_paper_response(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        将原始论文数据转换为统一的响应格式
        
        Args:
            paper (Dict[str, Any]): 原始论文数据
            
        Returns:
            Dict[str, Any]: 统一格式的论文信息
        """
        arxiv_id = paper.get('id', '')
        
        return {
            "id": f"http://arxiv.org/abs/{arxiv_id}",
            "title": paper.get('title', '').replace('\n', ' ').strip(),
            "summary": paper.get('abstract', '').replace('\n', ' ').strip(),
            "published": paper.get('update_date', ''),
            "updated": paper.get('update_date', ''),
            "authors": [paper.get('authors', '')],
            "categories": self._parse_categories(paper.get('categories', '')),
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
            "journal_reference": paper.get('journal-ref', ''),
            "comment": paper.get('comments', ''),
            "doi": paper.get('doi', ''),
            "arxiv_id": arxiv_id,
            "submitter": paper.get('submitter', ''),
            "versions": paper.get('versions', []),
            "authors_parsed": paper.get('authors_parsed', [])
        }
    
    def _match_query(self, paper: Dict[str, Any], query: str, field: str = "all") -> bool:
        """
        检查论文是否匹配查询条件
        
        Args:
            paper (Dict[str, Any]): 论文数据
            query (str): 查询字符串
            field (str): 搜索字段，可选值: all, title, authors, abstract, category
            
        Returns:
            bool: 是否匹配
        """
        query = query.lower().strip()
        
        if field == "title" or field == "all":
            title = paper.get('title', '').lower()
            if query in title:
                return True
        
        if field == "authors" or field == "all":
            authors = paper.get('authors', '').lower()
            if query in authors:
                return True
        
        if field == "abstract" or field == "all":
            abstract = paper.get('abstract', '').lower()
            if query in abstract:
                return True
        
        if field == "category" or field == "all":
            categories = paper.get('categories', '').lower()
            if query in categories:
                return True
        
        return False
    
    def search_papers(self, 
                     search_query: str = "",
                     id_list: Optional[List[str]] = None,
                     max_results: int = 10, 
                     start: int = 0,
                     sort_by: str = "relevance",
                     sort_order: str = "descending",
                     submitted_days_ago: Optional[int] = None) -> Dict[str, Any]:
        """
        搜索 arXiv 论文（本地数据集）
        
        Args:
            search_query (str): 搜索查询字符串
            id_list (Optional[List[str]]): arXiv 论文ID列表
            max_results (int): 返回结果的最大数量
            start (int): 起始索引
            sort_by (str): 排序方式（暂时仅支持 relevance）
            sort_order (str): 排序顺序（暂时仅支持 descending）
            submitted_days_ago (Optional[int]): 提交日期过滤（暂不支持）
            
        Returns:
            Dict[str, Any]: 搜索结果
        """
        logger.info(f"Searching local arXiv dataset with query: '{search_query}', max_results: {max_results}")
        
        results = []
        
        if id_list and len(id_list) > 0:
            id_set = set(id_list)
            for paper in self.data:
                if paper.get('id') in id_set:
                    results.append(paper)
                    if len(results) >= max_results:
                        break
        elif search_query:
            parts = search_query.split(':')
            if len(parts) == 2:
                field, query = parts[0].strip(), parts[1].strip()
                field_map = {
                    "ti": "title",
                    "au": "authors", 
                    "abs": "abstract",
                    "cat": "category",
                    "title": "title",
                    "authors": "authors",
                    "abstract": "abstract",
                    "category": "category"
                }
                field = field_map.get(field, "all")
            else:
                field = "all"
                query = search_query
            
            for paper in self.data:
                if self._match_query(paper, query, field):
                    results.append(paper)
                    if len(results) >= max_results + start:
                        break
        else:
            results = self.data[:max_results + start]
        
        paginated_results = results[start:start + max_results]
        papers = [self._build_paper_response(p) for p in paginated_results]
        
        result = {
            "query": search_query,
            "id_list": id_list,
            "total_results": len(results),
            "start_index": start,
            "items_per_page": len(papers),
            "papers": papers,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Found {len(papers)} papers out of {result['total_results']} total results")
        
        return result
    
    def search_by_author(self, author: str, **kwargs) -> Dict[str, Any]:
        """
        按作者搜索论文
        
        Args:
            author (str): 作者姓名
            **kwargs: 其他搜索参数
            
        Returns:
            Dict[str, Any]: 搜索结果
        """
        query = f"au:{author}"
        return self.search_papers(query, **kwargs)
    
    def search_by_title(self, title: str, **kwargs) -> Dict[str, Any]:
        """
        按标题搜索论文
        
        Args:
            title (str): 标题关键词
            **kwargs: 其他搜索参数
            
        Returns:
            Dict[str, Any]: 搜索结果
        """
        query = f"ti:{title}"
        return self.search_papers(query, **kwargs)
    
    def search_by_category(self, category: str, **kwargs) -> Dict[str, Any]:
        """
        按学科分类搜索论文
        
        Args:
            category (str): 学科分类代码
            **kwargs: 其他搜索参数
            
        Returns:
            Dict[str, Any]: 搜索结果
        """
        query = f"cat:{category}"
        return self.search_papers(query, **kwargs)
    
    def search_by_abstract(self, abstract: str, **kwargs) -> Dict[str, Any]:
        """
        按摘要搜索论文
        
        Args:
            abstract (str): 摘要关键词
            **kwargs: 其他搜索参数
            
        Returns:
            Dict[str, Any]: 搜索结果
        """
        query = f"abs:{abstract}"
        return self.search_papers(query, **kwargs)
    
    def search_advanced(self, 
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
                       submitted_days_ago: Optional[int] = None) -> Dict[str, Any]:
        """
        高级搜索，支持多条件组合
        
        Args:
            title (Optional[str]): 标题关键词
            author (Optional[str]): 作者姓名
            abstract (Optional[str]): 摘要关键词
            category (Optional[str]): 学科分类代码
            operator (str): 逻辑操作符（仅支持 AND）
            **kwargs: 其他搜索参数
            
        Returns:
            Dict[str, Any]: 搜索结果
        """
        if id_list and len(id_list) > 0:
            return self.search_papers(id_list=id_list, max_results=max_results, start=start)
        
        results = self.data
        filters = []
        
        if title:
            filters.append(lambda p: title.lower() in p.get('title', '').lower())
        if author:
            filters.append(lambda p: author.lower() in p.get('authors', '').lower())
        if abstract:
            filters.append(lambda p: abstract.lower() in p.get('abstract', '').lower())
        if category:
            filters.append(lambda p: category.lower() in p.get('categories', '').lower())
        if comment:
            filters.append(lambda p: comment.lower() in p.get('comments', '').lower())
        if journal_ref:
            filters.append(lambda p: journal_ref.lower() in p.get('journal-ref', '').lower())
        
        if filters:
            for filt in filters:
                results = [p for p in results if filt(p)]
        
        results = results[start:start + max_results]
        papers = [self._build_paper_response(p) for p in results]
        
        result = {
            "query": f"advanced_search",
            "id_list": id_list,
            "total_results": len(results),
            "start_index": start,
            "items_per_page": len(papers),
            "papers": papers,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def get_available_fields(self) -> List[Dict[str, str]]:
        """
        获取支持的搜索字段列表
        
        Returns:
            List[Dict[str, str]]: 字段前缀和说明列表
        """
        return [
            {"prefix": "ti", "field": "Title", "description": "搜索论文标题"},
            {"prefix": "au", "field": "Author", "description": "搜索作者姓名"},
            {"prefix": "abs", "field": "Abstract", "description": "搜索摘要"},
            {"prefix": "cat", "field": "Subject Category", "description": "搜索学科分类"},
            {"prefix": "all", "field": "All Fields", "description": "搜索所有字段"}
        ]
    
    def get_subject_categories(self) -> List[Dict[str, str]]:
        """
        获取完整的arXiv学科分类列表
        
        Returns:
            List[Dict[str, str]]: 分类代码和名称列表
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
            {"code": "astro-ph", "name": "Astrophysics"}
        ]