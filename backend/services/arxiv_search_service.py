from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime, timedelta, timezone
import requests
import feedparser
import os
import json
import urllib.parse
from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import random

from services.arxiv_query_builder import (
    ARXIV_SEARCH_CONFIG,
    MAX_ALLOWED_RESULTS,
    VALID_CATEGORY_OPERATORS,
    VALID_FIELD_OPERATORS,
    VALID_SORT_BY,
    VALID_SORT_ORDER,
    ArxivSearchValidationError,
    build_arxiv_field_clause as _build_arxiv_field_clause,
    build_arxiv_query_from_structured_params,
    build_arxiv_raw_query,
    build_arxiv_submitted_date_query,
    combine_arxiv_clauses as _combine_arxiv_clauses,
    normalize_id_list as _normalize_id_list,
    normalize_text_value as _normalize_text_value,
    quote_arxiv_text as _quote_arxiv_text,
    validate_arxiv_search_request as _validate_arxiv_search_request,
    validate_arxiv_search_request,
)
logger = logging.getLogger(__name__)

class RateLimitError(Exception):
    """
    自定义异常：API 请求被限流
    """
    pass

class SearchField(str, Enum):
    """
    arXiv API 搜索字段前缀枚举
    参考: https://info.arxiv.org/help/api/user-manual.html
    """
    TITLE = "ti"
    AUTHOR = "au"
    ABSTRACT = "abs"
    COMMENT = "co"
    JOURNAL_REFERENCE = "jr"
    SUBJECT_CATEGORY = "cat"
    REPORT_NUMBER = "rn"
    ID = "id"
    ALL = "all"

class ArxivSearchService:
    """
    arXiv论文搜索和获取服务类
    提供论文搜索、元数据获取和PDF下载功能
    支持完整的arXiv API查询语法
    """
    
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]
    
    DEFAULT_HEADERS = {
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Cache-Control": "max-age=0",
    }
    
    RATE_LIMIT_SECONDS = 15
    _last_request_time = 0
    
    def __init__(self, proxy_url: Optional[str] = None):
        """
        初始化arXiv搜索服务
        设置API端点和保存目录
        """
        self.api_base_url = "https://export.arxiv.org/api/query"
        self.papers_dir = "06-daily-arxiv-paper"
        os.makedirs(self.papers_dir, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)
        self._configure_proxy(proxy_url)

    def _configure_proxy(self, proxy_url: Optional[str] = None) -> None:
        resolved_proxy = (
            proxy_url
            or os.getenv("ARXIV_PROXY_URL")
            or os.getenv("HTTPS_PROXY")
            or os.getenv("HTTP_PROXY")
            or os.getenv("https_proxy")
            or os.getenv("http_proxy")
            or ""
        ).strip()
        if not resolved_proxy:
            return

        self.session.proxies.update({
            "http": resolved_proxy,
            "https": resolved_proxy,
        })
        self.proxy_url = resolved_proxy
        logger.info("Configured arXiv proxy: %s", resolved_proxy)
    
    def _get_random_user_agent(self) -> str:
        """
        获取随机的User-Agent
        """
        return random.choice(self.USER_AGENTS)
    
    def _wait_for_rate_limit(self):
        """
        等待速率限制间隔
        确保两次请求之间至少间隔 RATE_LIMIT_SECONDS 秒
        """
        import time
        current_time = time.time()
        time_since_last_request = current_time - ArxivSearchService._last_request_time
        
        if time_since_last_request < self.RATE_LIMIT_SECONDS:
            wait_time = self.RATE_LIMIT_SECONDS - time_since_last_request
            logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds before next request")
            time.sleep(wait_time)
        
        ArxivSearchService._last_request_time = time.time()
    
    def build_field_query(self, field: Union[SearchField, str], query: str) -> str:
        """
        构建字段限定查询
        
        Args:
            field (Union[SearchField, str]): 搜索字段
            query (str): 查询词
            
        Returns:
            str: 格式化为 "field:query" 的查询字符串
        """
        field_prefix = field.value if isinstance(field, SearchField) else field
        return f"{field_prefix}:{query}"
    
    def combine_queries(self, queries: List[str], operator: str = "AND") -> str:
        """
        组合多个查询条件
        
        Args:
            queries (List[str]): 查询条件列表
            operator (str): 逻辑操作符，"AND" 或 "OR"
            
        Returns:
            str: 组合后的查询字符串
        """
        if not queries:
            return ""
        if len(queries) == 1:
            return queries[0]
        return f" {operator} ".join(queries)
    
    def build_query_url(self, 
                       search_query: str = "",
                       id_list: Optional[List[str]] = None,
                       max_results: int = 10, 
                       start: int = 0,
                       sort_by: str = "relevance",
                       sort_order: str = "descending") -> str:
        """
        构建arXiv API查询URL
        
        Args:
            search_query (str): 搜索查询字符串，支持字段前缀语法
            id_list (Optional[List[str]]): arXiv论文ID列表，用于精确匹配
            max_results (int): 返回结果的最大数量，默认为10
            start (int): 起始索引，用于分页，默认为0
            sort_by (str): 排序方式："relevance", "lastUpdatedDate", "submittedDate"
            sort_order (str): 排序顺序："ascending", "descending"
            
        Returns:
            str: 构建好的查询URL
        """
        params = {
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order
        }
        
        if search_query:
            params["search_query"] = search_query
        
        if id_list and len(id_list) > 0:
            params["id_list"] = ",".join(id_list)
        
        query_string = urllib.parse.urlencode(params)
        return f"{self.api_base_url}?{query_string}"
    
    def parse_arxiv_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析单个arXiv论文条目，提取完整的元数据
        
        Args:
            entry (Dict[str, Any]): feedparser解析的原始条目
            
        Returns:
            Dict[str, Any]: 解析后的论文信息
        """
        paper = {
            "id": entry.get("id", ""),
            "title": entry.get("title", "").replace("\n", " ").strip(),
            "summary": entry.get("summary", "").replace("\n", " ").strip(),
            "published": entry.get("published", ""),
            "updated": entry.get("updated", ""),
            "authors": [author.get("name", "") for author in entry.get("authors", [])],
            "categories": [tag.get("term", "") for tag in entry.get("tags", [])],
            "pdf_url": "",
            "abs_url": "",
            "journal_reference": "",
            "comment": "",
            "doi": ""
        }
        
        for link in entry.get("links", []):
            if link.get("title") == "pdf":
                paper["pdf_url"] = link.get("href", "")
            elif link.get("type") == "text/html":
                paper["abs_url"] = link.get("href", "")
        
        arxiv_ext = entry.get("arxiv", {})
        if isinstance(arxiv_ext, dict):
            paper["journal_reference"] = arxiv_ext.get("journal_ref", "")
            paper["comment"] = arxiv_ext.get("comment", "")
            paper["doi"] = arxiv_ext.get("doi", "")
        
        paper["arxiv_id"] = paper["id"].split("/")[-1] if paper["id"] else ""
        
        return paper
    
    def build_submitted_date_query(self, days_ago: Optional[int] = 30) -> str:
        """
        构建提交日期范围查询字符串
        
        Args:
            days_ago (Optional[int]): 多少天前的日期作为起始日期，默认为30天（一个月）
            
        Returns:
            str: 格式化的submittedDate查询字符串，格式为 [YYYYMMDDTTTT+TO+YYYYMMDDTTTT]
        """
        return build_arxiv_submitted_date_query(days_ago)

    def search(
        self,
        search_query: Optional[str] = None,
        id_list: Optional[List[str]] = None,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> Dict[str, Any]:
        _validate_arxiv_search_request(
            search_query=search_query,
            id_list=id_list,
            max_results=max_results,
            start=start,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        normalized_search_query = _normalize_text_value(search_query)
        normalized_id_list = _normalize_id_list(id_list)
        logger.info(
            "Searching arXiv with query: '%s', id_list: %s, max_results: %s",
            normalized_search_query,
            normalized_id_list,
            max_results,
        )

        url = self.build_query_url(
            normalized_search_query,
            normalized_id_list,
            max_results,
            start,
            sort_by,
            sort_order,
        )

        response = self._make_request_with_retry(url)
        feed = feedparser.parse(response.content)

        papers = [self.parse_arxiv_entry(entry) for entry in feed.entries]
        result = {
            "query": normalized_search_query,
            "id_list": normalized_id_list,
            "total_results": int(feed.feed.get("opensearch_totalresults", 0)),
            "start_index": int(feed.feed.get("opensearch_startindex", 0)),
            "items_per_page": int(feed.feed.get("opensearch_itemsperpage", 0)),
            "papers": papers,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info("Found %s papers out of %s total results", len(papers), result["total_results"])
        return result
    
    def _make_request_with_retry(self, url: str) -> requests.Response:
        """
        带重试机制的HTTP请求方法
        
        Args:
            url (str): 请求的URL
            
        Returns:
            requests.Response: HTTP响应对象
            
        Raises:
            RateLimitError: 超过最大重试次数后仍然被限流
            requests.exceptions.RequestException: 其他请求错误
        """
        self._wait_for_rate_limit()
        logger.debug(
            "arXiv request prepared: url=%s, proxy_http=%s, proxy_https=%s",
            url,
            self.session.proxies.get("http", ""),
            self.session.proxies.get("https", ""),
        )
        
        @retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=2, min=5, max=30),
            retry=retry_if_exception_type((RateLimitError,)),
            before_sleep=before_sleep_log(logger, logging.WARNING)
        )
        def make_request():
            headers = {"User-Agent": self._get_random_user_agent()}
            
            response = self.session.get(url, timeout=30, headers=headers)
            
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 15))
                logger.warning(f"arXiv API rate limit exceeded. Retrying after {retry_after} seconds...")
                
                import time
                time.sleep(retry_after)
                raise RateLimitError(f"HTTP 429: Rate limit exceeded for {url}")
            
            response.raise_for_status()
            return response
        
        try:
            return make_request()
        except RateLimitError:
            logger.error(f"Failed after retries: Rate limit still exceeded for {url}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            raise
    
    def search_papers(self, 
                     search_query: str = "",
                     id_list: Optional[List[str]] = None,
                     max_results: int = 10, 
                     start: int = 0,
                     sort_by: str = "relevance",
                     sort_order: str = "descending",
                     submitted_days_ago: Optional[int] = 30) -> Dict[str, Any]:
        """
        搜索arXiv论文
        
        Args:
            search_query (str): 搜索查询字符串，支持字段前缀语法
                              例如: "ti:deep learning", "au:John+Doe", "cat:cs.AI"
            id_list (Optional[List[str]]): arXiv论文ID列表，用于精确匹配
            max_results (int): 返回结果的最大数量，默认为10
            start (int): 起始索引，用于分页，默认为0
            sort_by (str): 排序方式："relevance", "lastUpdatedDate", "submittedDate"
            sort_order (str): 排序顺序："ascending", "descending"
            submitted_days_ago (Optional[int]): 搜索提交日期在多少天内的文章，默认为30天（一个月）
                                              设置为None或0可搜索所有日期
            
        Returns:
            Dict[str, Any]: 搜索结果，包含论文列表和元信息
            
        Raises:
            Exception: 请求或解析失败时
        """
        try:
            raw_query = build_arxiv_raw_query(
                search_query=search_query,
                id_list=id_list,
                submitted_days_ago=submitted_days_ago,
                append_date_when_query_missing=True,
            )
            return self.search(
                search_query=raw_query["final_search_query"],
                id_list=raw_query["id_list"],
                max_results=max_results,
                start=start,
                sort_by=sort_by,
                sort_order=sort_order,
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error when searching arXiv: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error searching arXiv: {str(e)}")
            raise
    
    def search_by_author(self, author: str, **kwargs) -> Dict[str, Any]:
        """
        按作者搜索论文
        
        Args:
            author (str): 作者姓名
            **kwargs: 其他搜索参数（max_results, start, sort_by, sort_order）
            
        Returns:
            Dict[str, Any]: 搜索结果
        """
        query = self.build_field_query(SearchField.AUTHOR, author)
        return self.search_papers(query, **kwargs)
    
    def search_by_title(self, title: str, **kwargs) -> Dict[str, Any]:
        """
        按标题搜索论文
        
        Args:
            title (str): 标题关键词
            **kwargs: 其他搜索参数（max_results, start, sort_by, sort_order）
            
        Returns:
            Dict[str, Any]: 搜索结果
        """
        query = self.build_field_query(SearchField.TITLE, title)
        return self.search_papers(query, **kwargs)
    
    def search_by_category(self, category: str, **kwargs) -> Dict[str, Any]:
        """
        按学科分类搜索论文
        
        Args:
            category (str): 学科分类代码，如 "cs.AI", "physics.quant-ph"
            **kwargs: 其他搜索参数（max_results, start, sort_by, sort_order）
            
        Returns:
            Dict[str, Any]: 搜索结果
        """
        query = self.build_field_query(SearchField.SUBJECT_CATEGORY, category)
        return self.search_papers(query, **kwargs)
    
    def search_by_abstract(self, abstract: str, **kwargs) -> Dict[str, Any]:
        """
        按摘要搜索论文
        
        Args:
            abstract (str): 摘要关键词
            **kwargs: 其他搜索参数（max_results, start, sort_by, sort_order）
            
        Returns:
            Dict[str, Any]: 搜索结果
        """
        query = self.build_field_query(SearchField.ABSTRACT, abstract)
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
                       submitted_days_ago: Optional[int] = 30) -> Dict[str, Any]:
        """
        高级搜索，支持多条件组合
        
        Args:
            title (Optional[str]): 标题关键词
            author (Optional[str]): 作者姓名
            abstract (Optional[str]): 摘要关键词
            category (Optional[str]): 学科分类代码
            comment (Optional[str]): 评论关键词
            journal_ref (Optional[str]): 期刊引用关键词
            report_number (Optional[str]): 报告编号关键词
            operator (str): 逻辑操作符，"AND" 或 "OR"
            id_list (Optional[List[str]]): arXiv论文ID列表
            max_results (int): 返回结果的最大数量
            start (int): 起始索引
            sort_by (str): 排序方式
            sort_order (str): 排序顺序
            submitted_days_ago (Optional[int]): 搜索提交日期在多少天内的文章，默认为30天（一个月）
                                              设置为None或0可搜索所有日期
            
        Returns:
            Dict[str, Any]: 搜索结果
        """
        structured = build_arxiv_query_from_structured_params(
            query=None,
            title_query=title,
            author_query=author,
            abstract_query=abstract,
            categories=[category] if category else None,
            comment_query=comment,
            journal_ref_query=journal_ref,
            report_number_query=report_number,
            id_list=id_list,
            field_operator=operator,
            category_operator="OR",
            submitted_days_ago=submitted_days_ago,
        )
        return self.search(
            search_query=structured["final_search_query"],
            id_list=structured["id_list"],
            max_results=max_results,
            start=start,
            sort_by=sort_by,
            sort_order=sort_order,
        )
    
    def download_pdf(self, pdf_url: str, arxiv_id: str) -> str:
        """
        下载arXiv论文PDF
        
        Args:
            pdf_url (str): PDF文件的URL
            arxiv_id (str): arXiv论文ID，用于命名文件
            
        Returns:
            str: 保存的文件路径
            
        Raises:
            Exception: 下载失败时
        """
        try:
            filename = f"{arxiv_id}.pdf"
            filepath = os.path.join(self.papers_dir, filename)
            
            if os.path.exists(filepath):
                logger.info(f"PDF already exists: {filepath}")
                return filepath
            
            logger.info(f"Downloading PDF from: {pdf_url}")
            
            response = self._make_request_with_retry(pdf_url)
            
            with open(filepath, "wb") as f:
                f.write(response.content)
            
            logger.info(f"Successfully downloaded PDF to: {filepath}")
            return filepath
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error when downloading PDF: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error downloading PDF: {str(e)}")
            raise
    
    def save_search_results(self, search_result: Dict[str, Any]) -> str:
        """
        保存搜索结果到JSON文件
        
        Args:
            search_result (Dict[str, Any]): 搜索结果
            
        Returns:
            str: 保存文件的路径
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"arxiv_search_{timestamp}.json"
            filepath = os.path.join(self.papers_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(search_result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Search results saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving search results: {str(e)}")
            raise
    
    async def search_and_save(self, 
                             search_query: str = "",
                             id_list: Optional[List[str]] = None,
                             max_results: int = 10,
                             download_pdfs: bool = False,
                             **kwargs) -> Dict[str, Any]:
        """
        搜索论文并保存结果，可选择下载PDF
        
        Args:
            search_query (str): 搜索查询字符串
            id_list (Optional[List[str]]): arXiv论文ID列表
            max_results (int): 返回结果的最大数量
            download_pdfs (bool): 是否下载PDF文件
            **kwargs: 其他搜索参数
            
        Returns:
            Dict[str, Any]: 搜索结果，包含保存的文件路径
        """
        try:
            search_result = self.search_papers(search_query, id_list, max_results, **kwargs)
            
            search_filepath = self.save_search_results(search_result)
            
            downloaded_files = []
            if download_pdfs:
                for paper in search_result["papers"]:
                    if paper.get("pdf_url"):
                        try:
                            filepath = self.download_pdf(paper["pdf_url"], paper["arxiv_id"])
                            downloaded_files.append({
                                "arxiv_id": paper["arxiv_id"],
                                "title": paper["title"],
                                "filepath": filepath
                            })
                        except Exception as e:
                            logger.error(f"Failed to download {paper['arxiv_id']}: {str(e)}")
            
            return {
                "search_result": search_result,
                "search_filepath": search_filepath,
                "downloaded_files": downloaded_files
            }
            
        except Exception as e:
            logger.error(f"Error in search_and_save: {str(e)}")
            raise
    
    @staticmethod
    def get_available_fields() -> List[Dict[str, str]]:
        """
        获取支持的搜索字段列表
        
        Returns:
            List[Dict[str, str]]: 字段前缀和说明列表
        """
        return [
            {"prefix": "ti", "field": "Title", "description": "搜索论文标题"},
            {"prefix": "au", "field": "Author", "description": "搜索作者姓名"},
            {"prefix": "abs", "field": "Abstract", "description": "搜索摘要"},
            {"prefix": "co", "field": "Comment", "description": "搜索评论"},
            {"prefix": "jr", "field": "Journal Reference", "description": "搜索期刊引用"},
            {"prefix": "cat", "field": "Subject Category", "description": "搜索学科分类"},
            {"prefix": "rn", "field": "Report Number", "description": "搜索报告编号"},
            {"prefix": "id", "field": "ID", "description": "搜索论文ID（建议使用id_list参数）"},
            {"prefix": "all", "field": "All Fields", "description": "搜索所有字段"}
        ]
    
    @staticmethod
    def get_subject_categories() -> List[Dict[str, str]]:
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
            {"code": "stat.AP", "name": "Statistics Applications"},
            {"code": "stat.CO", "name": "Computational Statistics"},
            {"code": "stat.ME", "name": "Methodology"},
            {"code": "stat.ML", "name": "Machine Learning"},
            {"code": "stat.OT", "name": "Other Statistics"},
            {"code": "stat.TH", "name": "Statistics Theory"},
            {"code": "physics.quant-ph", "name": "Quantum Physics"},
            {"code": "math.AP", "name": "Analysis of PDEs"},
            {"code": "math.CV", "name": "Complex Variables"},
            {"code": "math.GR", "name": "Group Theory"},
            {"code": "math.LO", "name": "Logic"},
            {"code": "math.PR", "name": "Probability"},
            {"code": "math.ST", "name": "Statistics Theory"},
            {"code": "q-bio", "name": "Quantitative Biology"},
            {"code": "q-fin", "name": "Quantitative Finance"},
            {"code": "econ", "name": "Economics"},
            {"code": "hep-th", "name": "High Energy Physics - Theory"},
            {"code": "hep-ph", "name": "High Energy Physics - Phenomenology"},
            {"code": "hep-ex", "name": "High Energy Physics - Experiment"},
            {"code": "gr-qc", "name": "General Relativity and Quantum Cosmology"},
            {"code": "astro-ph", "name": "Astrophysics"}
        ]
