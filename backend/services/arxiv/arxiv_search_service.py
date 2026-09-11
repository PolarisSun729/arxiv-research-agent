"""arXiv 在线搜索服务模块。

该模块负责与 arXiv 官方 API 交互，提供查询构造、结果解析、限流控制、
重试请求以及 PDF 下载等能力。它是面向实时在线检索场景的服务实现，与
本地 OAI 数据库检索服务形成互补。
"""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import requests
import feedparser
import os
import json
from pathlib import Path
import re
import tempfile
import urllib.parse
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import random

from services.arxiv.contracts import ArxivDownloadRequestError, ArxivRemoteSearchError
from services.arxiv.arxiv_query_builder import (
    normalize_id_list as _normalize_id_list,
    normalize_text_value as _normalize_text_value,
    validate_arxiv_search_request as _validate_arxiv_search_request,
)
from utils.storage_paths import resolve_backend_artifact_path
logger = logging.getLogger(__name__)

_ARXIV_ID = re.compile(r"(?:[0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?|[a-z-]+(?:\.[A-Z]{2})?/[0-9]{7}(?:v[0-9]+)?)")
_PDF_HOSTS = frozenset({"arxiv.org", "www.arxiv.org", "export.arxiv.org"})


def _validated_pdf_url(url: str, arxiv_id: str, *, allow_http: bool = False) -> str:
    """仅接受同一论文的官方 PDF 地址；在联网和读写缓存前封闭 SSRF 与路径穿越入口。"""
    if not isinstance(arxiv_id, str) or len(arxiv_id) > 100 or not _ARXIV_ID.fullmatch(arxiv_id):
        raise ArxivDownloadRequestError("Invalid arXiv ID")
    if not isinstance(url, str) or any(ord(char) <= 32 or ord(char) == 127 for char in url):
        raise ArxivDownloadRequestError("Invalid arXiv PDF URL")
    try:
        parsed = urllib.parse.urlsplit(url)
        valid = (
            parsed.scheme in ({"http", "https"} if allow_http else {"https"})
            and parsed.hostname in _PDF_HOSTS
            and parsed.username is None and parsed.password is None
            and parsed.port in {None, 443}
            and parsed.path in {f"/pdf/{arxiv_id}", f"/pdf/{arxiv_id}.pdf"}
            and not parsed.query and not parsed.fragment
        )
    except ValueError:
        valid = False
    if not valid:
        raise ArxivDownloadRequestError("PDF URL must reference the same paper on arxiv.org")
    # 兼容旧 Atom 数据中的 HTTP 链接，但真实传输始终使用 HTTPS；重定向不得降级。
    return urllib.parse.urlunsplit(("https", parsed.hostname, parsed.path, "", ""))


class RateLimitError(Exception):
    """
    自定义异常：API 请求被限流。
    """
    pass

class ArxivSearchService:
    """
    arXiv 在线论文搜索服务。

    该服务面向在线场景，负责：
    1. 组装 arXiv API 查询 URL；
    2. 控制请求速率与失败重试；
    3. 解析 Atom feed 结果；
    4. 对外返回统一格式的论文元数据。
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
    MAX_PDF_BYTES = 50 * 1024 * 1024
    MAX_PDF_REDIRECTS = 3
    _last_request_time = 0
    
    def __init__(self, proxy_url: Optional[str] = None):
        """
        初始化 arXiv 搜索服务。

        参数:
            proxy_url (Optional[str]): 可选代理地址；为空时会继续尝试从环境变量读取。

        返回:
            None
        """
        self.api_base_url = "https://export.arxiv.org/api/query"
        # arXiv 下载产物属于后端运行资产，不能因启动目录不同散落到仓库根目录。
        self.papers_dir = resolve_backend_artifact_path(
            "06-daily-arxiv-paper",
            option_name="ARXIV_PAPERS_DIR",
        )
        os.makedirs(self.papers_dir, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)
        self._configure_proxy(proxy_url)

    def _configure_proxy(self, proxy_url: Optional[str] = None) -> None:
        """配置 requests 会话代理。

        参数:
            proxy_url (Optional[str]): 显式传入的代理地址。

        返回:
            None
        """
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
        获取随机的 User-Agent。

        返回:
            str: 随机选中的请求头 User-Agent。
        """
        return random.choice(self.USER_AGENTS)
    
    def _wait_for_rate_limit(self):
        """
        等待速率限制间隔。

        返回:
            None

        说明:
            通过类级时间戳保证连续两次请求之间至少间隔 ``RATE_LIMIT_SECONDS`` 秒，
            以降低触发 arXiv 侧限流的概率。
        """
        import time
        current_time = time.time()
        time_since_last_request = current_time - ArxivSearchService._last_request_time
        
        if time_since_last_request < self.RATE_LIMIT_SECONDS:
            wait_time = self.RATE_LIMIT_SECONDS - time_since_last_request
            logger.debug(f"Rate limiting: waiting {wait_time:.2f} seconds before next request")
            time.sleep(wait_time)
        
        ArxivSearchService._last_request_time = time.time()
    
    def build_query_url(self, 
                       search_query: str = "",
                       id_list: Optional[List[str]] = None,
                       max_results: int = 10, 
                       start: int = 0,
                       sort_by: str = "relevance",
                       sort_order: str = "descending") -> str:
        """
        构建 arXiv API 查询 URL。

        参数:
            search_query (str): 搜索查询字符串，支持字段前缀语法。
            id_list (Optional[List[str]]): arXiv 论文 ID 列表，用于精确匹配。
            max_results (int): 返回结果的最大数量。
            start (int): 起始索引，用于分页。
            sort_by (str): 排序方式。
            sort_order (str): 排序顺序。

        返回:
            str: 构建好的查询 URL。
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
        解析单个 arXiv 论文条目，提取完整元数据。

        参数:
            entry (Dict[str, Any]): ``feedparser`` 解析后的原始条目。

        返回:
            Dict[str, Any]: 统一格式的论文信息字典。
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
    
    def search(
        self,
        search_query: Optional[str] = None,
        id_list: Optional[List[str]] = None,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> Dict[str, Any]:
        """执行在线 arXiv 搜索。

        参数:
            search_query (Optional[str]): 查询字符串。
            id_list (Optional[List[str]]): 论文 ID 列表。
            max_results (int): 最大返回条数。
            start (int): 分页偏移量。
            sort_by (str): 排序字段。
            sort_order (str): 排序方向。

        返回:
            Dict[str, Any]: 标准化后的在线搜索结果。
        """
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

        try:
            response = self._make_request_with_retry(url)
        except requests.exceptions.RequestException as exc:
            raise ArxivRemoteSearchError(
                "远程 arXiv API 请求失败。",
                query=normalized_search_query,
                reason=str(exc),
            ) from exc
        except RateLimitError as exc:
            raise ArxivRemoteSearchError(
                "远程 arXiv API 请求被限流。",
                query=normalized_search_query,
                reason=str(exc),
            ) from exc

        feed = feedparser.parse(response.content)

        papers = [self.parse_arxiv_entry(entry) for entry in feed.entries]
        result = {
            "query": normalized_search_query,
            "id_list": normalized_id_list,
            "source": "api",
            "query_capability": {
                "source": "api",
                "mode": "remote_arxiv_api",
                "full_arxiv_syntax_supported": True,
                "supported_subset": None,
            },
            "warnings": [],
            "total_results": int(feed.feed.get("opensearch_totalresults", 0)),
            "start_index": int(feed.feed.get("opensearch_startindex", 0)),
            "items_per_page": int(feed.feed.get("opensearch_itemsperpage", 0)),
            "papers": papers,
            "timestamp": datetime.now().isoformat(),
        }

        logger.debug("Found %s papers out of %s total results", len(papers), result["total_results"])
        return result
    
    def _make_request_with_retry(self, url: str, *, pdf_arxiv_id: Optional[str] = None) -> requests.Response:
        """
        带重试机制的 HTTP 请求方法。

        参数:
            url (str): 请求 URL。
            pdf_arxiv_id: PDF 下载时绑定的论文 ID；启用逐跳校验和流式读取。

        返回:
            requests.Response: HTTP 响应对象。

        异常:
            RateLimitError: 超过最大重试次数后仍然被限流时抛出。
            requests.exceptions.RequestException: 其他网络请求错误。
        """
        self._wait_for_rate_limit()
        logger.info(
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
            
            if pdf_arxiv_id is None:
                response = self.session.get(url, timeout=30, headers=headers)
            else:
                target = _validated_pdf_url(url, pdf_arxiv_id)
                for redirect_count in range(self.MAX_PDF_REDIRECTS + 1):
                    # requests 默认跟随重定向，会在调用方检查前触达任意目标，因此必须逐跳验证。
                    response = self.session.get(target, timeout=30, headers=headers, allow_redirects=False, stream=True)
                    if response.status_code not in {301, 302, 303, 307, 308}:
                        break
                    location = response.headers.get("Location", "")
                    response.close()
                    if not location or redirect_count == self.MAX_PDF_REDIRECTS:
                        raise ValueError("Too many or invalid arXiv PDF redirects")
                    target = _validated_pdf_url(urllib.parse.urljoin(target, location), pdf_arxiv_id)
            
            if response.status_code == 429:
                response.close()
                retry_after = int(response.headers.get("Retry-After", 15))
                logger.warning(f"arXiv API rate limit exceeded. Retrying after {retry_after} seconds...")
                
                import time
                time.sleep(retry_after)
                raise RateLimitError(f"HTTP 429: Rate limit exceeded for {url}")
            
            try:
                response.raise_for_status()
            except requests.exceptions.RequestException:
                response.close()
                raise
            return response
        
        try:
            return make_request()
        except RateLimitError:
            logger.error(f"Failed after retries: Rate limit still exceeded for {url}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            raise
    
    def _is_valid_pdf(self, filepath: str, min_size_kb: int = 0) -> bool:
        """
        严格验证 PDF 文件是否有效并可解析

        Args:
            filepath: PDF 文件路径
            min_size_kb: 最小文件大小（KB），默认 0（不检查大小，依赖 pypdf 验证）

        Returns:
            True 如果是有效的 PDF

        Raises:
            不抛出异常，验证失败返回 False
        """
        try:
            # 第一步：检查文件大小（如果设置了最小值）
            file_size = os.path.getsize(filepath)
            if min_size_kb > 0 and file_size < min_size_kb * 1024:
                logger.warning(f"PDF file too small: {file_size} bytes at {filepath}")
                return False

            # 第二步：检查 PDF 魔数（magic number）
            with open(filepath, "rb") as f:
                header = f.read(5)
                if not header.startswith(b'%PDF-'):
                    logger.warning(f"File does not start with PDF header: {header[:10]} at {filepath}")
                    return False

                # 检查文件末尾是否有 EOF 标记
                # 避免文件过小时 seek 出错
                if file_size > 1024:
                    f.seek(-1024, os.SEEK_END)
                    tail = f.read()
                else:
                    f.seek(0)
                    tail = f.read()

                if b'%%EOF' not in tail:
                    logger.warning(f"PDF file missing %%EOF marker at {filepath}")
                    return False

            # 第三步：尝试用 pypdf 解析 PDF 结构（最严格的验证）
            try:
                try:
                    from pypdf import PdfReader
                except ImportError:
                    # 如果 pypdf 不可用，降级为只检查魔数和 EOF
                    logger.debug(f"pypdf not available, skipping strict validation for {filepath}")
                    return True

                reader = PdfReader(filepath)
                # 尝试读取页数，确保 PDF 结构完整
                num_pages = len(reader.pages)
                if num_pages == 0:
                    logger.warning(f"PDF has 0 pages at {filepath}")
                    return False
                logger.debug(f"PDF validation passed: {num_pages} pages at {filepath}")
                return True
            except ImportError:
                # pypdf 不可用，已经在上面处理
                return True
            except Exception as parse_error:
                logger.warning(f"PDF structure validation failed at {filepath}: {parse_error}")
                return False

        except Exception as e:
            logger.error(f"Error validating PDF at {filepath}: {e}")
            return False

    def download_pdf(self, pdf_url: str, arxiv_id: str) -> str:
        """
        下载 arXiv 论文 PDF 并验证完整性

        Args:
            pdf_url: PDF 文件的 URL
            arxiv_id: arXiv 论文 ID，用于命名文件

        Returns:
            保存的文件路径

        Raises:
            ValueError: ID、地址、大小或 PDF 完整性验证失败
            requests.exceptions.RequestException: 下载失败
        """
        pdf_url = _validated_pdf_url(pdf_url, arxiv_id, allow_http=True)
        # 旧式 ID 中的斜杠属于论文标识，不是目录；所有下载统一保存为根目录下的单文件。
        filename = f"{arxiv_id.replace('/', '_')}.pdf"
        papers_root = Path(self.papers_dir).resolve()
        filepath = papers_root / filename
        if filepath.is_symlink() or filepath.resolve().parent != papers_root:
            raise ValueError("PDF cache path is outside the configured directory")

        # 如果文件已存在，先验证
        if os.path.exists(filepath):
            if self._is_valid_pdf(filepath):
                logger.debug(f"Valid PDF already exists: {filepath}")
                return str(filepath)
            else:
                logger.warning(f"Existing PDF is invalid, re-downloading: {filepath}")
                # 只有新文件验证成功后才替换缓存，网络失败不能先删除已有内容。

        temp_filepath = None
        response = None

        try:
            logger.info(f"Downloading PDF from: {pdf_url}")
            response = self._make_request_with_retry(pdf_url, pdf_arxiv_id=arxiv_id)

            # 检查 HTTP 状态码
            response.raise_for_status()

            # 检查 Content-Type
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/pdf' not in content_type and 'application/octet-stream' not in content_type:
                raise ValueError(
                    f"Response is not a PDF, got Content-Type: {content_type} for {arxiv_id}"
                )

            declared_size = response.headers.get("Content-Length")
            if declared_size is not None and not 0 <= int(declared_size) <= self.MAX_PDF_BYTES:
                raise ValueError("PDF exceeds the 50 MiB download limit")

            # 独占临时文件避免同篇论文的并发请求互相覆盖；流式计数也覆盖缺失/伪造长度与压缩响应。
            with tempfile.NamedTemporaryFile(dir=papers_root, prefix=f".{filename}.", suffix=".tmp", delete=False) as f:
                temp_filepath = f.name
                received_bytes = 0
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    received_bytes += len(chunk)
                    if received_bytes > self.MAX_PDF_BYTES:
                        raise ValueError("PDF exceeds the 50 MiB download limit")
                    f.write(chunk)

            # 验证下载的 PDF
            if not self._is_valid_pdf(temp_filepath):
                raise ValueError(f"Downloaded file is not a valid PDF: {arxiv_id}")

            # 原子性地移动到最终位置
            os.replace(temp_filepath, filepath)

            logger.info(f"Successfully downloaded and verified PDF: {filepath}")
            return str(filepath)

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error when downloading PDF for {arxiv_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error downloading PDF for {arxiv_id}: {e}")
            raise
        finally:
            if response is not None:
                response.close()
            if temp_filepath is not None:
                # 只清理本次独占创建的临时文件；成功替换后原路径已不存在。
                try:
                    os.remove(temp_filepath)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.warning("Failed to remove temporary arXiv PDF")
    
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
            
            logger.debug(f"Search results saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving search results: {str(e)}")
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
