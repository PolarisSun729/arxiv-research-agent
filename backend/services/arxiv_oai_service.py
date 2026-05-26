import json
import logging
import os
import re
import time
import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from utils.config import OAI_SQLITE_CONFIG

logger = logging.getLogger(__name__)

OAI_ENDPOINT = "https://oaipmh.arxiv.org/oai"
TARGET_CATEGORIES = {"cs.CL", "cs.LG", "cs.IR", "cs.AI"}


@dataclass
class ArxivOaiSyncStats:
    requests_made: int = 0
    pages_processed: int = 0
    records_seen: int = 0
    records_with_metadata: int = 0
    records_matched: int = 0
    records_written: int = 0
    records_skipped: int = 0
    skipped_no_metadata: int = 0
    skipped_no_arxiv_id: int = 0
    skipped_no_categories: int = 0
    skipped_category_filter: int = 0
    skipped_parse_errors: int = 0
    errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ArxivOaiDatabaseService:
    def __init__(self, db_path: Optional[str] = None, check_same_thread: Optional[bool] = None):
        self.db_path = db_path or OAI_SQLITE_CONFIG["database_path"]
        self.check_same_thread = (
            OAI_SQLITE_CONFIG["check_same_thread"] if check_same_thread is None else bool(check_same_thread)
        )
        self._ensure_database_directory()
        self._initialize_database()

    def _ensure_database_directory(self) -> None:
        directory = os.path.dirname(self.db_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info("Created OAI database directory: %s", directory)

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=self.check_same_thread)

    def _parse_list_field(self, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return ""
            if text.startswith("[") or text.startswith("("):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except json.JSONDecodeError:
                    return text
            return text
        return value

    def get_paper(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        normalized_arxiv_id = str(arxiv_id or "").strip()
        if not normalized_arxiv_id:
            return None

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT
                        arxiv_id,
                        title,
                        abstract,
                        authors,
                        categories,
                        primary_category,
                        created,
                        updated,
                        abs_url,
                        pdf_url,
                        oai_datestamp,
                        fetched_at,
                        created_at,
                        updated_at
                    FROM arxiv_oai_papers
                    WHERE arxiv_id = ?
                    ''',
                    (normalized_arxiv_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                return {
                    "arxiv_id": row[0],
                    "title": row[1],
                    "abstract": row[2],
                    "authors": self._parse_list_field(row[3]),
                    "categories": self._parse_list_field(row[4]),
                    "primary_category": row[5],
                    "created": row[6],
                    "updated": row[7],
                    "abs_url": row[8],
                    "pdf_url": row[9],
                    "oai_datestamp": row[10],
                    "fetched_at": row[11],
                    "created_at": row[12],
                    "updated_at": row[13],
                }
        except Exception as exc:
            logger.error("Error getting OAI paper %s: %s", normalized_arxiv_id, exc)
            return None

    def get_recent_papers(
        self,
        categories: Optional[Sequence[str]] = None,
        max_age_months: int = 6,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        try:
            query_parts = [
                '''
                SELECT
                    arxiv_id,
                    title,
                    abstract,
                    authors,
                    categories,
                    primary_category,
                    created,
                    updated,
                    abs_url,
                    pdf_url,
                    oai_datestamp,
                    fetched_at,
                    created_at,
                    updated_at
                FROM arxiv_oai_papers
                '''
            ]
            params: List[Any] = []
            where_clauses: List[str] = []

            if max_age_months > 0:
                cutoff_date = (datetime.utcnow() - timedelta(days=max_age_months * 30)).date().isoformat()
                where_clauses.append(
                    "COALESCE(created, updated, oai_datestamp, fetched_at) >= ?"
                )
                params.append(cutoff_date)

            normalized_categories = [
                str(category).strip()
                for category in (categories or [])
                if str(category).strip()
            ]
            if normalized_categories:
                category_filters = []
                for category in normalized_categories:
                    category_filters.append("categories LIKE ? OR primary_category = ?")
                    params.extend([f'%"{category}"%', category])
                where_clauses.append(f"({' OR '.join(f'({item})' for item in category_filters)})")

            if where_clauses:
                query_parts.append("WHERE " + " AND ".join(where_clauses))

            query_parts.append(
                '''
                ORDER BY
                    COALESCE(created, updated, oai_datestamp, fetched_at) DESC,
                    arxiv_id DESC
                LIMIT ?
                '''
            )
            params.append(max(1, int(max_results)))

            query = "\n".join(part.strip() for part in query_parts if part.strip())

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [
                    {
                        "arxiv_id": row[0],
                        "title": row[1],
                        "abstract": row[2],
                        "authors": self._parse_list_field(row[3]),
                        "categories": self._parse_list_field(row[4]),
                        "primary_category": row[5],
                        "created": row[6],
                        "updated": row[7],
                        "abs_url": row[8],
                        "pdf_url": row[9],
                        "oai_datestamp": row[10],
                        "fetched_at": row[11],
                        "created_at": row[12],
                        "updated_at": row[13],
                    }
                    for row in rows
                ]
        except Exception as exc:
            logger.error("Error getting recent OAI papers: %s", exc)
            return []

    def _initialize_database(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''
                CREATE TABLE IF NOT EXISTS arxiv_oai_papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    arxiv_id TEXT NOT NULL UNIQUE,
                    title TEXT,
                    abstract TEXT,
                    authors TEXT,
                    categories TEXT,
                    primary_category TEXT,
                    created TEXT,
                    updated TEXT,
                    abs_url TEXT,
                    pdf_url TEXT,
                    oai_datestamp TEXT,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                '''
            )
            conn.commit()
            logger.info("OAI database tables initialized successfully: %s", self.db_path)

    def upsert_arxiv_oai_paper(self, paper: Dict[str, Any]) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    INSERT INTO arxiv_oai_papers (
                        arxiv_id,
                        title,
                        abstract,
                        authors,
                        categories,
                        primary_category,
                        created,
                        updated,
                        abs_url,
                        pdf_url,
                        oai_datestamp,
                        fetched_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(arxiv_id) DO UPDATE SET
                        title = excluded.title,
                        abstract = excluded.abstract,
                        authors = excluded.authors,
                        categories = excluded.categories,
                        primary_category = excluded.primary_category,
                        created = excluded.created,
                        updated = excluded.updated,
                        abs_url = excluded.abs_url,
                        pdf_url = excluded.pdf_url,
                        oai_datestamp = excluded.oai_datestamp,
                        fetched_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    ''',
                    (
                        paper.get("arxiv_id"),
                        paper.get("title"),
                        paper.get("abstract"),
                        paper.get("authors"),
                        paper.get("categories"),
                        paper.get("primary_category"),
                        paper.get("created"),
                        paper.get("updated"),
                        paper.get("abs_url"),
                        paper.get("pdf_url"),
                        paper.get("oai_datestamp"),
                    ),
                )
                conn.commit()
                logger.info("OAI paper upserted: %s", paper.get("arxiv_id"))
                return True
        except Exception as exc:
            logger.error("Error upserting OAI paper: %s", exc)
            return False


class ArxivOaiSyncService:
    def __init__(
        self,
        endpoint: str = OAI_ENDPOINT,
        request_interval_seconds: float = 5.0,
        request_timeout_seconds: float = 60.0,
        max_retries: int = 5,
        database_service: Optional[ArxivOaiDatabaseService] = None,
        user_agent: str = "rag-project01-framework-oai-sync/1.0",
    ):
        self.endpoint = endpoint
        self.request_interval_seconds = max(3.5, float(request_interval_seconds))
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.max_retries = max(1, int(max_retries))
        self.database_service = database_service
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def sync(self, from_date: str, until_date: str, dry_run: bool = False) -> ArxivOaiSyncStats:
        stats = ArxivOaiSyncStats()
        request_params: Dict[str, str] = {
            "verb": "ListRecords",
            "metadataPrefix": "arXiv",
            "from": from_date,
            "until": until_date,
        }
        resumption_token: Optional[str] = None

        logger.info(
            "Starting arXiv OAI-PMH sync: from=%s until=%s dry_run=%s endpoint=%s",
            from_date,
            until_date,
            dry_run,
            self.endpoint,
        )

        while True:
            if resumption_token:
                request_params = {"verb": "ListRecords", "resumptionToken": resumption_token}

            xml_text = self._fetch_page(request_params, stats)
            if xml_text is None:
                break

            try:
                root = ET.fromstring(xml_text)
            except ET.ParseError as exc:
                stats.errors += 1
                logger.error("Failed to parse OAI-PMH XML response: %s", exc)
                break

            page_error = self._extract_oai_error(root)
            if page_error:
                stats.errors += 1
                logger.error("OAI-PMH error response: %s", page_error)
                break

            records = root.findall(".//{*}record")
            stats.pages_processed += 1
            logger.info("Processing page %s with %s record nodes", stats.pages_processed, len(records))

            for record in records:
                self._process_record(record, stats, dry_run=dry_run)

            resumption_token = self._extract_resumption_token(root)
            if not resumption_token:
                logger.info("OAI-PMH resumptionToken exhausted; sync complete.")
                break

            logger.info("Received resumptionToken; continuing with next page.")
            time.sleep(self.request_interval_seconds)

        logger.info(
            (
                "OAI-PMH sync summary: requests=%s pages=%s seen=%s matched=%s "
                "written=%s skipped=%s filtered=%s no_metadata=%s no_categories=%s errors=%s dry_run=%s"
            ),
            stats.requests_made,
            stats.pages_processed,
            stats.records_seen,
            stats.records_matched,
            stats.records_written,
            stats.records_skipped,
            stats.skipped_category_filter,
            stats.skipped_no_metadata + stats.skipped_no_arxiv_id + stats.skipped_parse_errors,
            stats.skipped_no_categories,
            stats.errors,
            dry_run,
        )
        return stats

    def _fetch_page(self, params: Dict[str, str], stats: ArxivOaiSyncStats) -> Optional[str]:
        retry_delay = self.request_interval_seconds

        for attempt in range(1, self.max_retries + 1):
            stats.requests_made += 1
            try:
                response = self.session.get(self.endpoint, params=params, timeout=self.request_timeout_seconds)
            except requests.Timeout as exc:
                logger.warning(
                    "OAI-PMH request timeout on attempt %s/%s for params=%s: %s",
                    attempt,
                    self.max_retries,
                    params,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 60.0)
                    continue
                stats.errors += 1
                return None
            except requests.RequestException as exc:
                logger.error("OAI-PMH request error for params=%s: %s", params, exc)
                if attempt < self.max_retries:
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 60.0)
                    continue
                stats.errors += 1
                return None

            if response.status_code in {429, 503}:
                retry_after = self._parse_retry_after(response.headers.get("Retry-After"))
                sleep_seconds = max(retry_delay, retry_after or 0.0, self.request_interval_seconds)
                logger.warning(
                    "OAI-PMH returned %s on attempt %s/%s; sleeping %.1fs before retry",
                    response.status_code,
                    attempt,
                    self.max_retries,
                    sleep_seconds,
                )
                if attempt < self.max_retries:
                    time.sleep(sleep_seconds)
                    retry_delay = min(retry_delay * 2, 60.0)
                    continue
                stats.errors += 1
                return None

            if response.status_code >= 400:
                stats.errors += 1
                logger.error(
                    "OAI-PMH request failed with HTTP %s for params=%s: %s",
                    response.status_code,
                    params,
                    response.text[:1000],
                )
                return None

            return response.text

        return None

    def _process_record(self, record: ET.Element, stats: ArxivOaiSyncStats, dry_run: bool) -> None:
        stats.records_seen += 1
        metadata = self._find_first_child(record, "metadata")
        if metadata is None:
            stats.skipped_no_metadata += 1
            stats.records_skipped += 1
            logger.debug("Skipping record without metadata")
            return

        paper_elem = self._find_first_element_child(metadata)
        if paper_elem is None:
            stats.skipped_no_metadata += 1
            stats.records_skipped += 1
            logger.debug("Skipping record with empty metadata")
            return

        stats.records_with_metadata += 1
        paper = self._parse_paper_metadata(record, paper_elem)
        if paper is None:
            stats.skipped_parse_errors += 1
            stats.records_skipped += 1
            return

        arxiv_id = paper.get("arxiv_id", "")
        if not arxiv_id:
            stats.skipped_no_arxiv_id += 1
            stats.records_skipped += 1
            logger.warning("Skipping record without arxiv_id")
            return

        categories = paper.get("categories_list", [])
        if not categories:
            stats.skipped_no_categories += 1
            stats.records_skipped += 1
            logger.info("Skipping %s because categories are empty", arxiv_id)
            return

        if not self._is_allowed_categories(categories):
            stats.skipped_category_filter += 1
            stats.records_skipped += 1
            logger.debug("Skipping %s because categories=%s do not match target set", arxiv_id, categories)
            return

        stats.records_matched += 1
        if dry_run:
            stats.records_written += 1
            logger.info(
                "[dry-run] Would upsert %s | title=%s | categories=%s",
                arxiv_id,
                paper.get("title", ""),
                categories,
            )
            return

        if self.database_service is None:
            self.database_service = ArxivOaiDatabaseService()

        if self.database_service.upsert_arxiv_oai_paper(paper):
            stats.records_written += 1
        else:
            stats.errors += 1
            logger.error("Failed to upsert %s", arxiv_id)

    def _parse_paper_metadata(self, record: ET.Element, paper_elem: ET.Element) -> Optional[Dict[str, Any]]:
        try:
            raw_arxiv_id = self._extract_text(paper_elem, "id")
            arxiv_id = self._normalize_arxiv_id(raw_arxiv_id)
            title = self._normalize_whitespace(self._extract_text(paper_elem, "title"))
            abstract = self._normalize_whitespace(self._extract_text(paper_elem, "abstract"))
            created = self._normalize_whitespace(self._extract_text(paper_elem, "created"))
            updated = self._normalize_whitespace(self._extract_text(paper_elem, "updated") or self._extract_text(paper_elem, "updateDate"))
            categories_text = self._normalize_whitespace(self._extract_text(paper_elem, "categories"))
            categories = self._split_categories(categories_text)
            primary_category = self._extract_primary_category(paper_elem)
            authors = self._extract_authors(paper_elem)
            oai_datestamp = self._normalize_whitespace(self._extract_text(record, "datestamp"))

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
        authors_parent = self._find_first_child(paper_elem, "authors")
        search_root = authors_parent if authors_parent is not None else paper_elem
        authors: List[str] = []
        for node in search_root.iter():
            if self._local_name(node.tag) != "author":
                continue
            name_parts: List[str] = []
            forenames = self._normalize_whitespace(self._extract_text(node, "forenames"))
            keyname = self._normalize_whitespace(self._extract_text(node, "keyname"))
            full_name = self._normalize_whitespace(self._extract_text(node, "name"))
            if full_name:
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
        primary_category = self._find_first_descendant(paper_elem, "primary_category")
        if primary_category is None:
            primary_category = self._find_first_descendant(paper_elem, "primaryCategory")
        if primary_category is None:
            return ""
        term = primary_category.attrib.get("term", "").strip()
        if term:
            return term
        return self._normalize_whitespace(primary_category.text or "")

    def _extract_resumption_token(self, root: ET.Element) -> str:
        token_elem = self._find_first_descendant(root, "resumptionToken")
        if token_elem is None or token_elem.text is None:
            return ""
        return token_elem.text.strip()

    def _extract_oai_error(self, root: ET.Element) -> str:
        error_elem = self._find_first_descendant(root, "error")
        if error_elem is None:
            return ""
        code = error_elem.attrib.get("code", "").strip()
        text = self._normalize_whitespace(error_elem.text or "")
        if code and text:
            return f"{code}: {text}"
        return code or text

    def _is_allowed_categories(self, categories: Sequence[str]) -> bool:
        if not categories:
            return False
        normalized = [category.strip() for category in categories if str(category).strip()]
        if not normalized:
            return False
        return all(category in TARGET_CATEGORIES for category in normalized)

    def _split_categories(self, categories_text: str) -> List[str]:
        if not categories_text:
            return []
        raw_items = re.split(r"[\s,]+", categories_text.strip())
        return [item for item in (part.strip() for part in raw_items) if item]

    def _normalize_arxiv_id(self, raw_value: str) -> str:
        value = self._normalize_whitespace(raw_value)
        if not value:
            return ""
        value = re.sub(r"^oai:arXiv\.org:", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^arxiv:", "", value, flags=re.IGNORECASE)
        if "arxiv.org/abs/" in value.lower():
            value = value.rsplit("/", 1)[-1]
        value = value.split("?", 1)[0].strip()
        value = re.sub(r"v\d+$", "", value)
        return value

    def _parse_retry_after(self, retry_after: Optional[str]) -> float:
        if not retry_after:
            return 0.0
        try:
            return float(retry_after)
        except ValueError:
            return 0.0

    def _find_first_child(self, parent: ET.Element, local_name: str) -> Optional[ET.Element]:
        for child in list(parent):
            if self._local_name(child.tag) == local_name:
                return child
        return None

    def _find_first_element_child(self, parent: ET.Element) -> Optional[ET.Element]:
        for child in list(parent):
            if isinstance(child.tag, str):
                return child
        return None

    def _find_first_descendant(self, parent: ET.Element, local_name: str) -> Optional[ET.Element]:
        for node in parent.iter():
            if self._local_name(node.tag) == local_name:
                return node
        return None

    def _extract_text(self, parent: ET.Element, local_name: str) -> str:
        element = self._find_first_descendant(parent, local_name)
        if element is None or element.text is None:
            return ""
        return element.text.strip()

    def _local_name(self, tag: Any) -> str:
        if not isinstance(tag, str):
            return ""
        if "}" in tag:
            return tag.rsplit("}", 1)[-1]
        return tag

    def _normalize_whitespace(self, value: str) -> str:
        if not value:
            return ""
        return " ".join(value.split()).strip()
