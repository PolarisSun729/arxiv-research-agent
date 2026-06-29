"""修复数据库中作者名字的字符编码问题。

该脚本扫描 arxiv_oai_papers 表中的 authors 字段，检测并修复因 UTF-8/Latin-1
编码混淆导致的乱码问题。

使用方法:
    python fix_author_encoding.py [--dry-run] [--limit N]

参数:
    --dry-run: 只检测问题，不实际修改数据库
    --limit N: 限制处理的论文数量（默认处理全部）
"""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加父目录到路径，以便导入配置
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import OAI_SQLITE_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def is_likely_garbled(text: str) -> bool:
    """检测字符串是否可能包含编码错误的字符。

    UTF-8 被错误地用 Latin-1 解码后，会出现特定的字符模式：
    - Ä（\xc4）通常是 ć 的错误解码
    - Ã（\xc3）后跟重音字符，是常见的 UTF-8 乱码模式
    """
    if not text:
        return False

    # 检测常见的 UTF-8/Latin-1 混淆模式
    garbled_patterns = [
        '\xc4\x87',  # ć 的错误解码
        '\xc3\xb3',  # ó 的错误解码
        '\xc3\xa1',  # á 的错误解码
        '\xc3\xa9',  # é 的错误解码
        '\xc3\xad',  # í 的错误解码
        '\xc3\xba',  # ú 的错误解码
        '\xc3\xbc',  # ü 的错误解码
        '\xc3\xb1',  # ñ 的错误解码
    ]

    # 简化检测：如果包含 Ã 或 Ä 后跟非 ASCII 字符
    for i, char in enumerate(text):
        if char in '\xc3\xc4':
            if i + 1 < len(text) and ord(text[i + 1]) > 127:
                return True

    return False


def fix_garbled_text(text: str) -> Optional[str]:
    """尝试修复编码错误的文本。

    修复策略：将错误的 Latin-1 字符串重新编码为字节，然后用 UTF-8 解码。
    """
    if not text or not is_likely_garbled(text):
        return None

    try:
        # 将错误解码的字符串重新编码为原始字节（Latin-1），然后用正确的编码（UTF-8）解码
        fixed = text.encode('latin-1').decode('utf-8')

        # 验证修复是否成功（不应该还有乱码模式）
        if is_likely_garbled(fixed):
            return None

        return fixed
    except (UnicodeDecodeError, UnicodeEncodeError):
        # 修复失败，返回 None
        return None


def fix_authors_field(authors_json: str) -> Optional[str]:
    """修复 authors 字段（JSON 数组格式）中的编码问题。"""
    if not authors_json or not is_likely_garbled(authors_json):
        return None

    try:
        authors = json.loads(authors_json)
        if not isinstance(authors, list):
            return None

        fixed_authors = []
        has_changes = False

        for author in authors:
            author_str = str(author)
            if is_likely_garbled(author_str):
                fixed_author = fix_garbled_text(author_str)
                if fixed_author:
                    fixed_authors.append(fixed_author)
                    has_changes = True
                    logger.debug("Fixed author: %s -> %s", author_str, fixed_author)
                else:
                    fixed_authors.append(author_str)
            else:
                fixed_authors.append(author_str)

        if has_changes:
            return json.dumps(fixed_authors, ensure_ascii=False)
        return None
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Failed to parse authors JSON: %s", exc)
        return None


def scan_and_fix_database(db_path: str, dry_run: bool = False, limit: Optional[int] = None):
    """扫描并修复数据库中的编码问题。"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 查询所有论文的 arxiv_id 和 authors
    query = "SELECT arxiv_id, authors FROM arxiv_oai_papers"
    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)
    rows = cursor.fetchall()

    total_count = len(rows)
    garbled_count = 0
    fixed_count = 0
    failed_count = 0

    logger.info("Scanning %d papers for encoding issues...", total_count)

    for arxiv_id, authors_json in rows:
        if not authors_json or not is_likely_garbled(authors_json):
            continue

        garbled_count += 1
        logger.info("Found garbled authors in %s: %s", arxiv_id, authors_json[:100])

        fixed_json = fix_authors_field(authors_json)
        if fixed_json:
            logger.info("  Fixed to: %s", fixed_json[:100])

            if not dry_run:
                try:
                    cursor.execute(
                        "UPDATE arxiv_oai_papers SET authors = ? WHERE arxiv_id = ?",
                        (fixed_json, arxiv_id)
                    )
                    fixed_count += 1
                except Exception as exc:
                    logger.error("Failed to update %s: %s", arxiv_id, exc)
                    failed_count += 1
            else:
                fixed_count += 1
        else:
            logger.warning("  Failed to fix: %s", arxiv_id)
            failed_count += 1

    if not dry_run:
        conn.commit()
    conn.close()

    logger.info("\n=== Summary ===")
    logger.info("Total papers scanned: %d", total_count)
    logger.info("Papers with garbled authors: %d", garbled_count)
    logger.info("Successfully fixed: %d", fixed_count)
    logger.info("Failed to fix: %d", failed_count)

    if dry_run:
        logger.info("\n(Dry run mode - no changes were made to the database)")


def main():
    parser = argparse.ArgumentParser(
        description="Fix author name encoding issues in arXiv OAI database"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only detect issues, don't modify the database"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of papers to process"
    )

    args = parser.parse_args()

    db_path = OAI_SQLITE_CONFIG["database_path"]
    logger.info("Database path: %s", db_path)

    if not Path(db_path).exists():
        logger.error("Database file not found: %s", db_path)
        sys.exit(1)

    scan_and_fix_database(db_path, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
