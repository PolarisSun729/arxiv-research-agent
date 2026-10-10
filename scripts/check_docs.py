"""维护文档相对链接检查：确认 docs/ 与 README.md 中的相对链接都指向仓库内真实存在的文件。

用途：
    文档里经常用相对路径链接其他文档或代码文件（如 [隔离入口](../../scripts/test_security.py)），
    文件改名、移动或删除后这些链接会悄悄失效。本脚本离线扫描所有维护文档，找出失效链接。
    它是 check_quality.py 的 docs 阶段，CI 中也会运行。

用法（任意目录运行均可）：
    python scripts/check_docs.py

检查规则：
    - 扫描 docs/**/*.md 与仓库根目录 README.md；
    - 只检查普通链接 [文字](目标)，图片链接 ![](...) 不检查；
    - http(s)、mailto 等外部链接和纯锚点 #xxx 直接跳过，不访问网络；
    - 目标中的 #锚点 和 ?查询参数 会被去掉，只检查文件或目录本身是否存在，不校验锚点是否存在；
    - 相对路径跳出仓库根目录也算失效。

退出码：0 = 全部有效；1 = 存在失效链接（逐条打印“文件:行号: 无效链接 目标”）。
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
# docs 下的维护文档由递归扫描覆盖；仓库 README 是唯一位于 docs 外的维护入口。
ENTRY_DOCUMENTS = (REPO_ROOT / "README.md",)
# 匹配 [文字](目标) 形式的链接；(?<!!) 排除图片语法 ![alt](src)。
# 按行匹配，所以跨行的链接、引用式链接 [文字][ref] 不在检查范围内。
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\((?P<target>[^)]+)\)")
# 以这些前缀开头的目标视为外部资源，不做存在性检查。
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "tel:", "data:")


def _iter_documents() -> list[Path]:
    """返回需要检查的 Markdown 文件列表：docs 下全部 .md（排序后）加上 README.md。"""
    documents = sorted(DOCS_ROOT.rglob("*.md")) if DOCS_ROOT.exists() else []
    documents.extend(path for path in ENTRY_DOCUMENTS if path.exists())
    return documents


def _is_external_or_anchor(target: str) -> bool:
    """判断链接目标是否为外部地址或页内锚点（不区分大小写）。"""
    normalized = target.strip().lower()
    return normalized.startswith("#") or normalized.startswith(EXTERNAL_PREFIXES)


def _resolve_target(document: Path, target: str) -> Path | None:
    """把 document 中的链接目标解析为绝对路径。

    返回 None 表示无需检查（外部链接、锚点或空路径）；
    返回 __outside_repository__ 占位路径表示链接跳出了仓库，调用方会因其不存在而报错。
    """
    normalized = target.strip()
    # Markdown 允许用尖括号包住含空格的路径：[文字](<path with space.md>)。
    if normalized.startswith("<") and normalized.endswith(">"):
        normalized = normalized[1:-1].strip()
    if _is_external_or_anchor(normalized):
        return None

    # Markdown 链接可以附带锚点或查询参数；路径存在性只检查文件部分。
    # unquote 把 %20 等 URL 编码还原成真实文件名中的字符。
    path_text = unquote(normalized.split("#", 1)[0].split("?", 1)[0]).strip()
    if not path_text:
        return None
    # 相对链接以所在文档的目录为基准解析，与 GitHub 等渲染器的行为一致。
    resolved = (document.parent / path_text).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        # 文档不应通过相对路径跳出仓库，否则 CI 无法保证链接在克隆环境中仍成立。
        return Path("__outside_repository__")
    return resolved


def validate_documents() -> list[str]:
    """逐行扫描所有文档，返回失效链接描述列表；列表为空表示全部有效。"""
    failures: list[str] = []
    for document in _iter_documents():
        for line_number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), start=1):
            for match in MARKDOWN_LINK_RE.finditer(line):
                target = match.group("target").strip()
                resolved = _resolve_target(document, target)
                if resolved is None:
                    continue
                if not resolved.exists():
                    relative_document = document.relative_to(REPO_ROOT).as_posix()
                    failures.append(f"{relative_document}:{line_number}: 无效链接 {target}")
    return failures


def main() -> int:
    """命令行入口：打印检查结果，有失效链接返回 1，否则返回 0。"""
    failures = validate_documents()
    if failures:
        print("文档链接校验失败：")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"文档链接校验通过：检查了 {len(_iter_documents())} 个维护入口文档。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
