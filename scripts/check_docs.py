from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
# docs 下的维护文档由递归扫描覆盖；仓库 README 是唯一位于 docs 外的维护入口。
ENTRY_DOCUMENTS = (REPO_ROOT / "README.md",)
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\((?P<target>[^)]+)\)")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "tel:", "data:")


def _iter_documents() -> list[Path]:
    documents = sorted(DOCS_ROOT.rglob("*.md")) if DOCS_ROOT.exists() else []
    documents.extend(path for path in ENTRY_DOCUMENTS if path.exists())
    return documents


def _is_external_or_anchor(target: str) -> bool:
    normalized = target.strip().lower()
    return normalized.startswith("#") or normalized.startswith(EXTERNAL_PREFIXES)


def _resolve_target(document: Path, target: str) -> Path | None:
    normalized = target.strip()
    if normalized.startswith("<") and normalized.endswith(">"):
        normalized = normalized[1:-1].strip()
    if _is_external_or_anchor(normalized):
        return None

    # Markdown 链接可以附带锚点或查询参数；路径存在性只检查文件部分。
    path_text = unquote(normalized.split("#", 1)[0].split("?", 1)[0]).strip()
    if not path_text:
        return None
    resolved = (document.parent / path_text).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        # 文档不应通过相对路径跳出仓库，否则 CI 无法保证链接在克隆环境中仍成立。
        return Path("__outside_repository__")
    return resolved


def validate_documents() -> list[str]:
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
