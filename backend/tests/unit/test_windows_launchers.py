from __future__ import annotations

from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]
WINDOWS_LAUNCHERS = sorted((BACKEND_ROOT / "07-arxiv-tools").glob("*.cmd"))


def test_windows_launchers_exist() -> None:
    assert WINDOWS_LAUNCHERS


@pytest.mark.parametrize("launcher", WINDOWS_LAUNCHERS, ids=lambda path: path.name)
def test_windows_launcher_is_ascii_only(launcher: Path) -> None:
    # cmd.exe 按控制台代码页（中文 Windows 为 GBK）读取批处理，UTF-8 中文会让行偏移错位，
    # 中文注释之后的命令会被切成碎片执行；chcp 65001 也无法稳定规避，只能保持纯 ASCII。
    content = launcher.read_bytes()
    non_ascii_lines = [
        number for number, line in enumerate(content.splitlines(), start=1) if any(byte > 0x7F for byte in line)
    ]
    assert non_ascii_lines == [], f"{launcher.name} has non-ASCII bytes on lines {non_ascii_lines}"
