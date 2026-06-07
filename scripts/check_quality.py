from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = REPO_ROOT / "new_frontend"
QUALITY_TMP_ROOT = REPO_ROOT / "temp" / "quality-gate-tmp"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@dataclass(frozen=True)
class Stage:
    id: str
    title: str
    command: list[str]
    cwd: Path
    display_command: str
    description: str

    @property
    def reproduce_command(self) -> str:
        if self.cwd == REPO_ROOT:
            return self.display_command
        relative_cwd = self.cwd.relative_to(REPO_ROOT).as_posix()
        return f"cd {relative_cwd} && {self.display_command}"


@dataclass(frozen=True)
class StageResult:
    stage: Stage
    passed: bool
    duration_seconds: float
    return_code: int | None
    failure_summary: str
    stdout: str
    stderr: str


def _python_command(args: Iterable[str]) -> list[str]:
    return [sys.executable, *args]


def _npm_executable() -> str:
    # Windows 下直接调用 npm.ps1 容易受执行策略影响，优先使用 npm.cmd 保持入口跨平台稳定。
    preferred = "npm.cmd" if os.name == "nt" else "npm"
    return shutil.which(preferred) or shutil.which("npm") or preferred


NPM = _npm_executable()

STAGES: dict[str, Stage] = {
    "backend-static": Stage(
        id="backend-static",
        title="后端静态检查",
        command=_python_command(["scripts/backend_static_check.py"]),
        cwd=REPO_ROOT,
        display_command="python scripts/backend_static_check.py",
        description="编译所有后端 Python 文件，导入关键入口模块，并在可用时运行低误伤 ruff 规则。",
    ),
    "backend-compile": Stage(
        id="backend-compile",
        title="后端静态编译检查",
        command=_python_command(["-m", "compileall", "-q", "backend"]),
        cwd=REPO_ROOT,
        display_command="python -m compileall -q backend",
        description="提前发现 Python 语法错误和基础编译错误。",
    ),
    "backend-tests": Stage(
        id="backend-tests",
        title="后端自动化测试",
        command=_python_command(["-m", "pytest", "backend/tests", "--ignore=backend/tests/smoke"]),
        cwd=REPO_ROOT,
        display_command="python -m pytest backend/tests --ignore=backend/tests/smoke",
        description="通过 pytest 统一收集现有 unittest.TestCase 与 pytest 风格测试，不重复运行启动烟测。",
    ),
    "backend-startup-smoke": Stage(
        id="backend-startup-smoke",
        title="后端启动烟测",
        command=_python_command(["-m", "pytest", "backend/tests/smoke"]),
        cwd=REPO_ROOT,
        display_command="python -m pytest backend/tests/smoke",
        description="在 lazy 模式下创建 FastAPI app，检查路由、基础响应和统一错误结构。",
    ),
    "frontend-error-tests": Stage(
        id="frontend-error-tests",
        title="前端错误解析测试",
        command=[NPM, "run", "test:errors"],
        cwd=FRONTEND_ROOT,
        display_command="npm run test:errors",
        description="运行最轻量的前端错误解析回归检查。",
    ),
    "frontend-tests": Stage(
        id="frontend-tests",
        title="前端自动化测试",
        command=[NPM, "run", "test"],
        cwd=FRONTEND_ROOT,
        display_command="npm run test",
        description="运行 package.json 中已登记的前端测试脚本。",
    ),
    "frontend-build": Stage(
        id="frontend-build",
        title="前端构建检查",
        command=[NPM, "run", "build"],
        cwd=FRONTEND_ROOT,
        display_command="npm run build",
        description="确认 Vue / TypeScript / Vite 可以完成生产构建。",
    ),
    "doctor-basic": Stage(
        id="doctor-basic",
        title="环境体检 basic",
        command=_python_command(["scripts/doctor.py", "basic"]),
        cwd=REPO_ROOT,
        display_command="python scripts/doctor.py basic",
        description="只检查本地环境和本地依赖，不真实调用外部服务。",
    ),
    "doctor-full": Stage(
        id="doctor-full",
        title="环境体检 full",
        command=_python_command(["scripts/doctor.py", "full"]),
        cwd=REPO_ROOT,
        display_command="python scripts/doctor.py full",
        description="检查真实连接能力；可能计费的模型 API 默认仍会跳过。",
    ),
}

TARGETS: dict[str, list[str]] = {
    # 默认入口先跑本地环境体检，再进入离线质量检查；full doctor 仍保持显式触发，避免默认流程访问真实外部服务。
    "all": ["doctor-basic", "backend-static", "backend-tests", "backend-startup-smoke", "frontend-tests", "frontend-build"],
    "backend": ["backend-static", "backend-tests", "backend-startup-smoke"],
    "frontend": ["frontend-tests", "frontend-build"],
    "compile": ["backend-compile"],
    "static": ["backend-static"],
    "ci": ["doctor-basic", "backend-static", "backend-tests", "backend-startup-smoke", "frontend-tests", "frontend-build"],
    "doctor": ["doctor-basic"],
    # smoke 保留为离线轻量入口，用于快速确认语法、后端测试入口和前端最小脚本仍可运行。
    "smoke": ["backend-compile", "backend-startup-smoke", "frontend-error-tests"],
}


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _tail_non_empty_lines(text: str, limit: int = 18) -> list[str]:
    lines = [line.rstrip() for line in _strip_ansi(text).splitlines() if line.strip()]
    return lines[-limit:]


def _failure_context_lines(text: str, limit: int = 18) -> list[str]:
    raw_lines = _strip_ansi(text).splitlines()
    focused: list[str] = []
    capture_after = 0
    for raw_line in raw_lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        is_failure_header = bool(re.match(r"\[\d+\]\s+FAIL\b", stripped)) or stripped.startswith(("FAIL ", "ERROR "))
        is_exception_line = any(
            marker in stripped
            for marker in (
                "ImportError",
                "ModuleNotFoundError",
                "SyntaxError",
                "AssertionError",
                "TypeError",
                "ValueError",
                "npm ERR!",
                "error TS",
                "FAILED ",
            )
        )
        if is_failure_header or is_exception_line:
            # 质量入口的汇总需要优先保留真正失败项，避免 doctor 末尾的 SKIP/WARN 信息盖过 FAIL。
            focused.append(stripped)
            capture_after = 3
            continue
        if capture_after > 0:
            focused.append(stripped)
            capture_after -= 1

    return focused[:limit]


def _summarize_failure(stdout: str, stderr: str, return_code: int | None) -> str:
    combined = "\n".join(part for part in (stderr, stdout) if part.strip())
    focused = _failure_context_lines(combined)
    if focused:
        return "\n".join(focused)
    tail = _tail_non_empty_lines(combined)
    if not tail:
        return f"命令退出码 {return_code}，但没有输出更多错误信息。"
    return "\n".join(tail)


def _first_failure_line(summary: str) -> str:
    # 汇总表只放一行，优先挑出真正的异常/编译错误，避免被测试框架启动信息淹没。
    lines = [line.strip() for line in summary.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if re.match(r"\[\d+\]\s+FAIL\b", line):
            reason = next((candidate for candidate in lines[index + 1 : index + 4] if candidate.startswith("原因:")), "")
            return f"{line} - {reason}" if reason else line

    markers = (
        "E   ",
        "ImportError",
        "ModuleNotFoundError",
        "SyntaxError",
        "AssertionError",
        "TypeError",
        "ValueError",
        "npm ERR!",
        "error TS",
        "ERROR ",
        "FAILED ",
    )
    for marker in markers:
        for line in lines:
            if marker in line:
                return line
    for line in lines:
        if any(marker.lower() in line.lower() for marker in ("error", "failed", "exception")):
            return line
    return lines[0] if lines else "无失败摘要"


def _run_stage(stage: Stage, show_output: bool) -> StageResult:
    print(f"\n=== {stage.title} [{stage.id}] ===")
    print(f"职责: {stage.description}")
    print(f"命令: {stage.reproduce_command}")
    start = time.perf_counter()
    env = os.environ.copy()
    # 质量门禁只编排离线安全检查；该标记便于后续测试代码需要时显式识别当前运行模式。
    env.setdefault("RAG_QUALITY_GATE", "offline")
    env.setdefault("CI", "1")
    # Windows 本机临时目录偶尔会因权限或残留锁导致 pytest tmp_path 初始化失败；统一落到仓库内临时目录更适合本地和 CI 复现。
    QUALITY_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    env.setdefault("TMP", str(QUALITY_TMP_ROOT))
    env.setdefault("TEMP", str(QUALITY_TMP_ROOT))
    env.setdefault("TMPDIR", str(QUALITY_TMP_ROOT))

    try:
        completed = subprocess.run(
            stage.command,
            cwd=stage.cwd,
            env=env,
            text=True,
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        duration = time.perf_counter() - start
        passed = completed.returncode == 0
        if show_output or not passed:
            if completed.stdout:
                print("\n[stdout]")
                print(completed.stdout.rstrip())
            if completed.stderr:
                print("\n[stderr]")
                print(completed.stderr.rstrip())
        if passed:
            print(f"结果: PASS ({duration:.1f}s)")
            summary = ""
        else:
            summary = _summarize_failure(completed.stdout, completed.stderr, completed.returncode)
            print(f"结果: FAIL ({duration:.1f}s, exit={completed.returncode})")
            print("失败摘要:")
            print(summary)
        return StageResult(
            stage=stage,
            passed=passed,
            duration_seconds=duration,
            return_code=completed.returncode,
            failure_summary=summary,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except FileNotFoundError as exc:
        duration = time.perf_counter() - start
        # 依赖缺失要明确落到阶段结果里，避免用户只看到 Python traceback。
        summary = f"命令不可用: {exc.filename}. 请先安装对应依赖，或进入相关目录执行依赖安装。"
        print(f"结果: FAIL ({duration:.1f}s)")
        print("失败摘要:")
        print(summary)
        return StageResult(
            stage=stage,
            passed=False,
            duration_seconds=duration,
            return_code=None,
            failure_summary=summary,
            stdout="",
            stderr=str(exc),
        )


def _expand_targets(targets: list[str]) -> list[Stage]:
    selected_ids: list[str] = []
    for target in targets:
        ids = TARGETS.get(target, [target])
        for stage_id in ids:
            if stage_id not in selected_ids:
                selected_ids.append(stage_id)
    return [STAGES[stage_id] for stage_id in selected_ids]


def _print_summary(results: list[StageResult]) -> None:
    print("\n=== 质量门禁汇总 ===")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        exit_text = "exit=0" if result.passed else f"exit={result.return_code}"
        print(f"{status:4} {result.stage.id:22} {result.duration_seconds:6.1f}s  {exit_text}")
        if not result.passed:
            print(f"     复现: {result.stage.reproduce_command}")
            print(f"     摘要: {_first_failure_line(result.failure_summary)}")

    failed = [result for result in results if not result.passed]
    if failed:
        print(f"\n最终结果: FAIL，失败阶段 {len(failed)} 个。")
    else:
        print("\n最终结果: PASS，所有选中阶段均通过。")


def _parse_args() -> argparse.Namespace:
    choices = sorted(set(TARGETS) | set(STAGES))
    parser = argparse.ArgumentParser(
        description="统一质量门禁入口：后端编译、后端测试、前端测试和前端构建。",
    )
    parser.add_argument(
        "targets",
        nargs="*",
        choices=choices,
        help="要运行的目标，默认 all。常用值：backend、frontend、static、smoke。",
    )
    parser.add_argument("--fail-fast", action="store_true", help="首个阶段失败后立即停止。")
    parser.add_argument("--show-output", action="store_true", help="通过阶段也打印完整 stdout/stderr。")
    parser.add_argument("--list", action="store_true", help="列出可运行目标和阶段后退出。")
    return parser.parse_args()


def _print_available_targets() -> None:
    print("可用目标:")
    for name, stage_ids in TARGETS.items():
        print(f"  {name:10} -> {', '.join(stage_ids)}")
    print("\n可用阶段:")
    for stage in STAGES.values():
        print(f"  {stage.id:22} {stage.reproduce_command}")


def main() -> int:
    args = _parse_args()
    if args.list:
        _print_available_targets()
        return 0

    targets = args.targets or ["all"]
    stages = _expand_targets(targets)
    print(f"仓库根目录: {REPO_ROOT}")
    print(f"选中目标: {', '.join(targets)}")

    results: list[StageResult] = []
    for stage in stages:
        result = _run_stage(stage, show_output=args.show_output)
        results.append(result)
        if args.fail_fast and not result.passed:
            # fail-fast 用于本地快速定位；默认仍继续执行，确保最终汇总能覆盖每个选中阶段。
            print("\n已启用 --fail-fast，停止后续阶段。")
            break

    _print_summary(results)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
