"""三阶段安全测试入口：仅使用合成凭据、临时数据库及测试提供的业务替身。"""

# 用途：
#     在与开发/生产配置完全隔离的环境里运行安全相关的后端回归，结果不受本机 .env 影响，
#     也绝不会读写真实账号库、业务库或审计日志。
#     三个阶段对应 backend/tests/api/test_security_stage1~3.py：
#         stage1  API Key 认证
#         stage2  限流、IP 过滤、审计日志
#         stage3  JWT 账号、用户管理、配额
#     另外包含 init_security.py 的初始化测试。
#
# 用法（在任意目录运行均可，脚本会切换到仓库根目录）：
#     python scripts/test_security.py                       # 只跑上面四个安全测试文件
#     python scripts/test_security.py -k "redact"           # 按 pytest -k 表达式筛选用例
#     python scripts/test_security.py --full                # 跑全部 backend/tests，同样隔离
#
# 隔离手段（见 main）：
#     1. 清空进程环境变量，只保留系统运行必需的白名单，再写入一套合成配置和随机凭据；
#     2. 所有数据库、缓存、trace、审计日志都指向 temp/security-tests/run-*/ 下的新目录；
#     3. 拦截 python-dotenv 与 Starlette Config 读取文件，只允许读取本次运行目录内的 dotenv；
#     4. 注册审计钩子，一旦有代码试图打开默认账号库就立即报错。
#
# 退出码与 pytest 一致：0 表示全部通过；缺少 pytest/dotenv 时返回 2。
# 运行目录不会自动删除，其中保留 results.xml（JUnit 报告）和审计日志，便于事后排查。

from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
# 默认模式下运行的测试文件（相对仓库根目录）；--full 时改为整个 backend/tests。
SECURITY_TESTS = [
    "backend/tests/api/test_security_stage1.py",
    "backend/tests/api/test_security_stage2.py",
    "backend/tests/api/test_security_stage3.py",
    "backend/tests/unit/test_security_initialization.py",
]


def isolate_configuration_reads(runtime: Path) -> None:
    """允许本次测试生成的配置，阻止 dotenv 和 SlowAPI/Starlette 隐式读取真实部署文件。

    做法是给两个读取入口打猴子补丁：路径位于 runtime 目录内才放行，否则当作文件不存在。
    补丁作用于当前进程，pytest.main 在同一进程内运行，因此对全部测试生效。
    """
    import dotenv
    from starlette.config import Config

    original_load_dotenv = dotenv.load_dotenv
    original_read_file = Config._read_file

    def load_test_dotenv(dotenv_path=None, *extra_args, **kwargs):
        # CLI 回归有自己的临时部署文件，不能一概禁用 dotenv 而落到默认账号库。
        # dotenv_path 为 None 时 python-dotenv 会自动向上查找 .env，这里一律拒绝；返回 False 等同于“未找到文件”。
        if dotenv_path is None or not Path(dotenv_path).resolve().is_relative_to(runtime):
            return False
        return original_load_dotenv(dotenv_path, *extra_args, **kwargs)

    def read_test_config(self, file_name, encoding="utf-8"):
        # SlowAPI 使用 Starlette Config 直接打开 .env；只拦 python-dotenv 并不能隔离这个入口。
        if not Path(file_name).resolve().is_relative_to(runtime):
            return {}
        return original_read_file(self, file_name, encoding)

    dotenv.load_dotenv = load_test_dotenv
    Config._read_file = read_test_config


def main() -> int:
    """命令行入口：搭建隔离环境后在当前进程内调用 pytest，返回 pytest 的退出码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="运行全部后端离线回归，包含三个安全阶段。")
    parser.add_argument("-k", "--filter", default="", help="按 pytest 表达式选择回归，沿用相同的凭据和数据库隔离。")
    args = parser.parse_args()

    # 提前确认依赖可用，给出明确的安装提示，而不是在测试中途报 ImportError。
    try:
        import dotenv
        import pytest
    except ImportError as exc:
        print(f"缺少测试依赖 {exc.name}，请在项目 Python 环境中安装 requirements-dev.txt。", file=sys.stderr)
        return 2

    # SECURITY_TESTS 是相对路径，pytest 也要从仓库根目录收集 conftest。
    os.chdir(REPO_ROOT)
    scratch = REPO_ROOT / "temp" / "security-tests"
    scratch.mkdir(parents=True, exist_ok=True)
    # 每次创建独立目录且保留 JUnit 与日志，既不覆盖旧证据，也不清理其他运行的数据。
    runtime = Path(tempfile.mkdtemp(prefix="run-", dir=scratch))

    # 使用系统运行必需的环境白名单，避免从部署 shell 继承真实凭据、数据库或 pytest 插件。
    system_names = {
        "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "SYSTEMDRIVE",
        "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA",
        "USERNAME", "USER", "LOGNAME", "LNAME",
        "LANG", "LC_ALL", "TERM", "TZ", "VIRTUAL_ENV", "CONDA_PREFIX",
    }
    environment = {name: value for name, value in os.environ.items() if name.upper() in system_names}
    # 在白名单基础上写入测试专用配置：
    #   - 认证用 api_key 模式，访问密钥与 JWT 密钥每次随机生成，测试结束即作废；
    #   - 所有 SQLite、缓存、trace、审计日志都放进本次 runtime 目录；
    #   - 模型密钥置空、HuggingFace 离线、限流用内存存储，保证不访问任何外部服务；
    #   - RAG_QUALITY_GATE=offline / CI=1 让后端测试按离线质量门禁的约定运行。
    environment.update({
        "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "RAG_QUALITY_GATE": "offline", "CI": "1",
        "TMP": str(runtime), "TEMP": str(runtime), "TMPDIR": str(runtime),
        "AUTH_MODE": "api_key", "BACKEND_API_KEYS": secrets.token_urlsafe(32), "API_KEY_CONFIG_FILE": "",
        "JWT_SECRET_KEY": secrets.token_urlsafe(48), "JWT_SECRET_FILE": "", "ALLOW_PUBLIC_REGISTRATION": "false",
        "AUTH_DATABASE_PATH": str(runtime / "auth.sqlite3"),
        "SQLITE_DATABASE_PATH": str(runtime / "business.sqlite3"),
        "OAI_SQLITE_DATABASE_PATH": str(runtime / "oai.sqlite3"),
        "PAPER_QA_BUILD_CACHE_DIR": str(runtime / "qa-cache"),
        "BACKEND_REQUEST_TRACE_DIR": str(runtime / "request-traces"),
        "RETRIEVAL_TRACE_EXPORT_DIR": str(runtime / "retrieval-traces"),
        "AGENT_RUNTIME_CHECKPOINT_BACKEND": "memory", "BACKEND_SERVICE_LOAD_MODE": "lazy",
        "BACKEND_ACCESS_LOG": "false", "ENABLE_DEBUG_ROUTES": "false",
        "ALLOWED_ORIGINS": "https://research.example", "ALIYUN_API_KEY": "",
        "RATE_LIMIT_STORAGE": "memory://", "IP_FILTER_MODE": "disabled",
        "IP_BLACKLIST": "", "IP_WHITELIST": "", "TRUSTED_PROXY_IPS": "",
        "AUDIT_LOG_ENABLED": "true", "AUDIT_LOG_FILE": str(runtime / "audit.log"),
        "ARXIV_DATA_SOURCE": "api", "ARXIV_PROXY_URL": "",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TORCHINDUCTOR_CACHE_DIR": str(runtime / "torch-cache"),
    })
    # 普通回归不应碰到默认限额；限流测试在各自 fixture 内设置小额度并断言 429。
    for name in (
        "RATE_LIMIT_IP", "RATE_LIMIT_KEY", "RATE_LIMIT_USER", "RATE_LIMIT_AUTH", "RATE_LIMIT_LOGIN_ACCOUNT",
        "RATE_LIMIT_READ", "RATE_LIMIT_WRITE", "RATE_LIMIT_EXPENSIVE", "AUTH_FAILURE_LIMIT", "RATE_LIMIT_VIOLATION_LIMIT",
    ):
        environment[name] = "10000/minute"
    # 用隔离后的环境整体替换当前进程环境；tempfile 的默认目录也指向 runtime，
    # 测试里创建的临时文件都会留在本次运行目录中。
    os.environ.clear()
    os.environ.update(environment)
    sys.dont_write_bytecode = True
    tempfile.tempdir = str(runtime)

    isolate_configuration_reads(runtime)

    def guard_default_auth_database(event, audit_args):
        # 测试故意删除环境变量时，禁止 CLI 意外回退到真实账号库；失败要停在写入之前。
        if event == "sqlite3.connect" and str(audit_args[0]) != ":memory:":
            if Path(audit_args[0]).resolve() == REPO_ROOT / "backend/data/auth/auth.sqlite3":
                raise RuntimeError("安全测试不能打开默认账号库，请使用临时 AUTH_DATABASE_PATH。")

    # Python 审计钩子一经注册无法移除，覆盖本进程后续所有 sqlite3.connect 调用。
    sys.addaudithook(guard_default_auth_database)
    paths = ["backend/tests"] if args.full else SECURITY_TESTS
    print("运行全部后端回归。" if args.full else "运行 API Key、限流/IP/审计、JWT/用户/配额三阶段安全回归。", flush=True)
    print(f"测试证据目录：{runtime}", flush=True)
    # --basetemp / cache_dir 让 pytest 自身的临时文件和缓存也落在 runtime 中，
    # --junitxml 把结果保存为证据文件。
    result = int(pytest.main([
        *paths, "-q", "--basetemp", str(runtime / "pytest"),
        "--junitxml", str(runtime / "results.xml"), "-o", f"cache_dir={runtime / 'pytest-cache'}",
        *(["-k", args.filter] if args.filter else []),
    ]))
    if result == 0:
        print("离线回归通过；真实 HTTPS、Redis 持久化及模型/向量库连接仍需部署验收。")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
