"""三阶段安全测试入口：仅使用合成凭据、临时数据库及测试提供的业务替身。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
SECURITY_TESTS = [
    "backend/tests/api/test_security_stage1.py",
    "backend/tests/api/test_security_stage2.py",
    "backend/tests/api/test_security_stage3.py",
    "backend/tests/unit/test_security_initialization.py",
]


def isolate_configuration_reads(runtime: Path) -> None:
    """允许本次测试生成的配置，阻止 dotenv 和 SlowAPI/Starlette 隐式读取真实部署文件。"""
    import dotenv
    from starlette.config import Config

    original_load_dotenv = dotenv.load_dotenv
    original_read_file = Config._read_file

    def load_test_dotenv(dotenv_path=None, *extra_args, **kwargs):
        # CLI 回归有自己的临时部署文件，不能一概禁用 dotenv 而落到默认账号库。
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="运行全部后端离线回归，包含三个安全阶段。")
    parser.add_argument("-k", "--filter", default="", help="按 pytest 表达式选择回归，沿用相同的凭据和数据库隔离。")
    args = parser.parse_args()

    try:
        import dotenv
        import pytest
    except ImportError as exc:
        print(f"缺少测试依赖 {exc.name}，请在项目 Python 环境中安装 requirements.txt。", file=sys.stderr)
        return 2

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

    sys.addaudithook(guard_default_auth_database)
    paths = ["backend/tests"] if args.full else SECURITY_TESTS
    print("运行全部后端回归。" if args.full else "运行 API Key、限流/IP/审计、JWT/用户/配额三阶段安全回归。", flush=True)
    print(f"测试证据目录：{runtime}", flush=True)
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
