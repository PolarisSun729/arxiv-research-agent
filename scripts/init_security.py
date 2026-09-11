"""生成独立的本地/生产安全配置；不回显凭据、不覆盖已有配置或签名材料。"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOMAIN = "arxiv.001769.xyz"
_DOMAIN_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")


@lru_cache(maxsize=1)
def _windows_user_sid() -> str:
    result = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"], check=True, capture_output=True)
    # 本机命令仍可能输出 GBK；只消费纯 ASCII 的 SID，避免 UTF-8 模式下读线程解码异常。
    sid = next(csv.reader(result.stdout.decode("utf-8", errors="replace").strip().splitlines()))[-1]
    if not re.fullmatch(r"S-1-[0-9-]+", sid):
        raise RuntimeError("无法识别当前 Windows 用户。")
    return sid


def _write_private_file(path: Path, text: str) -> None:
    # 独占创建同时防止误覆盖与符号链接替换；已有凭据必须走明确的轮换流程。
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        if os.name == "nt":
            # Windows 的 chmod 不控制 NTFS 读取权限，写入秘密之前先移除继承并仅授权当前账号。
            subprocess.run(
                ["icacls", str(path), "/inheritance:r", "/grant:r", f"*{_windows_user_sid()}:(F)"],
                check=True, capture_output=True,
            )
        stream.write(text)


def initialize_security_config(repo_root: Path, *, profile: str, domain: str = DEFAULT_DOMAIN) -> tuple[Path, Path]:
    if profile not in {"development", "production"}:
        raise ValueError("未知的配置类型。")
    domain = domain.strip().lower()
    if len(domain) > 253 or "." not in domain or any(not _DOMAIN_LABEL.fullmatch(label) for label in domain.split(".")):
        raise ValueError("域名必须是完整主机名，不能包含协议、端口、路径或换行。")

    repo_root = repo_root.resolve()
    production = profile == "production"
    env_path = repo_root / (".env.production" if production else ".env")
    secret_path = repo_root / "backend" / "config" / f"{profile}.jwt-secret"
    # 先检查所有目标，再生成秘密；重复执行不能悄悄让现有会话或存储账号失效。
    for path in (env_path, secret_path):
        if not path.resolve().is_relative_to(repo_root) or path.exists() or path.is_symlink():
            raise FileExistsError("配置或签名文件已存在，或目标路径不在项目目录内。")
    template = (repo_root / ".env.example").read_text(encoding="utf-8")

    jwt_secret = secrets.token_urlsafe(48)
    redis_password = secrets.token_urlsafe(32)
    minio_password = secrets.token_urlsafe(48)
    values = {
        "AUTH_MODE": "jwt", "JWT_SECRET_KEY": "",
        "JWT_SECRET_FILE": secret_path.relative_to(repo_root).as_posix(),
        "AUTH_DATABASE_PATH": "backend/data/auth/production.sqlite3" if production else "backend/data/auth/auth.sqlite3",
        "ALLOW_PUBLIC_REGISTRATION": "false", "BACKEND_HOST": "127.0.0.1", "ENABLE_DEBUG_ROUTES": "false",
        "ALLOWED_ORIGINS": f"https://{domain}" if production else "http://localhost:5173,http://127.0.0.1:5173",
        "TRUSTED_PROXY_IPS": "127.0.0.1,::1" if production else "",
        "RATE_LIMIT_STORAGE": f"redis://:{quote(redis_password, safe='')}@127.0.0.1:6379/0" if production else "memory://",
        "REDIS_PASSWORD": redis_password,
        "MINIO_ROOT_USER": "arxiv_" + secrets.token_hex(8), "MINIO_ROOT_PASSWORD": minio_password,
        "AUDIT_LOG_ENABLED": "true", "AUDIT_LOG_FILE": "-" if production else "logs/audit.log",
    }
    if production:
        # 沿用小型部署的后台并发预算；生成配置并不代表 Docling 与模型服务已通过容量验收。
        values.update({
            "PAPER_QA_BUILD_LLM_MAX_WORKERS": "2", "PROFILE_EVIDENCE_MAX_WORKERS": "1",
            "RETRIEVAL_ROUTE_MAX_WORKERS": "2", "DOCLING_OCR_ENABLED": "true",
            "DOCLING_GENERATE_PAGE_IMAGES": "true", "DOCLING_GENERATE_PICTURE_IMAGES": "true",
            "DOCLING_IMAGES_SCALE": "2.0",
        })
    lines: list[str] = []
    remaining = dict(values)
    for line in template.splitlines():
        key, separator, _value = line.partition("=")
        if separator and key in remaining:
            line = f"{key}={remaining.pop(key)}"
        lines.append(line)
    lines.extend(f"{key}={value}" for key, value in remaining.items())

    secret_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _write_private_file(secret_path, jwt_secret + "\n")
    _write_private_file(env_path, "\n".join(lines) + "\n")
    return env_path, secret_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["development", "production"], required=True)
    parser.add_argument("--domain", default=DEFAULT_DOMAIN, help="生产前端 HTTPS 主机名（不含协议或路径）。")
    args = parser.parse_args()
    try:
        env_path, secret_path = initialize_security_config(REPO_ROOT, profile=args.profile, domain=args.domain)
    except FileExistsError:
        print("初始化停止：配置或签名文件已存在，保留原文件；请检查后手工修改，不能用初始化替代轮换。", file=sys.stderr)
        return 1
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
        # 系统异常可能带进配置上下文；只给稳定提示，已创建文件保留供检查，绝不输出凭据。
        print("初始化未完成：请检查域名、模板与文件权限；已创建的文件不会被覆盖或删除。", file=sys.stderr)
        return 1
    print(f"已创建配置：{env_path}")
    print(f"已创建私有签名文件：{secret_path}")
    print("凭据已独立随机生成；请在配置文件中填写模型供应商密钥，并使用 manage_users.py 创建管理员。")
    if args.profile == "production":
        print("生产配置还需在服务器完成 Redis、DNS、HTTPS 证书及 Nginx 验收。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
