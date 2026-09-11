"""在服务器本地创建初始管理员；密码通过隐藏输入读取，不进入命令历史或进程参数。"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def main() -> int:
    parser = argparse.ArgumentParser(description="创建 arXiv 账号（读取 AUTH_DATABASE_PATH；可用 --env-file 指定部署配置）")
    parser.add_argument("command", choices=["create-admin", "create-user", "set-password"])
    parser.add_argument("--username", required=True)
    parser.add_argument("--email")
    parser.add_argument("--role", choices=["researcher", "viewer", "guest"], default="researcher")
    parser.add_argument("--env-file")
    args = parser.parse_args()
    from dotenv import load_dotenv
    if args.env_file:
        # 指定部署文件缺失时不能悄悄回退到默认数据库，避免把管理员建到错误环境。
        if not Path(args.env_file).is_file():
            print("指定的环境配置文件不存在。", file=sys.stderr)
            return 1
        load_dotenv(args.env_file, override=False)
    else:
        # 与应用保持相同优先级：进程环境 > 仓库根 .env > backend/.env，不依赖当前工作目录。
        repo_root = Path(__file__).resolve().parents[1]
        load_dotenv(repo_root / ".env", override=False)
        load_dotenv(repo_root / "backend" / ".env", override=False)
    from auth.errors import AuthError
    from auth.passwords import hash_password
    from auth.schemas import AdminUserCreate
    from auth.settings import auth_database_path
    from auth.store import AuthStore
    from pydantic import ValidationError

    try:
        password = getpass.getpass("密码（至少8字符，包含大小写字母和数字，最多72字节）：")
        if password != getpass.getpass("再次输入密码："):
            print("两次密码不一致。", file=sys.stderr)
            return 1
        store = AuthStore(auth_database_path())
        if args.command == "set-password":
            # 仅操作员本地恢复密码，不添加无需旧密码的公网重置入口。
            user = store.get_by_username(args.username)
            if user is None:
                raise AuthError("user_not_found")
            store.change_password(user, hashed_password=hash_password(password))
            print("密码已更新，旧登录会话已全部撤销。")
        else:
            payload = AdminUserCreate(username=args.username, email=args.email or "", password=password,
                                      role="admin" if args.command == "create-admin" else args.role)
            user = store.create_user(username=payload.username, email=str(payload.email), role=payload.role,
                                     hashed_password=hash_password(payload.password.get_secret_value()))
            print(f"账号已创建：{user.username}，角色：{user.role}，ID：{user.user_id}")
        return 0
    except (AuthError, RuntimeError, ValueError, ValidationError) as exc:
        # Pydantic 可能在错误文本中包含输入，CLI 也不输出原始校验异常和密码。
        print(exc.message if isinstance(exc, AuthError) else "操作失败，请检查账号字段、密码规则和认证数据库配置。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
