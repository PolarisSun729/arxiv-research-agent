"""在服务器本地创建初始管理员；密码通过隐藏输入读取，不进入命令历史或进程参数。

用途：
    默认 ALLOW_PUBLIC_REGISTRATION=false，前端无法自助注册，账号只能由运维在本机用此脚本创建。
    它直接写入 AUTH_DATABASE_PATH 指向的 SQLite 账号库，不经过 HTTP 接口。

子命令：
    create-admin   创建管理员账号（角色固定为 admin，忽略 --role）
    create-user    创建普通账号，角色由 --role 指定：researcher（默认）/ viewer / guest
    set-password   重置已有账号的密码，并撤销该账号所有已登录会话

用法示例：
    python scripts/manage_users.py create-admin --username alice --email alice@example.com
    python scripts/manage_users.py create-user --username bob --role viewer
    python scripts/manage_users.py set-password --username bob
    # 服务器上操作生产账号库时，显式指定生产配置：
    python scripts/manage_users.py create-admin --username alice --env-file /opt/arxiv-research-agent/shared/.env.production

配置读取：
    指定 --env-file 时只加载该文件（文件不存在直接报错，避免误写到开发库）；
    否则与后端一致，依次读取进程环境变量、仓库根目录 .env、backend/.env。
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path


# 把 backend 加入模块搜索路径，下面才能复用后端的 auth 包（密码哈希、账号存储、字段校验）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def main() -> int:
    """命令行入口：成功返回 0，参数错误、账号冲突或数据库问题返回 1。"""
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
    # 后端模块放在加载环境变量之后再导入，保证它们读到的配置（尤其是 AUTH_DATABASE_PATH）来自上面选定的环境文件。
    from auth.errors import AuthError
    from auth.passwords import hash_password
    from auth.schemas import AdminUserCreate
    from auth.settings import auth_database_path
    from auth.store import AuthStore
    from pydantic import ValidationError

    try:
        # getpass 在终端隐藏输入；要求输入两次，防止手误设置出无人知道的密码。
        # 密码强度规则由 AdminUserCreate 校验，72 字节上限来自 bcrypt。
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
            # 先用 Pydantic 模型校验用户名、邮箱和密码规则，再哈希密码写库；用户名重复时 store 抛出 AuthError。
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
