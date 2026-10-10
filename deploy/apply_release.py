"""以普通部署用户安装离线发布包；失败只回退程序，不回滚用户数据库。"""

# 运行位置：生产服务器。publish_ssh.sh 把本文件、release_common.py 和 release.tar.gz
# 上传到 <root>/incoming/<release_id>/ 后，通过 SSH 以普通用户（arxiv）身份执行本脚本。
#
# 服务器部署目录布局（<root> 默认为 /opt/arxiv-research-agent）：
#   .cicd-layout              初始化标记；不存在时拒绝部署，防止误覆盖手工部署的旧目录
#   .deploy.lock              文件锁，保证同一时间只有一个部署在执行
#   last-deployment.json      最近一次成功部署的结果
#   incoming/<release_id>/    上传暂存区；安装成功后删除其中的发布包
#   releases/<release_id>/    每次发布一个独立目录，release_id = <SHA>-<run_id>-<attempt>
#     .venv -> venvs/...      指向该版本对应的依赖环境
#     backend/data 等         指向 shared/ 中对应目录的符号链接（见 SHARED_DIRECTORIES）
#   current  -> releases/X    当前运行版本，systemd 固定从这里启动
#   previous -> releases/Y    上一个版本，便于人工排查或回退
#   shared/                   跨版本持久化：.env.production、数据库、向量库、日志等
#   venvs/                    依赖环境；cache-<锁文件哈希> 链接到已完成安装的 venv，依赖不变时直接复用
#
# 部署流程：校验参数和哈希 -> 解压并校验发布包 -> 安装或复用 venv -> 链接 shared 数据
#          -> 原子切换 current -> 重启服务 -> 健康检查 -> 失败则自动切回上一版本。

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from release_common import (
    SHARED_DIRECTORIES, ReleaseError, extract_checked, sha256_file, validate_commit,
    validate_release, verify_runtime,
)


# 服务器上安装的 systemd 单元名（由 deploy/arxiv-agent-cicd.service 复制而来）。
SERVICE = "arxiv-agent.service"


def atomic_link(target: Path, link: Path) -> None:
    """原子地把 link 指向 target：先建临时链接，再用 rename 覆盖。

    os.replace 底层是 rename(2)，在同一文件系统内是原子操作，
    任何时刻读取 link 要么得到旧目标、要么得到新目标，不会出现链接不存在的空窗。
    """
    temporary = link.with_name(f".{link.name}-{uuid.uuid4().hex}")
    temporary.symlink_to(target, target_is_directory=True)
    try:
        os.replace(temporary, link)
    finally:
        # 替换成功后临时名已不存在；替换失败时清理残留的临时链接。
        temporary.unlink(missing_ok=True)


def current_release(root: Path) -> Path | None:
    """返回 current 链接实际指向的版本目录；首次部署（current 不存在）时返回 None。"""
    current = root / "current"
    if not current.is_symlink():
        if current.exists():
            raise ReleaseError("current 必须是版本链接，不能覆盖已有目录。")
        return None
    target = current.resolve(strict=True)
    # current 只能指向 releases/ 下的直接子目录，防止被篡改为指向其他位置。
    if target.parent != (root / "releases").resolve():
        raise ReleaseError("current 指向部署目录之外。")
    return target


def attach_shared_data(root: Path, release: Path) -> None:
    """在新版本目录中，为每个持久化目录创建指向 shared/ 的符号链接。"""
    for relative in SHARED_DIRECTORIES:
        shared = root / "shared" / relative
        # 首次部署时自动创建共享目录，权限 0700 只允许部署用户访问。
        shared.mkdir(parents=True, exist_ok=True, mode=0o700)
        link = release / relative
        # 打包阶段已剔除这些路径；如果仍然存在，说明发布包异常，拒绝继续以免数据被遮蔽。
        if link.exists() or link.is_symlink():
            raise ReleaseError(f"发布包占用了持久化路径：{relative}")
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(shared, target_is_directory=True)


def install_environment(root: Path, release: Path, dependency_hash: str) -> Path:
    """返回与 dependency_hash 对应的 venv 路径；已有完整缓存则复用，否则离线新建。"""
    environments = root / "venvs"
    environments.mkdir(exist_ok=True)
    # 缓存键是锁文件哈希：依赖完全不变的连续发布可以直接复用 venv，省去安装几百 MB 依赖的时间。
    cached = environments / f"cache-{dependency_hash}"
    if cached.is_symlink():
        environment = cached.resolve(strict=False)
        if environment.parent != environments.resolve():
            raise ReleaseError("依赖缓存指向部署目录之外。")
        # 就绪标记内容与哈希一致、关键可执行文件都在，才认为缓存可用；否则继续往下重新安装。
        marker = environment / ".release-ready"
        if (marker.is_file() and marker.read_text().strip() == dependency_hash
                and (environment / "bin/python").is_file() and (environment / "bin/gunicorn").is_file()):
            return environment
    elif cached.exists():
        raise ReleaseError("依赖缓存路径被普通文件占用。")

    # venv 的脚本含绝对路径，创建后不能再移动目录；只有完整安装成功后才发布缓存链接。
    # 安装失败的目录保留供排查，下一次部署使用新目录，不会复用半安装环境。
    environment = environments / f"{dependency_hash[:16]}-{uuid.uuid4().hex[:12]}"
    subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    python = environment / "bin/python"
    # --no-index：完全不访问 PyPI，只从包内 wheels 目录安装；
    # --require-hashes：每个 wheel 都必须与锁文件中的 SHA-256 一致；
    # --no-cache-dir：不在部署用户家目录积累 pip 缓存，节省磁盘。
    subprocess.run([
        str(python), "-m", "pip", "install", "--no-index", "--require-hashes", "--no-cache-dir",
        "--find-links", str(release / ".release/wheels"),
        "-r", str(release / ".release/requirements.lock"),
    ], check=True)
    # 检查已安装包之间的依赖声明是否互相满足。
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    (environment / ".release-ready").write_text(dependency_hash + "\n")
    atomic_link(environment, cached)
    return environment


class SystemdRuntime:
    """对 systemd 服务的最小封装：重启、停止、健康检查。测试中可替换为假实现。"""

    def restart(self) -> None:
        # sudo -n：非交互模式，需要密码时直接失败而不是卡住；
        # sudoers 只放行 restart/stop 这两条命令（见 docs/operations/cicd-deployment.md）。
        subprocess.run(["sudo", "-n", "systemctl", "restart", SERVICE], check=True)

    def stop(self) -> None:
        subprocess.run(["sudo", "-n", "systemctl", "stop", SERVICE], check=True)

    def healthy(self, release: Path) -> bool:
        """单次健康检查：服务 active、主进程运行在目标版本目录、/health 正常、认证层生效。"""
        if subprocess.run(["systemctl", "is-active", "--quiet", SERVICE]).returncode:
            return False
        pid = subprocess.check_output(["systemctl", "show", "--property=MainPID", "--value", SERVICE], text=True).strip()
        # 同时核对进程工作目录，防止旧进程或其他占用 8001 的服务让存活检查误报成功。
        if not pid.isdigit() or pid == "0" or Path(f"/proc/{pid}/cwd").resolve() != release / "backend":
            return False
        # 显式禁用代理，确保请求直连本机后端，不受 http_proxy 等环境变量影响。
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open("http://127.0.0.1:8001/health", timeout=3) as response:
            if response.status != 200:
                return False
        # 不带凭据访问受保护接口：预期返回 401，证明认证中间件已加载并在工作；
        # 如果直接返回 200，说明认证失效，这个版本不能上线。
        try:
            opener.open("http://127.0.0.1:8001/api/auth/me", timeout=3).close()
        except urllib.error.HTTPError as exc:
            return exc.code == 401
        return False

    def wait_healthy(self, release: Path, timeout: int) -> None:
        """每 2 秒检查一次，连续 3 次通过才算健康；超时则抛出 ReleaseError。"""
        deadline = time.monotonic() + timeout
        consecutive = 0
        while time.monotonic() < deadline:
            try:
                consecutive = consecutive + 1 if self.healthy(release) else 0
                if consecutive >= 3:
                    return
            except (OSError, urllib.error.URLError, subprocess.CalledProcessError):
                # 服务启动中连接被拒绝等属于正常过渡状态，计数清零后继续等待。
                consecutive = 0
            time.sleep(2)
        raise ReleaseError("新进程未通过存活和认证边界检查；请查看 systemd 日志。")


def activate_release(root: Path, release: Path, runtime: SystemdRuntime, timeout: int) -> None:
    """切换到新版本并验证；失败时自动恢复到切换前的版本。"""
    previous = current_release(root)
    try:
        atomic_link(release, root / "current")
        runtime.restart()
        runtime.wait_healthy(release, timeout)
    except Exception as failure:
        # 代码回退始终复用 shared 数据；恢复数据库快照会覆盖发布期间的新数据，不能自动执行。
        try:
            if previous is None:
                # 首次部署就失败：没有可回退的版本，删除 current 并停止服务。
                (root / "current").unlink(missing_ok=True)
                runtime.stop()
            else:
                atomic_link(previous, root / "current")
                runtime.restart()
                runtime.wait_healthy(previous, timeout)
        except Exception as rollback_failure:
            raise ReleaseError("发布失败，旧版本恢复检查也失败；请立即查看 arxiv-agent 日志。") from rollback_failure
        raise ReleaseError("发布失败，已恢复上一版程序；首次发布失败时服务保持停止。") from failure
    # 新版本上线成功后，记录上一个版本的位置。
    if previous is not None and previous != release:
        atomic_link(previous, root / "previous")


def apply_release(root: Path, archive: Path, expected_hash: str, commit: str, release_id: str, timeout: int) -> dict:
    """完整执行一次部署，返回部署结果（同时写入 last-deployment.json）。"""
    # ---- 1. 参数与前置条件校验，全部通过前不修改任何运行状态 ----
    validate_commit(commit)
    if not re.fullmatch(re.escape(commit) + r"-[0-9]+-[0-9]+", release_id):
        raise ReleaseError("版本目录必须绑定提交、Actions run ID 和 attempt。")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash) or sha256_file(archive) != expected_hash:
        raise ReleaseError("发布包 SHA-256 校验失败，未修改运行版本。")
    if not (root / "shared/.env.production").is_file():
        raise ReleaseError("缺少 shared/.env.production，请先完成服务器初始化。")
    if not (root / ".cicd-layout").is_file():
        raise ReleaseError("目录尚未初始化为 CI/CD 布局，不能覆盖旧部署。")
    releases = root / "releases"
    releases.mkdir(exist_ok=True)
    release = releases / release_id
    if release.exists():
        # 同一个 release_id 已经是当前版本：视为重复触发，确认健康后直接返回（幂等）。
        if current_release(root) == release:
            SystemdRuntime().wait_healthy(release, timeout)
            return {"status": "already_active", "commit": commit}
        # 目录存在但不是当前版本，可能是上次失败的残留，不复用，要求生成新的 attempt。
        raise ReleaseError("该发布目录已存在，请重跑 Actions 以生成新的 attempt。")
    # ---- 2. 解压并校验发布包内容 ----
    extract_checked(archive, release)
    manifest = validate_release(release, commit)
    # SSH 的 umask 可能较严格；代码和前端目录可读，shared 中的配置和数据仍保持私有权限。
    release.chmod(0o755)
    for path in release.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
    # ---- 3. 准备依赖环境与共享数据链接 ----
    environment = install_environment(root, release, manifest["dependency_hash"])
    attach_shared_data(root, release)
    # systemd 通过 current/.venv/bin/gunicorn 启动，因此回退代码时依赖环境也随之回退。
    (release / ".venv").symlink_to(environment, target_is_directory=True)
    # wheel 只用于安装，删掉本次包内的副本以控制小服务器磁盘占用；运行和回退使用独立 venv。
    shutil.rmtree(release / ".release/wheels")
    # ---- 4. 切换版本、重启并做健康检查（失败自动回退） ----
    activate_release(root, release, SystemdRuntime(), timeout)
    result = {"status": "deployed", "commit": commit, "release": release_id,
              "dependency_hash": manifest["dependency_hash"]}
    (root / "last-deployment.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """命令行入口，由 publish_ssh.sh 远程调用，参数见下方 argparse 定义。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--health-timeout", type=int, default=180)
    args = parser.parse_args()
    verify_runtime()
    # 禁止以 root 运行：部署用户只拥有部署目录，提权操作仅限 sudoers 放行的 systemctl 命令。
    if os.geteuid() == 0:
        raise ReleaseError("请使用普通部署用户；仅重启服务通过受限 sudo 执行。")
    root = args.root.resolve(strict=True)
    if root == Path("/") or not 10 <= args.health_timeout <= 900:
        raise ReleaseError("部署目录或检查超时不合法。")
    archive = args.archive.resolve(strict=True)
    # 只接受上传暂存区内的包，避免被指向服务器上任意位置的文件。
    if not archive.is_relative_to(root / "incoming"):
        raise ReleaseError("只接收 incoming 目录内的发布包。")
    # fcntl 仅在 Unix 上可用，放在函数内导入，使本模块在 Windows 上也能被测试导入。
    import fcntl
    # Actions 已串行化；服务器侧锁继续保护手工触发及其他 SSH 会话。
    with (root / ".deploy.lock").open("a") as lock:
        try:
            # LOCK_NB：拿不到锁立即失败，而不是排队等待。
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ReleaseError("已有部署正在执行，请等待完成。") from exc
        print(json.dumps(apply_release(root, archive, args.sha256, args.commit, args.release_id, args.health_timeout)))
        # 部署成功后删除上传的发布包，释放磁盘空间；失败时保留以便排查。
        archive.unlink()


if __name__ == "__main__":
    main()
