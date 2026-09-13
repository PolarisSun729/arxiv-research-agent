"""以普通部署用户安装离线发布包；失败只回退程序，不回滚用户数据库。"""

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


SERVICE = "arxiv-agent.service"


def atomic_link(target: Path, link: Path) -> None:
    temporary = link.with_name(f".{link.name}-{uuid.uuid4().hex}")
    temporary.symlink_to(target, target_is_directory=True)
    try:
        os.replace(temporary, link)
    finally:
        temporary.unlink(missing_ok=True)


def current_release(root: Path) -> Path | None:
    current = root / "current"
    if not current.is_symlink():
        if current.exists():
            raise ReleaseError("current 必须是版本链接，不能覆盖已有目录。")
        return None
    target = current.resolve(strict=True)
    if target.parent != (root / "releases").resolve():
        raise ReleaseError("current 指向部署目录之外。")
    return target


def attach_shared_data(root: Path, release: Path) -> None:
    for relative in SHARED_DIRECTORIES:
        shared = root / "shared" / relative
        shared.mkdir(parents=True, exist_ok=True, mode=0o700)
        link = release / relative
        if link.exists() or link.is_symlink():
            raise ReleaseError(f"发布包占用了持久化路径：{relative}")
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(shared, target_is_directory=True)


def install_environment(root: Path, release: Path, dependency_hash: str) -> Path:
    environments = root / "venvs"
    environments.mkdir(exist_ok=True)
    cached = environments / f"cache-{dependency_hash}"
    if cached.is_symlink():
        environment = cached.resolve(strict=False)
        if environment.parent != environments.resolve():
            raise ReleaseError("依赖缓存指向部署目录之外。")
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
    subprocess.run([
        str(python), "-m", "pip", "install", "--no-index", "--require-hashes", "--no-cache-dir",
        "--find-links", str(release / ".release/wheels"),
        "-r", str(release / ".release/requirements.lock"),
    ], check=True)
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    (environment / ".release-ready").write_text(dependency_hash + "\n")
    atomic_link(environment, cached)
    return environment


class SystemdRuntime:
    def restart(self) -> None:
        subprocess.run(["sudo", "-n", "systemctl", "restart", SERVICE], check=True)

    def stop(self) -> None:
        subprocess.run(["sudo", "-n", "systemctl", "stop", SERVICE], check=True)

    def healthy(self, release: Path) -> bool:
        if subprocess.run(["systemctl", "is-active", "--quiet", SERVICE]).returncode:
            return False
        pid = subprocess.check_output(["systemctl", "show", "--property=MainPID", "--value", SERVICE], text=True).strip()
        # 同时核对进程工作目录，防止旧进程或其他占用 8001 的服务让存活检查误报成功。
        if not pid.isdigit() or pid == "0" or Path(f"/proc/{pid}/cwd").resolve() != release / "backend":
            return False
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open("http://127.0.0.1:8001/health", timeout=3) as response:
            if response.status != 200:
                return False
        try:
            opener.open("http://127.0.0.1:8001/api/auth/me", timeout=3).close()
        except urllib.error.HTTPError as exc:
            return exc.code == 401
        return False

    def wait_healthy(self, release: Path, timeout: int) -> None:
        deadline = time.monotonic() + timeout
        consecutive = 0
        while time.monotonic() < deadline:
            try:
                consecutive = consecutive + 1 if self.healthy(release) else 0
                if consecutive >= 3:
                    return
            except (OSError, urllib.error.URLError, subprocess.CalledProcessError):
                consecutive = 0
            time.sleep(2)
        raise ReleaseError("新进程未通过存活和认证边界检查；请查看 systemd 日志。")


def activate_release(root: Path, release: Path, runtime: SystemdRuntime, timeout: int) -> None:
    previous = current_release(root)
    try:
        atomic_link(release, root / "current")
        runtime.restart()
        runtime.wait_healthy(release, timeout)
    except Exception as failure:
        # 代码回退始终复用 shared 数据；恢复数据库快照会覆盖发布期间的新数据，不能自动执行。
        try:
            if previous is None:
                (root / "current").unlink(missing_ok=True)
                runtime.stop()
            else:
                atomic_link(previous, root / "current")
                runtime.restart()
                runtime.wait_healthy(previous, timeout)
        except Exception as rollback_failure:
            raise ReleaseError("发布失败，旧版本恢复检查也失败；请立即查看 arxiv-agent 日志。") from rollback_failure
        raise ReleaseError("发布失败，已恢复上一版程序；首次发布失败时服务保持停止。") from failure
    if previous is not None and previous != release:
        atomic_link(previous, root / "previous")


def apply_release(root: Path, archive: Path, expected_hash: str, commit: str, release_id: str, timeout: int) -> dict:
    validate_commit(commit)
    if not re.fullmatch(re.escape(commit) + r"-[0-9]+-[0-9]+", release_id):
        raise ReleaseError("版本目录必须绑定提交、Actions run ID 和 attempt。")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash) or sha256_file(archive) != expected_hash:
        raise ReleaseError("发布包 SHA-256 校验失败，未修改运行版本。")
    if not (root / "shared/.env.production").is_file():
        raise ReleaseError("缺少 shared/.env.production，请先完成服务器初始化。")
    if not (root / ".cicd-layout").is_file():
        raise ReleaseError("目录尚未初始化为 CI/CD 布局，不能覆盖旧部署。")
    current_release(root)
    releases = root / "releases"
    releases.mkdir(exist_ok=True)
    release = releases / release_id
    if release.exists():
        if current_release(root) == release:
            SystemdRuntime().wait_healthy(release, timeout)
            return {"status": "already_active", "commit": commit}
        raise ReleaseError("该发布目录已存在，请重跑 Actions 以生成新的 attempt。")
    extract_checked(archive, release)
    manifest = validate_release(release, commit)
    # SSH 的 umask 可能较严格；代码和前端目录可读，shared 中的配置和数据仍保持私有权限。
    release.chmod(0o755)
    for path in release.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
    environment = install_environment(root, release, manifest["dependency_hash"])
    attach_shared_data(root, release)
    (release / ".venv").symlink_to(environment, target_is_directory=True)
    # wheel 只用于安装，删掉本次包内的副本以控制小服务器磁盘占用；运行和回退使用独立 venv。
    shutil.rmtree(release / ".release/wheels")
    activate_release(root, release, SystemdRuntime(), timeout)
    result = {"status": "deployed", "commit": commit, "release": release_id,
              "dependency_hash": manifest["dependency_hash"]}
    (root / "last-deployment.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--health-timeout", type=int, default=180)
    args = parser.parse_args()
    verify_runtime()
    if os.geteuid() == 0:
        raise ReleaseError("请使用普通部署用户；仅重启服务通过受限 sudo 执行。")
    root = args.root.resolve(strict=True)
    if root == Path("/") or not 10 <= args.health_timeout <= 900:
        raise ReleaseError("部署目录或检查超时不合法。")
    archive = args.archive.resolve(strict=True)
    if not archive.is_relative_to(root / "incoming"):
        raise ReleaseError("只接收 incoming 目录内的发布包。")
    import fcntl
    # Actions 已串行化；服务器侧锁继续保护手工触发及其他 SSH 会话。
    with (root / ".deploy.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ReleaseError("已有部署正在执行，请等待完成。") from exc
        print(json.dumps(apply_release(root, archive, args.sha256, args.commit, args.release_id, args.health_timeout)))
        archive.unlink()


if __name__ == "__main__":
    main()
