"""在通过质量检查的 Debian CI 环境中打包代码、前端和带哈希的 CPU 依赖。"""

# 运行位置：GitHub Actions 的质量门工作流（.github/workflows/quality-gate.yml），
# 只在 package_release 输入为 true（main 分支发布）且全部检查通过后执行。
#
# 产物（默认写入 temp/ci-release/）：
#   release.tar.gz   发布包，解压后的目录结构如下：
#     backend/ ...                 git archive 导出的源码（已剔除运行时数据与敏感配置）
#     frontend/dist/               CI 中构建好的前端静态文件
#     .release/wheels/*.whl        全部运行时依赖的离线 wheel
#     .release/requirements.lock   "包名==版本 --hash=sha256:..." 格式的锁文件
#     release.json                 清单：提交 SHA、运行环境、锁文件哈希、wheel 数量
#   release.sha256   发布包整体的 SHA-256，服务器安装前据此校验传输完整性。
#
# 之后由 publish_ssh.sh 把产物传到服务器，再由 apply_release.py 安装。

from __future__ import annotations

import argparse
import email
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

from release_common import EXPECTED_RUNTIME, ReleaseError, extract_checked, sha256_file, validate_commit, verify_runtime


# 项目根目录（本文件位于 <root>/deploy/ 下）。
ROOT = Path(__file__).resolve().parents[1]


def lock_wheels(wheels: Path, lock_path: Path) -> int:
    """读取 wheels 目录中每个 wheel 的元数据，生成带 SHA-256 的锁文件，返回依赖数量。

    服务器端用 `pip install --require-hashes` 安装，任何 wheel 被替换或损坏都会安装失败。
    """
    entries = {}
    for wheel in sorted(wheels.glob("*.whl")):
        # wheel 本质是 zip 包，包名和版本记录在 <name>-<ver>.dist-info/METADATA 中。
        with zipfile.ZipFile(wheel) as archive:
            # wheel 可能把 vendored 依赖的 dist-info 放在子目录中（例如 setuptools/_vendor）。
            # 这里只认归档根目录下的主包元数据，避免把内部依赖误当成多个主包元数据。
            metadata_files = [
                name for name in archive.namelist()
                if name.count("/") == 1 and name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_files) != 1:
                raise ReleaseError(f"wheel 元数据不完整：{wheel.name}")
            # METADATA 采用 RFC 822 邮件头格式，可以直接用 email 模块解析。
            metadata = email.message_from_bytes(archive.read(metadata_files[0]))
        raw_name, version = metadata["Name"], metadata["Version"]
        if not raw_name or not version or not re.fullmatch(r"[A-Za-z0-9_.+-]+", version):
            raise ReleaseError(f"wheel 名称或版本不合法：{wheel.name}")
        # 按 PEP 503 规范化包名：连续的 - _ . 统一为 -，再转小写（如 Foo_Bar -> foo-bar）。
        name = re.sub(r"[-_.]+", "-", raw_name).lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
            raise ReleaseError(f"wheel 名称不合法：{wheel.name}")
        # 两阶段 pip 安装可能因上游依赖升级换回 CUDA 包；在传包前拒绝这种资源配置漂移。
        if name.startswith("nvidia-") or (name in {"torch", "torchvision", "torchaudio"} and not version.endswith("+cpu")):
            raise ReleaseError("发布包只允许 CPU Torch，不接受 CUDA 运行库。")
        if name in entries:
            raise ReleaseError(f"同一依赖出现多个 wheel：{name}")
        entries[name] = f"{name}=={version} --hash=sha256:{sha256_file(wheel)}"
    # 生产启动必需的关键依赖：gunicorn 提供 Web 进程，milvus-lite 提供向量库，torch 运行本地模型。
    if not {"gunicorn", "milvus-lite", "torch"}.issubset(entries):
        raise ReleaseError("CI 环境缺少生产运行依赖。")
    # 包含完整的传递依赖及文件哈希，服务器不重新解析可变版本，也不联网编译依赖。
    lock_path.write_text("\n".join(entries[name] for name in sorted(entries)) + "\n", encoding="utf-8")
    return len(entries)


def package_release(output: Path, commit: str, runtime_freeze: Path) -> dict:
    """为指定提交生成 release.tar.gz 与 release.sha256，返回提交、wheel 数量和产物路径。

    runtime_freeze 是 CI 在安装测试依赖之前执行 `pip freeze --all` 得到的运行时依赖快照。
    """
    # 前置检查：环境一致、提交合法、工作区正是本次通过 CI 的提交、前端已构建。
    verify_runtime()
    validate_commit(commit)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != commit:
        raise ReleaseError("工作区不是本次通过 CI 的提交。")
    if not (ROOT / "frontend/dist/index.html").is_file():
        raise ReleaseError("请先通过包含前端构建的统一质量门禁。")
    output.mkdir(parents=True, exist_ok=True)
    # 所有中间文件放在 output 下的临时目录中，函数结束（含异常）时自动清理。
    with tempfile.TemporaryDirectory(prefix="release-build-", dir=output) as temporary:
        staging = Path(temporary)
        # 1. 用 git archive 导出该提交的已跟踪文件：未提交的本地改动和被忽略的文件都不会混进来。
        source_tar = staging / "source.tar"
        subprocess.run(["git", "archive", "--format=tar", "-o", str(source_tar), commit], cwd=ROOT, check=True)
        # 2. 安全解压到 payload/，source_archive=True 会顺带剔除运行时目录和敏感配置。
        payload = staging / "payload"
        extract_checked(source_tar, payload, source_archive=True)
        # 3. 前端构建产物不在 Git 中，单独从 CI 工作区复制进来。
        shutil.copytree(ROOT / "frontend/dist", payload / "frontend/dist", symlinks=True)
        # 4. .release/ 存放仅供安装使用的元数据：离线 wheel 和锁文件。
        metadata_dir = payload / ".release"
        metadata_dir.mkdir(exist_ok=False)
        # 只打包安装测试依赖之前的运行时快照，pytest/fakeredis 等不进入生产环境。
        frozen = staging / "installed.txt"
        shutil.copyfile(runtime_freeze, frozen)
        wheels = metadata_dir / "wheels"
        wheels.mkdir()
        # 原生扩展在与生产一致的 Debian 12 中构建；CPU Torch 的本地版本号由 freeze 原样保留。
        # --no-deps：freeze 已列出全部传递依赖，不再让 pip 重新解析，避免拉入快照之外的版本。
        subprocess.run([
            sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(wheels),
            "--extra-index-url", "https://download.pytorch.org/whl/cpu", "-r", str(frozen),
        ], check=True)
        # 5. 生成带哈希的锁文件和 release.json 清单。
        lock = metadata_dir / "requirements.lock"
        wheel_count = lock_wheels(wheels, lock)
        manifest = {"schema_version": 1, "commit": commit, "runtime": EXPECTED_RUNTIME,
                    "dependency_hash": sha256_file(lock), "wheel_count": wheel_count}
        (payload / "release.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # 6. 打成 tar.gz。wheel 本身已是压缩格式，compresslevel=1 足够且明显更快；
        #    按排序后的路径逐个添加（recursive=False），保证归档顺序稳定。
        archive_path = output / "release.tar.gz"
        with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
            for path in sorted(payload.rglob("*")):
                # 服务器端 checked_members 会拒绝链接，这里提前失败，问题在 CI 中就能暴露。
                if path.is_symlink():
                    raise ReleaseError("构建产物不能包含符号链接。")
                archive.add(path, arcname=path.relative_to(payload).as_posix(), recursive=False)
        # 7. 写出整包哈希，发布脚本会把它作为参数传给服务器端校验。
        (output / "release.sha256").write_text(sha256_file(archive_path) + "\n", encoding="ascii")
    return {"commit": commit, "wheel_count": wheel_count, "archive": str(archive_path)}


def main() -> None:
    """命令行入口，例如：
    python deploy/package_release.py --commit <40位SHA> --runtime-freeze /tmp/runtime-freeze.txt
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "temp/ci-release")
    parser.add_argument("--runtime-freeze", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package_release(args.output.resolve(), args.commit, args.runtime_freeze.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
