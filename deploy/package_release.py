"""在通过质量检查的 Debian CI 环境中打包代码、前端和带哈希的 CPU 依赖。"""

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


ROOT = Path(__file__).resolve().parents[1]


def lock_wheels(wheels: Path, lock_path: Path) -> int:
    entries = {}
    for wheel in sorted(wheels.glob("*.whl")):
        with zipfile.ZipFile(wheel) as archive:
            # wheel 可能把 vendored 依赖的 dist-info 放在子目录中（例如 setuptools/_vendor）。
            # 这里只认归档根目录下的主包元数据，避免把内部依赖误当成多个主包元数据。
            metadata_files = [
                name for name in archive.namelist()
                if name.count("/") == 1 and name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_files) != 1:
                raise ReleaseError(f"wheel 元数据不完整：{wheel.name}")
            metadata = email.message_from_bytes(archive.read(metadata_files[0]))
        raw_name, version = metadata["Name"], metadata["Version"]
        if not raw_name or not version or not re.fullmatch(r"[A-Za-z0-9_.+-]+", version):
            raise ReleaseError(f"wheel 名称或版本不合法：{wheel.name}")
        name = re.sub(r"[-_.]+", "-", raw_name).lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
            raise ReleaseError(f"wheel 名称不合法：{wheel.name}")
        # 两阶段 pip 安装可能因上游依赖升级换回 CUDA 包；在传包前拒绝这种资源配置漂移。
        if name.startswith("nvidia-") or (name in {"torch", "torchvision", "torchaudio"} and not version.endswith("+cpu")):
            raise ReleaseError("发布包只允许 CPU Torch，不接受 CUDA 运行库。")
        if name in entries:
            raise ReleaseError(f"同一依赖出现多个 wheel：{name}")
        entries[name] = f"{name}=={version} --hash=sha256:{sha256_file(wheel)}"
    if not {"gunicorn", "milvus-lite", "torch"}.issubset(entries):
        raise ReleaseError("CI 环境缺少生产运行依赖。")
    # 包含完整的传递依赖及文件哈希，服务器不重新解析可变版本，也不联网编译依赖。
    lock_path.write_text("\n".join(entries[name] for name in sorted(entries)) + "\n", encoding="utf-8")
    return len(entries)


def package_release(output: Path, commit: str) -> dict:
    verify_runtime()
    validate_commit(commit)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != commit:
        raise ReleaseError("工作区不是本次通过 CI 的提交。")
    if not (ROOT / "new_frontend/dist/index.html").is_file():
        raise ReleaseError("请先通过包含前端构建的统一质量门禁。")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="release-build-", dir=output) as temporary:
        staging = Path(temporary)
        source_tar = staging / "source.tar"
        subprocess.run(["git", "archive", "--format=tar", "-o", str(source_tar), commit], cwd=ROOT, check=True)
        payload = staging / "payload"
        extract_checked(source_tar, payload, source_archive=True)
        shutil.copytree(ROOT / "new_frontend/dist", payload / "new_frontend/dist", symlinks=True)
        metadata_dir = payload / ".release"
        metadata_dir.mkdir(exist_ok=False)
        frozen = staging / "installed.txt"
        frozen.write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"], text=True), encoding="utf-8")
        wheels = metadata_dir / "wheels"
        wheels.mkdir()
        # 原生扩展在与生产一致的 Debian 12 中构建；CPU Torch 的本地版本号由 freeze 原样保留。
        subprocess.run([
            sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(wheels),
            "--extra-index-url", "https://download.pytorch.org/whl/cpu", "-r", str(frozen),
        ], check=True)
        lock = metadata_dir / "requirements.lock"
        wheel_count = lock_wheels(wheels, lock)
        manifest = {"schema_version": 1, "commit": commit, "runtime": EXPECTED_RUNTIME,
                    "dependency_hash": sha256_file(lock), "wheel_count": wheel_count}
        (payload / "release.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        archive_path = output / "release.tar.gz"
        with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
            for path in sorted(payload.rglob("*")):
                if path.is_symlink():
                    raise ReleaseError("构建产物不能包含符号链接。")
                archive.add(path, arcname=path.relative_to(payload).as_posix(), recursive=False)
        (output / "release.sha256").write_text(sha256_file(archive_path) + "\n", encoding="ascii")
    return {"commit": commit, "wheel_count": wheel_count, "archive": str(archive_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "temp/ci-release")
    args = parser.parse_args()
    print(json.dumps(package_release(args.output.resolve(), args.commit), ensure_ascii=False))


if __name__ == "__main__":
    main()
