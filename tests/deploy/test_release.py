"""发布测试只操作临时目录；systemd、pip 和 HTTP 均用替身，不接触开发或生产数据。"""

from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, call, patch
import urllib.error
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "deploy"))
import apply_release as deploy
import package_release as package
from release_common import (
    EXPECTED_RUNTIME, ReleaseError, extract_checked, is_runtime_path,
    sha256_file, validate_release, verify_runtime,
)


COMMIT = "a" * 40
RELEASE_ID = f"{COMMIT}-123-1"


class TemporaryReleaseTest(unittest.TestCase):
    def setUp(self) -> None:
        # 使用仓库内临时目录，避开 Windows 用户临时目录权限和真实数据路径。
        temporary_root = REPO_ROOT / "temp" / "deployment-tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()

    def make_payload(self, **manifest_overrides: object) -> Path:
        payload = self.directory / "payload"
        for name, content in {
            "backend/main.py": "app = None\n",
            "new_frontend/dist/index.html": "<html>release</html>",
            ".release/requirements.lock": "example==1.0 --hash=sha256:" + "0" * 64,
            ".release/wheels/example.whl": "wheel placeholder",
        }.items():
            path = payload / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        manifest = {
            "schema_version": 1, "commit": COMMIT, "runtime": EXPECTED_RUNTIME,
            "dependency_hash": sha256_file(payload / ".release/requirements.lock"), "wheel_count": 1,
        }
        manifest.update(manifest_overrides)
        (payload / "release.json").write_text(json.dumps(manifest), encoding="utf-8")
        return payload

    def make_archive(self, payload: Path, destination: Path | None = None) -> Path:
        archive = destination or self.directory / "release.tar.gz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w:gz") as stream:
            for path in sorted(payload.rglob("*")):
                stream.add(path, arcname=path.relative_to(payload).as_posix(), recursive=False)
        return archive

    def require_symlinks(self) -> None:
        probe = self.directory / "link-probe"
        target = self.directory / "link-target"
        target.mkdir()
        try:
            probe.symlink_to(target, target_is_directory=True)
        except OSError as error:
            # Windows 未启用开发者模式时可能禁止 symlink；CI 的 Debian 环境必须执行这些用例。
            if sys.platform == "win32" and getattr(error, "winerror", None) == 1314:
                self.skipTest(f"本机不允许符号链接：{error}")
            raise
        probe.unlink()


class ArchiveTests(TemporaryReleaseTest):
    def test_runtime_rejects_incompatible_python_and_architecture(self) -> None:
        for major, minor, machine, accepted in [(3, 11, "x86_64", True), (3, 12, "x86_64", False), (3, 11, "aarch64", False)]:
            with self.subTest(python=(major, minor), machine=machine):
                version = Mock(major=major, minor=minor)
                with patch("release_common.platform.system", return_value="Linux"), patch("release_common.platform.machine", return_value=machine), patch("release_common.sys.version_info", version), patch.object(Path, "is_file", return_value=True), patch.object(Path, "read_text", return_value='ID=debian\nVERSION_ID="12"\n'):
                    if accepted:
                        verify_runtime()
                    else:
                        with self.assertRaises(ReleaseError):
                            verify_runtime()

    def test_extract_release_and_verify_manifest(self) -> None:
        payload = self.make_payload()
        destination = self.directory / "extracted"
        extract_checked(self.make_archive(payload), destination)
        self.assertEqual(validate_release(destination, COMMIT)["commit"], COMMIT)
        self.assertEqual((destination / "new_frontend/dist/index.html").read_text(), "<html>release</html>")

    def test_unsafe_members_rejected_before_any_extraction(self) -> None:
        cases = [
            ("../outside.txt", tarfile.REGTYPE), ("/absolute.txt", tarfile.REGTYPE),
            ("C:/outside.txt", tarfile.REGTYPE), ("a\\outside.txt", tarfile.REGTYPE),
            ("a/./alias.txt", tarfile.REGTYPE), ("a//alias.txt", tarfile.REGTYPE),
            ("link", tarfile.SYMTYPE), ("hardlink", tarfile.LNKTYPE),
            ("device", tarfile.CHRTYPE), ("pipe", tarfile.FIFOTYPE),
        ]
        for index, (name, member_type) in enumerate(cases):
            with self.subTest(name=name):
                archive = self.directory / f"unsafe-{index}.tar"
                with tarfile.open(archive, "w") as stream:
                    safe = tarfile.TarInfo("safe.txt")
                    safe.size = 4
                    stream.addfile(safe, io.BytesIO(b"safe"))
                    unsafe = tarfile.TarInfo(name)
                    unsafe.type = member_type
                    unsafe.linkname = "../outside.txt" if member_type in {tarfile.SYMTYPE, tarfile.LNKTYPE} else ""
                    stream.addfile(unsafe)
                destination = self.directory / f"destination-{index}"
                with self.assertRaises(ReleaseError):
                    extract_checked(archive, destination)
                self.assertFalse(destination.exists())

    def test_duplicate_path_rejected(self) -> None:
        archive = self.directory / "duplicate.tar"
        with tarfile.open(archive, "w") as stream:
            stream.addfile(tarfile.TarInfo("same.txt"))
            stream.addfile(tarfile.TarInfo("same.txt"))
        with self.assertRaisesRegex(ReleaseError, "重复"):
            extract_checked(archive, self.directory / "extracted")

    def test_runtime_data_filtered_from_source_but_rejected_in_release(self) -> None:
        names = [
            "backend/06-database/recommendation.db", "backend/data/auth/auth.sqlite3-wal",
            "backend/config/production.jwt-secret", "backend/config/api_keys.json",
            ".env.production", ".venv/bin/python", "logs/audit.log", "temp/report.json",
            "backend/01-loaded-docs/paper.json", "stray.sqlite3-shm",
        ]
        archive = self.directory / "source.tar"
        with tarfile.open(archive, "w") as stream:
            for name in [*names, ".env.example", "backend/main.py"]:
                stream.addfile(tarfile.TarInfo(name))
        with self.assertRaises(ReleaseError):
            extract_checked(archive, self.directory / "rejected")
        extracted = self.directory / "filtered"
        extract_checked(archive, extracted, source_archive=True)
        self.assertTrue((extracted / ".env.example").is_file())
        self.assertTrue((extracted / "backend/main.py").is_file())
        for name in names:
            with self.subTest(name=name):
                self.assertTrue(is_runtime_path(name))
                self.assertFalse((extracted / name).exists())

    def test_dependency_lock_tampering_rejected(self) -> None:
        payload = self.make_payload()
        (payload / ".release/requirements.lock").write_text("tampered")
        with self.assertRaisesRegex(ReleaseError, "锁文件"):
            validate_release(payload, COMMIT)

    def test_missing_frontend_rejected(self) -> None:
        payload = self.make_payload()
        (payload / "new_frontend/dist/index.html").unlink()
        with self.assertRaisesRegex(ReleaseError, "必要文件"):
            validate_release(payload, COMMIT)


class DeploymentTests(TemporaryReleaseTest):
    def setUp(self) -> None:
        super().setUp()
        self.require_symlinks()
        self.root = self.directory / "server"
        self.old = self.root / "releases/old"
        self.old.mkdir(parents=True)
        (self.root / "shared").mkdir()
        (self.root / "shared/.env.production").write_text("private configuration")
        (self.root / ".cicd-layout").write_text("1\n")
        self.database = self.root / "shared/backend/06-database/recommendation.db"
        self.database.parent.mkdir(parents=True)
        self.database.write_bytes(b"existing user history")
        deploy.atomic_link(self.old, self.root / "current")
        self.runtime = Mock(spec=deploy.SystemdRuntime)

    def request(self, archive: Path, expected_hash: str | None = None) -> dict:
        return deploy.apply_release(self.root, archive, expected_hash or sha256_file(archive), COMMIT, RELEASE_ID, 30)

    def test_hash_failure_preserves_running_version(self) -> None:
        archive = self.make_archive(self.make_payload())
        with patch.object(deploy, "install_environment") as install:
            with self.assertRaisesRegex(ReleaseError, "SHA-256"):
                self.request(archive, "0" * 64)
            install.assert_not_called()
        self.assertEqual(deploy.current_release(self.root), self.old)
        self.assertFalse((self.root / "releases" / RELEASE_ID).exists())

    def test_mismatched_manifest_never_switches_current(self) -> None:
        for field, value in [("commit", "b" * 40), ("runtime", {"python": "3.12"}), ("schema_version", 999)]:
            with self.subTest(field=field):
                payload = self.make_payload(**{field: value})
                archive = self.make_archive(payload)
                # 不同 attempt 独立落盘；模拟每次 CI 的新运行，不能复用已失败的解压目录。
                release_id = f"{COMMIT}-123-{len(list((self.root / 'releases').iterdir()))}"
                with patch.object(deploy, "install_environment") as install:
                    with self.assertRaises(ReleaseError):
                        deploy.apply_release(self.root, archive, sha256_file(archive), COMMIT, release_id, 30)
                    install.assert_not_called()
                self.assertEqual(deploy.current_release(self.root), self.old)

    def test_success_switches_code_and_reuses_user_data(self) -> None:
        archive = self.make_archive(self.make_payload(), self.root / "incoming/release.tar.gz")
        environment = self.root / "venvs/ready"
        environment.mkdir(parents=True)
        with patch.object(deploy, "install_environment", return_value=environment), patch.object(deploy, "SystemdRuntime", return_value=self.runtime):
            result = self.request(archive)
        active = self.root / "releases" / RELEASE_ID
        self.assertEqual(result["status"], "deployed")
        self.assertEqual(deploy.current_release(self.root), active)
        self.assertEqual((self.root / "previous").resolve(), self.old)
        self.assertEqual((active / ".venv").resolve(), environment)
        self.assertEqual((active / "backend/06-database/recommendation.db").read_bytes(), b"existing user history")
        self.assertEqual((self.root / "shared/.env.production").read_text(), "private configuration")
        self.assertFalse((active / ".release/wheels").exists())
        self.assertEqual(json.loads((self.root / "last-deployment.json").read_text())["commit"], COMMIT)
        self.runtime.wait_healthy.assert_called_once_with(active, 30)
        with patch.object(deploy, "install_environment") as install, patch.object(deploy, "SystemdRuntime", return_value=self.runtime):
            self.assertEqual(self.request(archive)["status"], "already_active")
            install.assert_not_called()

    def test_failed_install_does_not_switch_or_restart(self) -> None:
        archive = self.make_archive(self.make_payload())
        with patch.object(deploy, "install_environment", side_effect=subprocess.CalledProcessError(1, "pip")), patch.object(deploy, "SystemdRuntime") as runtime:
            with self.assertRaises(subprocess.CalledProcessError):
                self.request(archive)
            runtime.assert_not_called()
        self.assertEqual(deploy.current_release(self.root), self.old)

    def test_startup_failure_restores_previous_version_and_keeps_new_data(self) -> None:
        candidate = self.root / "releases/new"
        candidate.mkdir()
        def probe(release: Path, timeout: int) -> None:
            if release == candidate:
                self.database.write_bytes(b"data written during release")
                raise ReleaseError("unhealthy")
        self.runtime.wait_healthy.side_effect = probe
        with self.assertRaisesRegex(ReleaseError, "已恢复上一版"):
            deploy.activate_release(self.root, candidate, self.runtime, 30)
        self.assertEqual(deploy.current_release(self.root), self.old)
        self.assertEqual(self.database.read_bytes(), b"data written during release")
        self.assertEqual(self.runtime.restart.call_count, 2)
        self.runtime.wait_healthy.assert_has_calls([call(candidate, 30), call(self.old, 30)])

    def test_first_failed_deployment_stops_service(self) -> None:
        (self.root / "current").unlink()
        candidate = self.root / "releases/new"
        candidate.mkdir()
        self.runtime.wait_healthy.side_effect = ReleaseError("unhealthy")
        with self.assertRaises(ReleaseError):
            deploy.activate_release(self.root, candidate, self.runtime, 30)
        self.assertIsNone(deploy.current_release(self.root))
        self.runtime.stop.assert_called_once()

    def test_failed_rollback_is_reported(self) -> None:
        self.runtime.restart.side_effect = RuntimeError("systemd failure")
        with self.assertRaisesRegex(ReleaseError, "恢复检查也失败"):
            deploy.activate_release(self.root, self.root / "releases/new", self.runtime, 30)
        self.assertEqual(deploy.current_release(self.root), self.old)

    def test_non_link_and_external_current_rejected(self) -> None:
        (self.root / "current").unlink()
        (self.root / "current").mkdir()
        with self.assertRaisesRegex(ReleaseError, "版本链接"):
            deploy.current_release(self.root)
        (self.root / "current").rmdir()
        deploy.atomic_link(self.directory, self.root / "current")
        with self.assertRaisesRegex(ReleaseError, "之外"):
            deploy.current_release(self.root)


class EnvironmentTests(TemporaryReleaseTest):
    def setUp(self) -> None:
        super().setUp()
        self.require_symlinks()
        self.release = self.make_payload()
        self.dependency_hash = sha256_file(self.release / ".release/requirements.lock")
        self.root = self.directory / "server"
        self.root.mkdir()

    @staticmethod
    def fake_install(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        if "venv" in command:
            binaries = Path(command[-1]) / "bin"
            binaries.mkdir(parents=True)
            (binaries / "python").touch()
            (binaries / "gunicorn").touch()
        return subprocess.CompletedProcess(command, 0)

    def test_completed_environment_reused_without_moving_it(self) -> None:
        with patch.object(deploy.subprocess, "run", side_effect=self.fake_install) as run:
            first = deploy.install_environment(self.root, self.release, self.dependency_hash)
            self.assertEqual(run.call_count, 3)
            install_args = run.call_args_list[1].args[0]
            self.assertIn("--no-index", install_args)
            self.assertIn("--require-hashes", install_args)
            self.assertEqual(install_args[0], str(first / "bin/python"))
            run.reset_mock()
            second = deploy.install_environment(self.root, self.release, self.dependency_hash)
            self.assertEqual(first, second)
            run.assert_not_called()

    def test_incomplete_install_not_cached(self) -> None:
        def fail_pip(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            if "install" in command:
                raise subprocess.CalledProcessError(1, command)
            return self.fake_install(command, **kwargs)
        with patch.object(deploy.subprocess, "run", side_effect=fail_pip):
            with self.assertRaises(subprocess.CalledProcessError):
                deploy.install_environment(self.root, self.release, self.dependency_hash)
        partial = next((self.root / "venvs").iterdir())
        self.assertFalse((partial / ".release-ready").exists())
        with patch.object(deploy.subprocess, "run", side_effect=self.fake_install):
            recovered = deploy.install_environment(self.root, self.release, self.dependency_hash)
        self.assertNotEqual(partial, recovered)
        self.assertTrue((recovered / ".release-ready").is_file())

    def test_broken_cache_is_rebuilt(self) -> None:
        environments = self.root / "venvs"
        environments.mkdir()
        deploy.atomic_link(environments / "missing", environments / f"cache-{self.dependency_hash}")
        with patch.object(deploy.subprocess, "run", side_effect=self.fake_install):
            recovered = deploy.install_environment(self.root, self.release, self.dependency_hash)
        self.assertEqual((environments / f"cache-{self.dependency_hash}").resolve(), recovered)


class WheelLockTests(TemporaryReleaseTest):
    def wheel(self, name: str, version: str = "1.0", filename: str | None = None) -> Path:
        wheels = self.directory / "wheels"
        wheels.mkdir(exist_ok=True)
        path = wheels / (filename or f"{name}-{version}-py3-none-any.whl")
        with zipfile.ZipFile(path, "w") as stream:
            stream.writestr(f"{name}-{version}.dist-info/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        return path

    def test_lock_contains_actual_wheel_hashes_and_normalized_names(self) -> None:
        paths = [self.wheel("gunicorn"), self.wheel("milvus_lite"), self.wheel("torch", "2.9.0+cpu")]
        lock = self.directory / "requirements.lock"
        self.assertEqual(package.lock_wheels(self.directory / "wheels", lock), 3)
        contents = lock.read_text()
        for path in paths:
            self.assertIn(sha256_file(path), contents)
        self.assertIn("milvus-lite==1.0", contents)
        self.assertIn("torch==2.9.0+cpu", contents)

    def test_missing_runtime_dependency_rejected(self) -> None:
        self.wheel("gunicorn")
        with self.assertRaisesRegex(ReleaseError, "缺少生产运行依赖"):
            package.lock_wheels(self.directory / "wheels", self.directory / "lock")

    def test_duplicate_package_versions_rejected(self) -> None:
        self.wheel("gunicorn", "1.0")
        self.wheel("gunicorn", "2.0")
        with self.assertRaisesRegex(ReleaseError, "多个 wheel"):
            package.lock_wheels(self.directory / "wheels", self.directory / "lock")

    def test_cuda_torch_rejected(self) -> None:
        self.wheel("torch", "2.9.0")
        with self.assertRaisesRegex(ReleaseError, "CPU Torch"):
            package.lock_wheels(self.directory / "wheels", self.directory / "lock")


class ActivationContractTests(TemporaryReleaseTest):
    """跨平台验证失败状态流转；真实符号链接和数据目录仍由 DeploymentTests 单独验证。"""

    def setUp(self) -> None:
        super().setUp()
        self.root = self.directory
        self.old = self.root / "releases/old"
        self.candidate = self.root / "releases/new"
        self.runtime = Mock(spec=deploy.SystemdRuntime)
        self.history = self.root / "history.db"
        self.history.write_bytes(b"user history")

        def fake_link(target: Path, link: Path) -> None:
            # 仅替换需要操作系统特权的链接边界；其余发布逻辑执行真实实现。
            link.write_text(str(target), encoding="utf-8")

        def fake_current(root: Path) -> Path | None:
            current = root / "current"
            return Path(current.read_text(encoding="utf-8")) if current.exists() else None

        self.current = fake_current
        for replacement in [patch.object(deploy, "atomic_link", side_effect=fake_link), patch.object(deploy, "current_release", side_effect=fake_current)]:
            replacement.start()
            self.addCleanup(replacement.stop)
        fake_link(self.old, self.root / "current")

    def test_success_records_previous_after_health_check(self) -> None:
        self.runtime.wait_healthy.side_effect = lambda *_: self.assertEqual(self.current(self.root), self.candidate)
        deploy.activate_release(self.root, self.candidate, self.runtime, 30)
        self.assertEqual(self.current(self.root), self.candidate)
        self.assertEqual((self.root / "previous").read_text(encoding="utf-8"), str(self.old))
        self.runtime.restart.assert_called_once()

    def test_unhealthy_release_rolls_back_without_reverting_user_writes(self) -> None:
        def health(release: Path, timeout: int) -> None:
            if release == self.candidate:
                self.history.write_bytes(b"new user history")
                raise ReleaseError("startup failed")
        self.runtime.wait_healthy.side_effect = health
        with self.assertRaisesRegex(ReleaseError, "已恢复上一版"):
            deploy.activate_release(self.root, self.candidate, self.runtime, 30)
        self.assertEqual(self.current(self.root), self.old)
        self.assertEqual(self.history.read_bytes(), b"new user history")
        self.assertEqual(self.runtime.restart.call_count, 2)

    def test_failed_first_release_stops_service_and_removes_current(self) -> None:
        (self.root / "current").unlink()
        self.runtime.wait_healthy.side_effect = ReleaseError("startup failed")
        with self.assertRaises(ReleaseError):
            deploy.activate_release(self.root, self.candidate, self.runtime, 30)
        self.assertIsNone(self.current(self.root))
        self.runtime.stop.assert_called_once()

    def test_failed_service_restart_attempts_rollback(self) -> None:
        self.runtime.restart.side_effect = [subprocess.CalledProcessError(1, "systemctl"), None]
        with self.assertRaisesRegex(ReleaseError, "已恢复上一版"):
            deploy.activate_release(self.root, self.candidate, self.runtime, 30)
        self.assertEqual(self.current(self.root), self.old)
        self.runtime.wait_healthy.assert_called_once_with(self.old, 30)

    def test_double_failure_is_not_reported_as_successful_rollback(self) -> None:
        self.runtime.wait_healthy.side_effect = ReleaseError("startup failed")
        with self.assertRaisesRegex(ReleaseError, "恢复检查也失败"):
            deploy.activate_release(self.root, self.candidate, self.runtime, 30)
        self.assertEqual(self.current(self.root), self.old)


class HealthTests(unittest.TestCase):
    def test_requires_consecutive_successes(self) -> None:
        runtime = deploy.SystemdRuntime()
        with patch.object(runtime, "healthy", side_effect=[True, False, True, True, True]) as healthy, patch.object(deploy.time, "sleep"):
            runtime.wait_healthy(Path("release"), 10)
        self.assertEqual(healthy.call_count, 5)

    def test_wrong_process_directory_rejected_before_http(self) -> None:
        with patch.object(deploy.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), patch.object(deploy.subprocess, "check_output", return_value="123"), patch.object(Path, "resolve", return_value=Path("old/backend")), patch.object(deploy.urllib.request, "build_opener") as opener:
            self.assertFalse(deploy.SystemdRuntime().healthy(Path("new")))
            opener.assert_not_called()

    def test_authentication_boundary_must_return_401(self) -> None:
        release = Path("release")
        for status in [401, 200, 403, 500]:
            with self.subTest(status=status):
                health_response = Mock()
                health_response.status = 200
                health_context = Mock()
                health_context.__enter__ = Mock(return_value=health_response)
                health_context.__exit__ = Mock(return_value=False)
                auth_result = Mock() if status == 200 else urllib.error.HTTPError("http://localhost", status, "response", {}, None)
                opener = Mock()
                opener.open.side_effect = [health_context, auth_result]
                with patch.object(deploy.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), patch.object(deploy.subprocess, "check_output", return_value="123"), patch.object(Path, "resolve", return_value=release / "backend"), patch.object(deploy.urllib.request, "build_opener", return_value=opener):
                    self.assertEqual(deploy.SystemdRuntime().healthy(release), status == 401)


if __name__ == "__main__":
    unittest.main()
