import multiprocessing
import os
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path

from services.document import isolated_subprocess
from services.document.isolated_subprocess import IsolatedSubprocessError, run_in_isolated_subprocess

IS_LINUX = sys.platform.startswith("linux")

try:
    import fcntl
except ImportError:
    fcntl = None


# 子进程目标函数必须是模块级函数，spawn 出的子进程才能按名字导入。
def _echo(value):
    return value


def _raise_value_error():
    raise ValueError("boom from child")


def _sleep_forever():
    time.sleep(600)


def _exit_with_code():
    os._exit(3)


def _kill_self():
    os.kill(os.getpid(), signal.SIGKILL)


def _large_payload(size):
    return b"x" * size


def _read_oom_score_adj():
    return Path("/proc/self/oom_score_adj").read_text(encoding="ascii").strip()


def _write_pid_then_sleep(pid_file):
    Path(pid_file).write_text(str(os.getpid()), encoding="ascii")
    time.sleep(600)


def _run_sleeper_through_runner(pid_file):
    run_in_isolated_subprocess(
        _write_pid_then_sleep,
        pid_file,
        timeout_seconds=600,
        lock_path=None,
        lock_timeout_seconds=1,
    )


def _process_gone(pid):
    """进程已退出或只剩僵尸（容器里可能没有 init 回收）都视为已结束。"""
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
    except FileNotFoundError:
        return True
    return "State:\tZ" in status


class IsolatedSubprocessTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.lock_path = os.path.join(self.temp_dir.name, "parse.lock")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _run(self, target, *args, timeout_seconds=30.0, lock_timeout_seconds=5.0):
        return run_in_isolated_subprocess(
            target,
            *args,
            timeout_seconds=timeout_seconds,
            lock_path=self.lock_path,
            lock_timeout_seconds=lock_timeout_seconds,
            task_label="test parse",
        )

    def test_returns_child_result(self):
        self.assertEqual(self._run(_echo, {"pages": [1, 2]}), {"pages": [1, 2]})

    def test_large_result_does_not_deadlock_on_pipe(self):
        size = 20 * 1024 * 1024
        self.assertEqual(len(self._run(_large_payload, size)), size)

    def test_child_exception_is_reported_with_traceback(self):
        with self.assertRaises(IsolatedSubprocessError) as caught:
            self._run(_raise_value_error)
        self.assertEqual(caught.exception.reason, "child_error")
        self.assertIn("boom from child", str(caught.exception))

    def test_timeout_kills_child(self):
        started = time.monotonic()
        with self.assertRaises(IsolatedSubprocessError) as caught:
            self._run(_sleep_forever, timeout_seconds=1.0)
        self.assertEqual(caught.exception.reason, "timeout")
        self.assertLess(time.monotonic() - started, 30)
        # 超时后锁已释放，下一次解析可以立即开始。
        self.assertEqual(self._run(_echo, 1, lock_timeout_seconds=0.1), 1)

    def test_child_exit_without_result_is_crashed(self):
        with self.assertRaises(IsolatedSubprocessError) as caught:
            self._run(_exit_with_code)
        self.assertEqual(caught.exception.reason, "crashed")
        self.assertIn("3", str(caught.exception))

    @unittest.skipIf(sys.platform == "win32", "Windows 没有 SIGKILL")
    def test_child_killed_by_sigkill_is_reported_as_killed(self):
        with self.assertRaises(IsolatedSubprocessError) as caught:
            self._run(_kill_self)
        self.assertEqual(caught.exception.reason, "killed")
        self.assertIn("OOM", str(caught.exception))

    def test_busy_when_another_parse_holds_process_lock(self):
        acquired = isolated_subprocess._process_lock.acquire(timeout=5)
        self.assertTrue(acquired)
        try:
            with self.assertRaises(IsolatedSubprocessError) as caught:
                self._run(_echo, 1, lock_timeout_seconds=0.2)
        finally:
            isolated_subprocess._process_lock.release()
        self.assertEqual(caught.exception.reason, "busy")

    @unittest.skipIf(fcntl is None, "flock 仅在 POSIX 平台可用")
    def test_busy_when_another_process_holds_file_lock(self):
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            with self.assertRaises(IsolatedSubprocessError) as caught:
                self._run(_echo, 1, lock_timeout_seconds=0.5)
        finally:
            os.close(fd)
        self.assertEqual(caught.exception.reason, "busy")
        self.assertEqual(self._run(_echo, 2, lock_timeout_seconds=0.5), 2)

    @unittest.skipUnless(IS_LINUX, "oom_score_adj 仅 Linux 支持")
    def test_child_prefers_itself_as_oom_victim(self):
        self.assertEqual(self._run(_read_oom_score_adj), "1000")

    @unittest.skipUnless(IS_LINUX, "PR_SET_PDEATHSIG 仅 Linux 支持")
    def test_child_exits_when_parent_is_killed(self):
        pid_file = os.path.join(self.temp_dir.name, "child.pid")
        parent = multiprocessing.get_context("spawn").Process(target=_run_sleeper_through_runner, args=(pid_file,))
        parent.start()
        try:
            deadline = time.monotonic() + 60
            while not os.path.exists(pid_file) or not Path(pid_file).read_text(encoding="ascii"):
                self.assertLess(time.monotonic(), deadline, "解析子进程没有启动")
                time.sleep(0.1)
            child_pid = int(Path(pid_file).read_text(encoding="ascii"))

            os.kill(parent.pid, signal.SIGKILL)
            parent.join(10)

            deadline = time.monotonic() + 10
            while not _process_gone(child_pid):
                self.assertLess(time.monotonic(), deadline, "父进程被杀后解析子进程仍在运行")
                time.sleep(0.1)
        finally:
            if parent.is_alive():
                parent.kill()
                parent.join()
            # 断言失败时清理残留的解析子进程，避免测试泄漏常驻进程。
            if os.path.exists(pid_file) and Path(pid_file).read_text(encoding="ascii"):
                try:
                    os.kill(int(Path(pid_file).read_text(encoding="ascii")), signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    unittest.main()
