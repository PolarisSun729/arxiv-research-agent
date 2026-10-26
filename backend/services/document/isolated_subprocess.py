"""在隔离子进程中执行重型解析任务（Docling 建索引、快速问答全文解析），保证子进程出问题时只影响当前任务。

设计目标：子进程内存不足、卡住或崩溃时，只让这一次调用失败，Web 进程不受拖累，也不留下残余进程。

1. 超时强杀：父进程最多等待 timeout_seconds；超时先 terminate，宽限期后 kill，最后一定 join 回收。
2. 生命周期绑定：Linux 下子进程通过 PR_SET_PDEATHSIG 绑定父进程，gunicorn worker 被杀时子进程随之退出，
   不会再变成占着内存的孤儿进程。
3. 并发名额：默认全局互斥，同一时间最多运行一个子进程。进程内用 threading.Lock，跨进程用 flock 文件锁；
   flock 在持有者进程退出时由内核自动释放，不会因 worker 被杀而残留死锁。
   调用方可传入自己的信号量（process_slots），与默认的全局互斥互不占用名额。
4. 优先被 OOM 选中：子进程把 oom_score_adj 调到最大，cgroup 内存超限时内核优先杀它而不是 Web worker。
"""

from __future__ import annotations

import ctypes
import logging
import multiprocessing
import os
import signal
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

try:
    import fcntl
except ImportError:  # Windows 本地开发没有 fcntl，退回只做进程内互斥。
    fcntl = None

logger = logging.getLogger(__name__)

# terminate 之后等待子进程自行退出的宽限期，超过后改用 kill。
_TERMINATE_GRACE_SECONDS = 5.0
# 子进程已回传结果后，等待它正常退出的时间。
_EXIT_AFTER_RESULT_SECONDS = 10.0
_LOCK_POLL_INTERVAL_SECONDS = 0.5
# 回传给父进程的子进程异常堆栈最多保留的字符数，避免超长堆栈写进任务失败原因。
_CHILD_TRACEBACK_LIMIT = 4000
_PR_SET_PDEATHSIG = 1

_process_lock = threading.Lock()


class IsolatedSubprocessError(RuntimeError):
    """隔离子进程执行失败；reason 区分失败类型，便于日志与任务失败原因定位。

    reason 取值：
        busy         等待互斥锁超时，已有其他解析在运行
        timeout      子进程运行超过 timeout_seconds，已被强制结束
        killed       子进程被信号杀死（SIGKILL 通常意味着内存超限被 OOM killer 选中）
        crashed      子进程未回传结果就以非零退出码退出
        child_error  目标函数在子进程内抛出异常
    """

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def run_in_isolated_subprocess(
    target: Callable[..., Any],
    *args: Any,
    timeout_seconds: float,
    lock_path: Optional[str],
    lock_timeout_seconds: float,
    task_label: str = "isolated subprocess",
    process_slots: Optional[threading.Semaphore] = None,
) -> Any:
    """在 spawn 出的子进程中执行 target(*args) 并返回其结果。

    target 和 args 必须可被 pickle（模块级函数）；返回值同样经 pickle 传回父进程。
    process_slots 为进程内并发名额，缺省时与 Docling 解析共用全局互斥锁。

    注意：PR_SET_PDEATHSIG 绑定的是创建子进程的线程而不是整个进程。调用方线程在子进程运行期间
    一直阻塞在本函数里，因此线程先于子进程退出的情况不会发生；不要改成"启动后立即返回"的用法。
    """
    slots = process_slots if process_slots is not None else _process_lock
    with _exclusive_slot(slots, lock_path, lock_timeout_seconds, task_label):
        return _run_child(target, args, timeout_seconds=timeout_seconds, task_label=task_label)


@contextmanager
def _exclusive_slot(
    slots: Any, lock_path: Optional[str], timeout_seconds: float, task_label: str
) -> Iterator[None]:
    """获取进程内名额与跨进程文件锁；两者共用同一个截止时间。"""
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    if not slots.acquire(timeout=max(0.0, timeout_seconds)):
        raise IsolatedSubprocessError("busy", f"{task_label}: 等待 {timeout_seconds:.0f} 秒仍有其他解析任务在运行")
    try:
        if fcntl is None or not lock_path:
            yield
            return
        os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise IsolatedSubprocessError(
                            "busy", f"{task_label}: 等待 {timeout_seconds:.0f} 秒仍有其他进程持有解析锁"
                        )
                    time.sleep(_LOCK_POLL_INTERVAL_SECONDS)
            yield
        finally:
            # 关闭文件描述符即释放 flock。
            os.close(fd)
    finally:
        slots.release()


def _run_child(target: Callable[..., Any], args: tuple, *, timeout_seconds: float, task_label: str) -> Any:
    """启动子进程并等待结果；无论成功、失败还是超时，返回前都确保子进程已被回收。"""
    # spawn 而不是 fork：父进程已有多个线程，fork 后子进程可能继承到被锁住的锁。
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    # 不设 daemon：daemon 进程不允许再创建子进程，解析库内部若用到多进程会直接报错。
    process = context.Process(
        target=_child_entry,
        args=(sender, os.getpid(), target, args),
        name=f"{task_label} worker",
    )
    started_at = time.monotonic()
    process.start()
    # 父进程关闭自己手里的发送端：子进程退出后 receiver 才能读到 EOF，而不是一直等待。
    sender.close()
    try:
        # 必须先 recv 再 join：结果较大时子进程会阻塞在写管道上，先 join 会互相等死。
        if not receiver.poll(max(0.0, timeout_seconds)):
            elapsed = time.monotonic() - started_at
            logger.error("%s timed out after %.0fs, killing pid=%s", task_label, elapsed, process.pid)
            _stop_process(process)
            raise IsolatedSubprocessError("timeout", f"{task_label}: 超过 {timeout_seconds:.0f} 秒未完成，已强制结束")
        try:
            status, payload = receiver.recv()
        except EOFError:
            # 子进程没有回传任何结果就退出了：被信号杀死或解释器异常退出。
            process.join(_TERMINATE_GRACE_SECONDS)
            raise _exit_error(process.exitcode, task_label) from None
    finally:
        receiver.close()
        if process.is_alive():
            process.join(_EXIT_AFTER_RESULT_SECONDS)
        _stop_process(process)

    if status == "error":
        raise IsolatedSubprocessError("child_error", f"{task_label}: 子进程内解析失败\n{payload}")
    logger.info("%s finished in %.1fs", task_label, time.monotonic() - started_at)
    return payload


def _exit_error(exitcode: Optional[int], task_label: str) -> IsolatedSubprocessError:
    if exitcode is not None and exitcode < 0:
        signal_number = -exitcode
        hint = "，很可能是内存超限被 OOM killer 杀掉" if signal_number == signal.SIGKILL else ""
        return IsolatedSubprocessError("killed", f"{task_label}: 子进程被信号 {signal_number} 终止{hint}")
    return IsolatedSubprocessError("crashed", f"{task_label}: 子进程异常退出，退出码 {exitcode}")


def _stop_process(process: Any) -> None:
    """先 terminate，宽限期后 kill，最后 join，保证不留下僵尸或孤儿进程。"""
    if process.is_alive():
        process.terminate()
        process.join(_TERMINATE_GRACE_SECONDS)
    if process.is_alive():
        process.kill()
    process.join()


def _child_entry(sender: Any, parent_pid: int, target: Callable[..., Any], args: tuple) -> None:
    """子进程入口：先绑定生命周期和 OOM 优先级，再执行 target 并把结果或错误回传。"""
    _bind_lifetime_to_parent(parent_pid)
    _prefer_as_oom_victim()
    try:
        result = target(*args)
        message = ("ok", result)
    except BaseException:
        # 异常对象不一定能 pickle，只回传格式化后的堆栈文本。
        message = ("error", traceback.format_exc()[-_CHILD_TRACEBACK_LIMIT:])
    try:
        sender.send(message)
    except Exception:
        # 结果无法 pickle 时改为回传错误说明，避免父进程只看到 EOF。
        sender.send(("error", traceback.format_exc()[-_CHILD_TRACEBACK_LIMIT:]))
    finally:
        sender.close()


def _bind_lifetime_to_parent(parent_pid: int) -> None:
    """Linux 下让内核在父线程退出时向本进程发送 SIGKILL；其他平台不支持，直接跳过。"""
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
            logger.warning("prctl(PR_SET_PDEATHSIG) failed: errno=%s", ctypes.get_errno())
    except (OSError, AttributeError):
        logger.warning("prctl unavailable; child will not follow parent death", exc_info=True)
    # 父进程可能在 prctl 生效前就已退出，此时本进程已被 init 收养，不会再收到信号，只能自行退出。
    if os.getppid() != parent_pid:
        os._exit(1)


def _prefer_as_oom_victim() -> None:
    """把 oom_score_adj 调到最大；调高不需要特权，非 Linux 平台没有该文件时直接跳过。"""
    try:
        with open("/proc/self/oom_score_adj", "w", encoding="ascii") as handle:
            handle.write("1000")
    except OSError:
        pass
