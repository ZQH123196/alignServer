"""本地异步日志：调用线程只做格式化与入队，落盘由后台线程批量完成。

输出端由 main.py 在配置校验通过后调用 install(path) 切换。未调用时是
SyncLogWriter，直接写 stdout 并即时 flush（与旧行为一致）。

性能约定：
- 调用线程不碰文件句柄、不做 fsync。`emit()` 只拼字符串再放进队列。
- 后台线程一次取出队列里已有的全部条目，拼成一个块再写，一次事件循环耗一个 write。
- 队列满时不阻塞调用线程。INFO 直接丢弃；WARN 及以上转写 stderr，绝不让关键
  错误日志随队列一起消失。
- 文件超过 MAX_BYTES 由后台线程自行轮转，不依赖 pm2-logrotate。轮转只改文件名字，
  不做阻塞式拷贝压缩。
- 写盘失败按连续次数计，超过 MAX_CONSECUTIVE_WRITE_FAILURES 直接结束进程。
  日志文件写不进去等同于服务失去可观测性，继续跑只会让故障无从排查；
  结束进程交给 pm2 拉起，比留一个无声服务的代价小。
"""

import atexit
import os
import queue
import sys
import threading
import time

QUEUE_MAX_ITEMS = 20000
FLUSH_EVERY_LINES = 200
FLUSH_INTERVAL_SEC = 1.0
MAX_BYTES = 64 * 1024 * 1024
ROTATE_BACKUP_COUNT = 5

# 连续写盘失败到这么多次就结束进程。一次失败可能只是瞬时占用，连续失败说明盘或权限
# 已经没了，再等下去只会丢更多日志。
MAX_CONSECUTIVE_WRITE_FAILURES = 3

# 单行的保留上限只作用在 log() 的单行场景；log_block 逐行处理，堆栈不受此限。
MAX_LINE_CHARS = 4000


class AsyncLogWriter:
    """后台写盘线程。用组合持有队列，避免继承 queue.Queue 时与线程方法混淆。"""

    def __init__(self, path: str) -> None:
        self.path = path
        self._queue: "queue.Queue[str | None]" = queue.Queue(maxsize=QUEUE_MAX_ITEMS)
        # dropped 用计数器的原子自增，不借队列锁：队列锁只保护 put_nowait，
        # 写盘期间绝不能持有它，否则慢磁盘会顺着锁把调用线程一起拖住。
        self._dropped = 0
        self._last_reported_dropped = 0
        self._closed = False
        self._consecutive_failures = 0
        self._thread = threading.Thread(target=self._drain_loop, name="async-log-writer", daemon=True)
        self._thread.start()

    def emit(self, text: str) -> bool:
        """调用线程侧。返回是否入队成功；失败时调用方必须自行兜底关键级别。"""
        if self._closed:
            # 关闭期间的写入同样计入丢弃，运维能从计数看出停服时还有多少日志被打断。
            self._dropped += 1
            return False
        try:
            self._queue.put_nowait(text)
            return True
        except queue.Full:
            self._dropped += 1
            return False

    def dropped_count(self) -> int:
        return self._dropped

    def close(self, timeout: float = 2.0) -> None:
        """冲洗队列并停线程。只用于进程退出，调用后本 writer 不再接收条目。

        顺序不能反：先置关闭标记拒收新条目，再投哨兵让线程把已有条目写完，
        最后才等线程退出。反过来做，停线程之后写进来的条目就没有消费者了。
        """
        if self._closed:
            return
        self._closed = True
        # 队列满时投不进哨兵。此时线程仍在消费，等它腾出位置再投。
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self._queue.put_nowait(None)
                break
            except queue.Full:
                time.sleep(0.005)
        self._thread.join(timeout)

    # ── 后台线程 ────────────────────────────────────────────────────────

    def _drain_loop(self) -> None:
        pending: list[str] = []
        last_flush = time.monotonic()
        while True:
            try:
                item = self._queue.get(timeout=FLUSH_INTERVAL_SEC)
            except queue.Empty:
                item = ""

            if item is None:
                self._write_block(pending)
                return

            if item:
                pending.append(item)
                # 一次取出队列里此刻已有的全部条目，合并成一次写。
                while True:
                    try:
                        more = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if more is None:
                        self._write_block(pending)
                        return
                    pending.append(more)

            if pending and (
                len(pending) >= FLUSH_EVERY_LINES
                or time.monotonic() - last_flush >= FLUSH_INTERVAL_SEC
            ):
                self._write_block(pending)
                pending = []
                last_flush = time.monotonic()

    def _write_block(self, lines: list[str]) -> None:
        if not lines:
            return
        new_dropped = self._dropped - self._last_reported_dropped
        if new_dropped:
            self._last_reported_dropped += new_dropped
            lines = lines + [f"[LOGGER] 队列已满，丢弃 {new_dropped} 条日志\n"]
        try:
            self._rotate_if_needed()
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write("".join(lines))
        except OSError as exc:
            self._on_write_failure(exc)
            return
        self._consecutive_failures = 0

    def _on_write_failure(self, exc: OSError) -> None:
        self._consecutive_failures += 1
        # 先保证 stderr 有一条：文件这条路已经不通了。
        sys.stderr.write(
            f"[LOGGER] 写日志失败 {self._consecutive_failures}/{MAX_CONSECUTIVE_WRITE_FAILURES}"
            f" | 文件={self.path} | 原因={exc}\n"
        )
        sys.stderr.flush()
        if self._consecutive_failures >= MAX_CONSECUTIVE_WRITE_FAILURES:
            # 日志写不进去 = 服务已经不可观测。留给 pm2 拉起，不在无声状态下继续处理请求。
            sys.stderr.write("[LOGGER] 日志文件持续不可写，结束进程交由 pm2 重启\n")
            sys.stderr.flush()
            os._exit(1)

    def _rotate_if_needed(self) -> None:
        try:
            if os.path.getsize(self.path) < MAX_BYTES:
                return
        except OSError:
            return
        # 只做改名：.4 -> .5、.3 -> .4 …，最老的直接删。比归档压缩快得多，
        # 也不会在后台线程里锁住一个正在写入的大文件。
        oldest = f"{self.path}.{ROTATE_BACKUP_COUNT}"
        if os.path.exists(oldest):
            os.remove(oldest)
        for index in range(ROTATE_BACKUP_COUNT - 1, 0, -1):
            source = f"{self.path}.{index}"
            if os.path.exists(source):
                os.replace(source, f"{self.path}.{index + 1}")
        os.replace(self.path, f"{self.path}.1")


class SyncLogWriter:
    """未启用落盘时的输出端。行为与旧实现一致：直接写 stdout 并 flush。"""

    path = ""

    def emit(self, text: str) -> bool:
        try:
            sys.stdout.write(text)
            sys.stdout.flush()
        except (OSError, ValueError):
            return False
        return True

    def dropped_count(self) -> int:
        return 0

    def close(self, timeout: float = 2.0) -> None:
        return


_writer = SyncLogWriter()


def install(path: str) -> None:
    """安装异步写盘。路径可用性由启动校验保证，这里只管接管输出。"""
    global _writer
    path = os.path.abspath(path)
    writer = AsyncLogWriter(path)
    previous = _writer
    _writer = writer
    previous.close()
    atexit.register(writer.close)


def is_installed() -> bool:
    return isinstance(_writer, AsyncLogWriter)


def file_path() -> str:
    return _writer.path


def dropped_count() -> int:
    return _writer.dropped_count()


def emit(text: str) -> bool:
    """写入一条已经带好换行的完整日志文本。"""
    return _writer.emit(text)


def truncate_if_needed(text: str, limit: int = MAX_LINE_CHARS) -> str:
    """超长截断。调用方必须把换行一起传进来，截断后补的那个换行才不会重复。"""
    if len(text) <= limit:
        return text
    body = text.rstrip("\n")
    return body[:limit] + f"…[截断，原长 {len(body)} 字符]\n"
