"""本地异步日志的落盘契约。

覆盖三件怕的事：
1. 队列积压时不能反压调用方 —— 慢磁盘不得冻结对齐链路。
2. INFO 攒批落地要能在有限时间内可见，不能攒到进程退出才出现。
3. 日志文件写不进去必须在启动期拦下，不能跑起来才发现没有日志。

只碰本项目根下的模块与临时目录，不 import main.py（它一导入就付 torch 的代价）。
"""

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import async_logging
import env_config
from env_config import StartupConfigError, load_server_env


class AsyncLogWriterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log_path = str(Path(self.tmp.name) / "alignServer.log")

    def _writer(self):
        writer = async_logging.AsyncLogWriter(self.log_path)
        self.addCleanup(writer.close)
        return writer

    def _read(self) -> str:
        path = Path(self.log_path)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    # 这些用例要等真实的后台线程刷盘，属慢测试（实测每组约 2 秒）。
    # 不 mock 时间：被测的正是「线程何时真的落盘」，假时钟会把这条契约测没。
    def _wait_for(self, predicate, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()

    def test_lines_are_written_in_order(self):
        # Arrange
        writer = self._writer()

        # Act
        for index in range(5):
            writer.emit(f"line-{index}\n")

        # Assert
        self.assertTrue(self._wait_for(lambda: self._read().count("\n") == 5))
        self.assertEqual(
            self._read().splitlines(),
            [f"line-{index}" for index in range(5)],
        )

    def _writer_with_stalled_queue(self, maxsize: int):
        """造一个队列已满、后台线程不再消费的 writer。

        后台线程停掉之后队列只进不出，用公开的 emit() 灌到满即可。
        """
        writer = self._writer()
        writer.close()
        self.assertTrue(self._wait_for(lambda: not writer._thread.is_alive()))
        writer._closed = False  # 只为让 emit() 继续尝试入队
        writer._queue.maxsize = maxsize
        while writer.emit("fill\n"):
            pass
        writer._dropped = 0  # 灌满过程中必然丢过一次，从这里起算才看得清
        return writer

    def test_full_queue_does_not_block_caller(self):
        # Arrange
        writer = self._writer_with_stalled_queue(maxsize=2)

        # Act
        started = time.monotonic()
        rejected = writer.emit("overflow\n")
        elapsed = time.monotonic() - started

        # Assert
        self.assertFalse(rejected)
        self.assertLess(elapsed, 0.05, "队列满时调用方不得被阻塞")
        self.assertEqual(writer.dropped_count(), 1)


    def test_dropped_count_is_reported_into_the_file(self):
        # Arrange
        writer = self._writer_with_stalled_queue(maxsize=1)
        writer.emit("overflow\n")

        # Act
        writer._write_block(["kept\n"])

        # Assert
        content = self._read()
        self.assertIn("kept\n", content)
        self.assertIn("[LOGGER] 队列已满，丢弃 1 条日志", content)

    def test_rotation_renames_instead_of_copying(self):
        # Arrange：把阈值压到极小，让首批写入立刻触发轮转
        writer = self._writer()
        writer.emit("first\n")
        self.assertTrue(self._wait_for(lambda: self._read() == "first\n"))

        # Act
        with patch.object(async_logging, "MAX_BYTES", 1):
            writer.emit("second\n")
            # 轮转由后台线程执行，触发一次刷写后观察
            self.assertTrue(self._wait_for(lambda: Path(self.log_path + ".1").exists(), 3.0))

        # Assert
        self.assertEqual(Path(self.log_path + ".1").read_text(encoding="utf-8"), "first\n")

    def test_close_flushes_pending_lines(self):
        # Arrange
        writer = async_logging.AsyncLogWriter(self.log_path)
        writer.emit("pending\n")

        # Act
        writer.close()

        # Assert
        self.assertEqual(self._read(), "pending\n")

    def test_emit_after_close_is_rejected_without_blocking(self):
        # Arrange：停服后请求线程仍可能在写日志，这条路径不能阻塞
        writer = self._writer()
        writer.close()

        # Act
        started = time.monotonic()
        accepted = writer.emit("late\n")
        elapsed = time.monotonic() - started

        # Assert
        self.assertFalse(accepted)
        self.assertLess(elapsed, 0.05)
        self.assertEqual(writer.dropped_count(), 1, "关闭后的拒收也要计数，才能看出停服打断了多少日志")

    def test_concurrent_emit_during_drain_loses_nothing(self):
        """停服与请求线程并发写日志时不得静默丢行。

        怕的是：drain 先投哨兵，后台线程见哨兵就退出，随后写进来的条目再也没有消费者。
        """
        # Arrange
        writer = self._writer()
        emitted = 300
        accepted = []
        barrier = threading.Barrier(2)

        def producer():
            barrier.wait()
            for index in range(emitted):
                if writer.emit(f"line-{index}\n"):
                    accepted.append(index)

        producer_thread = threading.Thread(target=producer)
        producer_thread.start()
        barrier.wait()

        # Act：关闭与写入并发进行。主线程让生产者先跑一段再关，
        # 这样既有「关闭前已入队」也有「关闭后被拒」两种情形。
        time.sleep(0.05)
        writer.close()
        producer_thread.join()

        # Assert：凡是被接收的条目都必须出现在文件里，一条都不能少
        self.assertGreater(len(accepted), 0, "关闭前应当至少接收了一条")
        self.assertEqual(
            self._read().count("\n") - self._read().count("[LOGGER]"),
            len(accepted),
            "被接收的条目必须与落盘条目数一致",
        )

    def test_stack_block_keeps_tail_lines(self):
        """多行块按行截断，堆栈尾部（异常类型与最内层帧）必须保留。"""
        block = ["堆栈"] + [f"  File \"module_{index}.py\", line {index}" for index in range(60)]

        # Act：模拟 main.log_block 的拼装
        head = block[0]
        body = "".join(
            "\n  " + async_logging.truncate_if_needed(line + "\n").rstrip("\n")
            for line in block[1:]
        )
        text = head + body + "\n"

        # Assert
        self.assertIn("line 59", text)
        self.assertNotIn("截断", text)

    def test_short_line_is_not_truncated(self):
        self.assertEqual(async_logging.truncate_if_needed("短\n"), "短\n")

    def test_long_line_is_truncated_with_a_marker(self):
        # Arrange
        body = "x" * (async_logging.MAX_LINE_CHARS + 10)

        # Act
        result = async_logging.truncate_if_needed(body)

        # Assert
        self.assertTrue(result.startswith("x" * async_logging.MAX_LINE_CHARS))
        self.assertIn("截断", result)


class SyncFallbackTests(unittest.TestCase):
    def test_default_writer_does_not_touch_disk(self):
        # 未 install 时必须是同步直写 stdout，不能悄悄建出文件
        self.assertFalse(async_logging.is_installed())
        self.assertEqual(async_logging.file_path(), "")


class LoggingFileConfigTests(unittest.TestCase):
    VALID_ENV = {
        "DEVICE": "cpu",
        "HOST": "127.0.0.1",
        "PORT": "6008",
        "dashscope_apikey": "sk-0123456789abcdef",
    }

    def setUp(self):
        self._saved = dict(os.environ)
        self.addCleanup(os.environ.update, self._saved)
        self.addCleanup(os.environ.clear)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _load(self, **overrides):
        os.environ.clear()
        env = dict(self.VALID_ENV)
        env.update(overrides)
        for key, value in env.items():
            if value is not None:
                os.environ[key] = str(value)
        return load_server_env()

    def test_unwritable_log_directory_rejects_startup(self):
        # Arrange：把父路径指成一个普通文件，mkdir 必然失败
        blocker = Path(self.tmp.name) / "not-a-dir"
        blocker.write_text("x")

        # Act & Assert
        with self.assertRaises(StartupConfigError) as ctx:
            self._load(LOGGING_FILE=str(blocker / "sub" / "alignServer.log"))
        self.assertIn("LOGGING_FILE", "；".join(ctx.exception.problems))

    def test_relative_path_resolves_next_to_the_project(self):
        # Act
        env = self._load(LOGGING_FILE="logs/test-alignServer.log")

        # Assert
        self.assertTrue(
            Path(env.logging_file).is_absolute(),
            "相对路径必须按模块目录解析，不能落到进程 cwd",
        )
        Path(env.logging_file).unlink(missing_ok=True)

    def test_default_logging_file_is_used_when_unset(self):
        # Act & Assert
        self.assertTrue(self._load(LOGGING_FILE=None).logging_file.endswith(".log"))


if __name__ == "__main__":
    unittest.main()
