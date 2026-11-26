import os
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import env_config
from env_config import (
    StartupConfigError,
    check_device_capability,
    check_lockfile,
    load_server_env,
)

# 本文件在 test/ 下，项目根是上一级。锁文件与 pyproject 都在根上，
# 用 __file__ 的 parent 会指向 test/ 而找不到它们。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALID_ENV = {
    "DEVICE": "cpu",
    "HOST": "127.0.0.1",
    "PORT": "6008",
    "dashscope_apikey": "sk-0123456789abcdef",
}


class EnvConfigTests(unittest.TestCase):
    """启动配置校验：配置不完整必须在启动时拒绝，禁止拖到请求处理中途才报错。"""

    def setUp(self):
        self._saved_environ = dict(os.environ)
        self.addCleanup(os.environ.update, self._saved_environ)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _load(self, **overrides):
        os.environ.clear()
        env = dict(VALID_ENV)
        env.update(overrides)
        for key, value in env.items():
            if value is not None:
                os.environ[key] = str(value)
        return load_server_env()

    def _model_dir(self, with_config: bool = True) -> str:
        path = Path(self.tmp.name) / "model"
        path.mkdir(parents=True, exist_ok=True)
        if with_config:
            (path / "config.json").write_text("{}")
        return str(path)

    def test_missing_dashscope_key_rejects_startup(self):
        # Act & Assert
        with self.assertRaises(StartupConfigError) as ctx:
            self._load(dashscope_apikey=None)
        self.assertIn("dashscope_apikey", "；".join(ctx.exception.problems))

    def test_uppercase_dashscope_key_alias_is_accepted(self):
        # Act
        env = self._load(dashscope_apikey=None, DASHSCOPE_API_KEY="sk-0123456789abcdef")

        # Assert
        self.assertEqual(env.dashscope_apikey, "sk-0123456789abcdef")

    def test_non_integer_port_rejects_startup(self):
        # Act & Assert
        with self.assertRaises(StartupConfigError) as ctx:
            self._load(PORT="not-a-port")
        self.assertIn("PORT", "；".join(ctx.exception.problems))

    def test_missing_local_model_dir_rejects_startup(self):
        # Arrange：仓库 ID 形态放行，路径形态必须已下载
        model_dir = self._model_dir(with_config=False)

        # Act & Assert
        with self.assertRaises(StartupConfigError) as ctx:
            self._load(ASR_MODEL_PATH=model_dir, FA_MODEL_PATH="/definitely/not/downloaded")
        problems = "；".join(ctx.exception.problems)
        self.assertIn("config.json", problems)
        self.assertIn("目录不存在", problems)

    def test_valid_startup_config_is_reported_without_leaking_api_key(self):
        # Act
        env = self._load()

        # Assert
        self.assertEqual((env.host, env.port, env.device), ("127.0.0.1", 6008, "cpu"))
        self.assertNotIn("sk-0123456789abcdef", "；".join(env.describe()))


class DeviceValidationTests(unittest.TestCase):
    """DEVICE 写错时不能静默走错分支：pipeline 靠它选 dtype 与 attention 实现。"""

    def setUp(self):
        self._saved_environ = dict(os.environ)
        self.addCleanup(os.environ.update, self._saved_environ)

    def _load(self, device):
        os.environ.clear()
        for key, value in VALID_ENV.items():
            os.environ[key] = device if key == "DEVICE" else value
        return load_server_env()

    def test_gpu_spelling_is_rejected_because_valid_value_is_cuda(self):
        # Act & Assert：用户最可能写成 gpu，但代码里判断的是 cuda
        with self.assertRaises(StartupConfigError) as ctx:
            self._load("gpu")
        self.assertIn("DEVICE", "；".join(ctx.exception.problems))

    def test_typo_device_is_rejected(self):
        # Act & Assert
        with self.assertRaises(StartupConfigError) as ctx:
            self._load("cpuu")
        self.assertIn("DEVICE", "；".join(ctx.exception.problems))

    def test_cuda_is_normalized_to_lowercase(self):
        # Act
        env = self._load("CUDA")

        # Assert
        self.assertEqual(env.device, "cuda")


class LockfileTests(unittest.TestCase):
    """依赖锁自检：绕开 uv.lock 装过包的环境必须在启动时被抓住。

    检查是单向的——遍历实际已装的包，比对它被锁钉成的版本。
    所以这里构造的是「已装分布」，而不是锁里的包清单。
    """

    def _lockfile(self, body: str) -> Path:
        path = Path(self.tmp.name) / "uv.lock"
        path.write_text(body, encoding="utf-8")
        return path

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _check(self, lock_body: str, installed: dict[str, str]):
        """installed 映射包名 → 实际版本，作为环境里已装包的集合。"""
        path = self._lockfile(lock_body)
        fake = [
            types.SimpleNamespace(metadata={"Name": name}, version=ver)
            for name, ver in installed.items()
        ]
        with (
            patch.object(env_config, "LOCK_FILE", path),
            patch.object(env_config, "distributions", lambda: fake),
        ):
            problems: list[str] = []
            check_lockfile(problems)
            return problems

    def test_missing_lock_file_is_reported_with_its_path(self):
        # Arrange：报错必须带完整路径，否则用户不知道该去哪找锁文件
        absent = Path(self.tmp.name) / "absent.lock"

        # Act
        with patch.object(env_config, "LOCK_FILE", absent):
            problems: list[str] = []
            check_lockfile(problems)

        # Assert
        self.assertTrue(any(str(absent) in p for p in problems))

    def test_consistent_environment_has_no_problem(self):
        # Act
        problems = self._check(
            '[[package]]\nname = "soundfile"\nversion = "0.14.0"\n',
            {"soundfile": "0.14.0"},
        )

        # Assert
        self.assertEqual(problems, [])

    def test_version_drift_is_reported_with_both_versions(self):
        # Act
        problems = self._check(
            '[[package]]\nname = "transformers"\nversion = "4.57.6"\n',
            {"transformers": "4.40.0"},
        )

        # Assert
        joined = "；".join(problems)
        self.assertIn("transformers", joined)
        self.assertIn("4.40.0", joined)
        self.assertIn("4.57.6", joined)

    def test_packages_absent_from_the_lock_are_not_reported(self):
        # Act：手工装的包（如可选的 flash-attn）不在锁里，不属本检查范围
        problems = self._check(
            '[[package]]\nname = "soundfile"\nversion = "0.14.0"\n',
            {"soundfile": "0.14.0", "flash-attn": "2.8.3"},
        )

        # Assert
        self.assertEqual(problems, [])

    def test_platform_excluded_and_virtual_packages_are_not_reported_missing(self):
        # Arrange：这三类包在锁里但本就不该装，正向遍历锁会误报。
        # colorama 仅 win32、modelscope 属未启用的 ms extra、根包是 virtual。
        lock = (
            '[[package]]\nname = "long-audio-aligner"\nversion = "0.1.0"\n'
            '[[package]]\nname = "colorama"\nversion = "0.4.6"\n'
            '[[package]]\nname = "modelscope"\nversion = "1.40.2"\n'
            '[[package]]\nname = "soundfile"\nversion = "0.14.0"\n'
        )

        # Act：环境里只有该装的
        problems = self._check(lock, {"soundfile": "0.14.0"})

        # Assert
        self.assertEqual(problems, [])

    def test_torch_cpu_build_counts_as_match(self):
        # Arrange：CPU 环境装的是带 +cpu 后缀的构建，锁里记的是基版本号。
        # 两者在 PEP 440 下不等，但确实是同一个锁定版本的不同二进制变体。
        problems = self._check(
            '[[package]]\nname = "torch"\nversion = "2.14.1"\n',
            {"torch": "2.14.1+cpu"},
        )

        # Assert
        self.assertEqual(problems, [])

    def test_torch_genuinely_different_version_is_still_reported(self):
        # Act：后缀剥离不能把真正的版本差异也放过
        problems = self._check(
            '[[package]]\nname = "torch"\nversion = "2.14.1"\n',
            {"torch": "2.5.1+cpu"},
        )

        # Assert
        self.assertIn("torch", "；".join(problems))

    def test_name_normalization_matches_underscore_to_dash(self):
        # Act：锁里写 some_pkg，已装元数据写 some-pkg，属同一包不该报漂移
        problems = self._check(
            '[[package]]\nname = "some_pkg"\nversion = "1.0.0"\n',
            {"some-pkg": "1.0.0"},
        )

        # Assert
        self.assertEqual(problems, [])


class DeviceCapabilityTests(unittest.TestCase):
    """DEVICE 与实际 CUDA 能力矛盾时必须在启动期拦住。

    锁文件管版本号、管不了二进制变体：CPU 版 torch 装在 GPU 机器上版本号完全一致，
    但表现为「模型加载到一半才炸」。
    """

    def _fake_torch(self, *, cuda_available: bool, device_count: int = 1, cuda: str = "12.8"):
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: cuda_available,
            device_count=lambda: device_count,
            is_bf16_supported=lambda: True,
        )
        torch.version = types.SimpleNamespace(cuda=cuda)
        return {"torch": torch}

    def test_cuda_device_without_cuda_support_is_rejected(self):
        # Act
        with patch.dict(sys.modules, self._fake_torch(cuda_available=False)):
            problems: list[str] = []
            receipt = check_device_capability("cuda", problems)

        # Assert
        self.assertIn("CUDA 不可用", "；".join(problems))
        self.assertEqual(receipt, "")

    def test_cuda_device_receipt_reports_probe_result(self):
        # Act
        with patch.dict(sys.modules, self._fake_torch(cuda_available=True)):
            problems: list[str] = []
            receipt = check_device_capability("cuda", problems)

        # Assert
        self.assertEqual(problems, [])
        self.assertIn("CUDA 版本=12.8", receipt)
        self.assertIn("bf16=支持", receipt)

    def test_cpu_device_does_not_require_cuda(self):
        # Act：CPU 环境里 CUDA 不可用是正常的，不该报错
        with patch.dict(sys.modules, self._fake_torch(cuda_available=False)):
            problems: list[str] = []
            receipt = check_device_capability("cpu", problems)

        # Assert
        self.assertEqual(problems, [])
        self.assertIn("CPU 核数=", receipt)
        self.assertIn("CPU 构建", receipt)

    def test_cpu_device_with_cuda_build_is_reported_as_reversed(self):
        # Act & Assert：CPU 环境装着 CUDA 版 torch，同样是变体装错，必须指出而非放过
        with patch.dict(sys.modules, self._fake_torch(cuda_available=True)):
            problems: list[str] = []
            receipt = check_device_capability("cpu", problems)

        self.assertIn("CUDA 版 torch", "；".join(problems))
        self.assertEqual(receipt, "")


class ShippedLockfileTests(unittest.TestCase):
    """仓库里提交的 uv.lock 必须本身自洽。

    这条不 mock 任何东西：锁文件写坏了、版本号手改成不存在、直接依赖忘了钉版本，
    都会被它抓住——这类问题只有真读一次文件才会暴露。
    """

    def test_committed_lock_file_parses_and_covers_declared_dependencies(self):
        # Arrange
        lock_path = _PROJECT_ROOT / "uv.lock"
        pyproject = _PROJECT_ROOT / "pyproject.toml"

        # Act
        import tomllib

        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
        project = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        locked = {pkg["name"] for pkg in lock.get("package", []) if pkg.get("name")}

        # Assert
        self.assertTrue(lock.get("package"), "uv.lock 里没有任何 package 记录")
        for direct in project["project"]["dependencies"]:
            # qwen-asr 走 URL 而非版本号，其条目在锁文件里的 name 仍是 qwen-asr
            name = re.match(r"[A-Za-z0-9._-]+", direct)
            self.assertIsNotNone(name, f"无法解析依赖名：{direct}")
            self.assertIn(name.group(0).lower(), locked, f"直接依赖 {name.group(0)} 不在 uv.lock 里")
            # 直接依赖必须钉住来源：写死版本号，或钉到具体 commit 的源码包
            self.assertTrue(
                "==" in direct or "@" in direct,
                f"直接依赖既未钉版本也未钉源码 commit：{direct}",
            )

    def test_qwen_asr_is_pinned_to_a_commit_not_a_branch(self):
        # Arrange：滚动分支引用每次安装代码都可能不同，是无法复现的依赖
        pyproject = _PROJECT_ROOT / "pyproject.toml"

        # Act
        import tomllib

        project = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        qwen = [d for d in project["project"]["dependencies"] if d.startswith("qwen-asr")]
        dependency = qwen[0] if qwen else ""

        # Assert
        self.assertIn("@", dependency)
        url = dependency.split("@", 1)[1].strip()
        self.assertNotIn("refs/heads", url, f"qwen-asr 仍指向滚动分支：{url}")
        self.assertRegex(url, r"/[0-9a-f]{40}\.zip$", "qwen-asr 的源码引用必须是完整 commit sha")

    def test_freshly_installed_environment_produces_no_false_alarm(self):
        # Arrange：模拟「按锁装好」的环境——已装集 = 锁里排除本就不该装的那些包。
        # 若检查逻辑退化成正向遍历锁全集，虚拟根包、仅 win32 的 colorama、
        # 未启用 extra 的 modelscope 都会被误报成「未安装」，服务直接拒绝启动。
        import tomllib

        lock_path = _PROJECT_ROOT / "uv.lock"
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
        installed = {
            pkg["name"]: pkg["version"]
            for pkg in lock.get("package", [])
            if pkg.get("name")
            and pkg.get("source", {}).get("virtual") is None
            and pkg["name"] not in ("colorama", "tzdata", "audioop-lts", "modelscope")
        }
        fake = [
            types.SimpleNamespace(metadata={"Name": name}, version=ver)
            for name, ver in installed.items()
        ]

        # Act
        with patch.object(env_config, "distributions", lambda: fake):
            problems: list[str] = []
            check_lockfile(problems)

        # Assert
        self.assertEqual(problems, [], f"按锁装好却报了 {len(problems)} 处假漂移")


if __name__ == "__main__":
    unittest.main()
