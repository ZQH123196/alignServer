import os
import tempfile
import unittest
from pathlib import Path

from env_config import StartupConfigError, load_server_env

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


if __name__ == "__main__":
    unittest.main()
