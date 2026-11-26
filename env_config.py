"""启动期环境变量与运行前置检查。

约定：环境变量取值不合法、本地模型目录缺失这类"迟早会炸"的问题，一律在服务启动时
一次性报完并拒绝启动，禁止拖到请求处理中途才暴露。校验失败抛 StartupConfigError
（携带全部问题项），由 main.py 用结构化日志打印后退出。
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 必须先于任何 os.environ 读取执行
load_dotenv()

ENV_FILE_HINT = ".env（与 main.py 同目录，由 autoInstall.sh 生成，可手工编辑）"

DEFAULT_DEVICE = "cpu"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 6008
DEFAULT_ASR_MODEL_PATH = "Qwen/Qwen3-ASR-0.6B"
DEFAULT_FA_MODEL_PATH = "Qwen/Qwen3-ForcedAligner-0.6B"

# 百炼 API Key。全仓统一用小写名（与 trans_app config.toml 的 ${env:dashscope_apikey} 同源），
# 大写名作为兼容别名一并接受。
DASHSCOPE_API_KEY_VARS = ("dashscope_apikey", "DASHSCOPE_API_KEY")


class StartupConfigError(RuntimeError):
    """启动配置检查未通过。problems 为可直接落日志的逐行描述。"""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("；".join(problems))


@dataclass(frozen=True)
class ServerEnv:
    device: str
    host: str
    port: int
    asr_model_path: str
    fa_model_path: str
    dashscope_apikey: str

    def describe(self) -> list[str]:
        """启动日志用的配置回执。Key 只输出首尾片段，禁止明文落日志。"""
        key = self.dashscope_apikey
        masked = f"{key[:4]}…{key[-2:]}({len(key)} 位)" if len(key) > 8 else "***"
        return [
            f"DEVICE={self.device}",
            f"ASR_MODEL_PATH={self.asr_model_path}",
            f"FA_MODEL_PATH={self.fa_model_path}",
            f"监听={self.host}:{self.port}",
            f"百炼 Key={masked}",
        ]


_SERVER_ENV: ServerEnv | None = None


def _read(name: str, default: str) -> str:
    return (os.environ.get(name) or "").strip() or default


def _read_api_key() -> str:
    for name in DASHSCOPE_API_KEY_VARS:
        key = (os.environ.get(name) or "").strip()
        if key:
            return key
    return ""


def _check_port(raw: str, problems: list[str]) -> int:
    try:
        port = int(raw)
    except ValueError:
        problems.append(f"PORT={raw!r} 不是整数")
        return 0
    if not 1 <= port <= 65535:
        problems.append(f"PORT={port} 不在 1–65535 范围内")
    return port


def _check_model_path(name: str, value: str, problems: list[str]) -> None:
    """绝对/相对路径形态按本地目录校验；org/name 形态按 HuggingFace 仓库 ID 放行。

    仓库 ID 会在首次启动时联网下载数 GB 权重，属预期行为，不在此拦截；
    路径写错则表现为模型加载阶段深层报错，必须在这里拦住。
    """
    if not value.startswith(("/", "./", "~/", "~")):
        return
    path = Path(value).expanduser()
    if not path.is_dir():
        problems.append(f"{name}={value} 指向的目录不存在（模型未下载或路径写错）")
    elif not (path / "config.json").is_file():
        problems.append(f"{name}={value} 目录内没有 config.json，不是可加载的模型目录")


def load_server_env() -> ServerEnv:
    """校验启动配置，通过则缓存为运行时唯一事实源，不通过抛 StartupConfigError。"""
    problems: list[str] = []

    host = _read("HOST", DEFAULT_HOST)
    port = _check_port(_read("PORT", str(DEFAULT_PORT)), problems)
    asr_model_path = _read("ASR_MODEL_PATH", DEFAULT_ASR_MODEL_PATH)
    fa_model_path = _read("FA_MODEL_PATH", DEFAULT_FA_MODEL_PATH)
    dashscope_apikey = _read_api_key()

    if not host:
        problems.append(f"HOST 为空，请检查 {ENV_FILE_HINT}")
    if not dashscope_apikey:
        problems.append(
            f"缺少百炼 API Key：请在 {ENV_FILE_HINT} 中设置 "
            + " 或 ".join(DASHSCOPE_API_KEY_VARS)
            + "（cloud ASR 分支必需）"
        )
    _check_model_path("ASR_MODEL_PATH", asr_model_path, problems)
    _check_model_path("FA_MODEL_PATH", fa_model_path, problems)

    if problems:
        raise StartupConfigError(problems)

    global _SERVER_ENV
    _SERVER_ENV = ServerEnv(
        device=_read("DEVICE", DEFAULT_DEVICE),
        host=host,
        port=port,
        asr_model_path=asr_model_path,
        fa_model_path=fa_model_path,
        dashscope_apikey=dashscope_apikey,
    )
    return _SERVER_ENV


def server_env() -> ServerEnv:
    """取已通过校验的配置。启动入口先 load_server_env()，走到这里必然已就绪。"""
    if _SERVER_ENV is None:
        raise StartupConfigError(["load_server_env() 尚未执行：启动校验必须先于业务模块读取配置"])
    return _SERVER_ENV
