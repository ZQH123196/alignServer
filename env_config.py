"""启动期环境变量、依赖锁与运行前置检查。

约定：环境变量取值不合法、已装依赖与 uv.lock 不一致、本地模型目录缺失、
DEVICE 与实际 CUDA 能力矛盾这类「迟早会炸」的问题，一律在服务启动时一次性报完
并拒绝启动，禁止拖到请求处理中途才暴露。校验失败抛 StartupConfigError
（携带全部问题项），由 main.py 用结构化日志打印后退出。

三类检查的分工：
- 配置校验（load_server_env）：读 .env 与环境变量，只碰字符串，不 import 任何重依赖。
- 依赖校验（check_lockfile）：比对 uv.lock 与实际已装版本，抓「锁了但没按锁装」的情况。
- 硬件校验（check_device_capability）：抓 DEVICE 与实际 CUDA 能力矛盾的情况。
分开的理由是排查时能一眼看出是哪一层坏了，且配置校验保持零重依赖、秒级返回。
"""

import os
import re
import sys
import tomllib
from dataclasses import dataclass
from importlib.metadata import distributions
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

# 本地日志文件。相对路径按本文件所在目录解析，与 .env 同处。
# .env 里不写这一项时落到此默认值；写进去就必须能写，否则启动期直接失败。
DEFAULT_LOGGING_FILE = "./logs/alignServer.log"

# pyproject 的 requires-python 下限。与之不符的解析器组合会让 torch 的 wheel 选错。
REQUIRED_PYTHON = (3, 12)

# DEVICE 的合法取值。写错时不会被自动纠正，只会静默走错分支（见 _check_device），
# 故显式枚举。
VALID_DEVICES = ("cpu", "cuda")

# 百炼 API Key。全仓统一用小写名（与 trans_app config.toml 的 ${env:dashscope_apikey} 同源），
# 大写名作为兼容别名一并接受。
DASHSCOPE_API_KEY_VARS = ("dashscope_apikey", "DASHSCOPE_API_KEY")

LOCK_FILE = Path(__file__).resolve().parent / "uv.lock"


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
    logging_file: str

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
            f"日志文件={self.logging_file}",
        ]


_SERVER_ENV: ServerEnv | None = None


def _read(name: str, default: str) -> str:
    return (os.environ.get(name) or "").strip() or default


def _normalize(name: str) -> str:
    """包名归一化到 PEP 503 形态，使锁文件与 importlib.metadata 的写法可比。

    锁文件写的是 `Levenshtein`，已装元数据里可能是 `levenshtein`；下划线与连字符同理。
    """
    return re.sub(r"[-_.]+", "-", name).lower()


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


def _check_device(raw: str, problems: list[str]) -> str:
    """DEVICE 必须是 cpu / cuda，大小写不敏感。

    不做别的容错：pipeline.py 靠它选 dtype 与 attention 实现，写错一个字母的后果是
    「按 CPU 的 dtype 去加载本该上 GPU 的模型」，报错点远在模型加载阶段，与根因无关。
    故只统一大小写（CUDA/cuda 属同一写法习惯），其余取值一律拒绝。
    """
    device = (raw or "").strip().lower()
    if device not in VALID_DEVICES:
        problems.append(
            f"DEVICE={raw!r} 不是合法取值，只能是 {' / '.join(VALID_DEVICES)}"
            f"（gpu 请写 cuda），请检查 {ENV_FILE_HINT}"
        )
        return DEFAULT_DEVICE
    return device


def _check_python(problems: list[str]) -> None:
    """解释器版本必须满足 pyproject 的 requires-python。

    不满足时 torch 会挑到不匹配的 wheel，报错发生在安装或 import 期，与根因无关；
    在启动期拦住才能给出「换解释器」这种可执行的结论。
    """
    current = sys.version_info[:2]
    if current < REQUIRED_PYTHON:
        problems.append(
            f"Python {'.'.join(map(str, current))} 低于本项目要求的 "
            f"{'.'.join(map(str, REQUIRED_PYTHON))}，请执行 autoInstall.sh 重建 .venv"
        )


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


def _resolve_log_path(raw: str, problems: list[str]) -> str:
    """解析并实测日志文件可写。写不进去就在启动期失败，不留到第一个请求。

    相对路径按本文件所在目录解析，不按进程 cwd —— 从 systemd、pm2、cron 启动时
    cwd 各不相同，按 cwd 解析会让日志散落在不可预期的位置。
    只走文件句柄，不做 os.access 探测：后者能看到的权限，在真正写入时未必成立。
    """
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8"):
            pass
    except OSError as exc:
        problems.append(f"LOGGING_FILE={path} 无法写入：{exc}。日志文件不可用时不启动")
        return str(path)
    return str(path)


def check_lockfile(problems: list[str]) -> None:
    """比对已装版本与 uv.lock，抓「绕开锁装包」造成的版本漂移。

    比对方向是单向的：遍历**实际已装**的包，看它在锁里被钉成什么版本。
    反过来遍历锁的全集是错的——锁里含三类本就不该装的包，报出来全是假警报：
    - 虚拟根包自身（`[tool.uv] package = false`，uv 不把它装进 venv）；
    - 平台 marker 排除的包（colorama 只被 win32 边引用、tzdata 只在 Windows、
      audioop-lts 要求 Python ≥ 3.13）；
    - 未启用 extra 的包（modelscope 属 ms extra，本项目默认只启用 hf）。

    这也意味着「锁里有但没装」不在本函数的职责内：那属于安装问题，
    autoInstall.sh 负责依赖安装，漏了会在 import 时报出明确的
    ModuleNotFoundError。
    """
    if not LOCK_FILE.is_file():
        problems.append(f"找不到依赖锁文件 {LOCK_FILE}，无从校验已装版本；请先跑 uv sync --frozen")
        return

    try:
        lock = tomllib.loads(LOCK_FILE.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        problems.append(f"依赖锁文件 {LOCK_FILE} 解析失败：{exc}")
        return

    locked: dict[str, set[str]] = {}
    for package in lock.get("package", []):
        name = package.get("name")
        ver = package.get("version")
        if name and ver:
            locked.setdefault(_normalize(name), set()).add(ver)

    drifted: list[str] = []
    for dist in distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        candidates = locked.get(_normalize(name))
        if candidates is None:
            # 锁里没有的包（如可选的 flash-attn）不在本检查范围内
            continue
        actual = dist.version
        if actual not in candidates and actual.split("+", 1)[0] not in {
            c.split("+", 1)[0] for c in candidates
        }:
            drifted.append(f"{name} 实际 {actual} ≠ 锁定的 {'/'.join(sorted(candidates))}")

    if drifted:
        problems.append(
            "已装依赖与 uv.lock 不一致，共 "
            f"{len(drifted)} 处：" + "；".join(drifted[:8])
            + ("；……其余见 uv sync --frozen" if len(drifted) > 8 else "")
            + "。请执行 `uv sync --frozen` 重建环境"
        )


def check_device_capability(device: str, problems: list[str]) -> str:
    """DEVICE 与实际 torch 变体必须自洽，双向校验。

    锁文件只管版本号、管不了二进制变体——CPU 版与 CUDA 版 torch 版本号相同。
    这类事故的表现是「模型加载到一半才炸」或「静默慢一个数量级」，故在启动期双向拦：

    - DEVICE=cuda 但 CUDA 不可用 → 多半装成了 CPU 版 torch；
    - DEVICE=cpu 但 CUDA 可用    → 多半装反了，CPU 环境应装 CPU 版。

    返回探针结果描述（供启动日志回执）；不可用时返回空串，失败原因已进 problems。
    """
    import torch  # 局部导入：配置校验阶段不付几秒的 torch 导入代价

    cuda_available = torch.cuda.is_available()

    if device == "cuda":
        if not cuda_available:
            problems.append(
                "DEVICE=cuda 但当前 torch 报告 CUDA 不可用，多半是装成了 CPU 版 torch。"
                "请用 autoInstall.sh 重装环境（DEVICE=cuda）"
            )
            return ""
        return (
            f"CUDA 设备数={torch.cuda.device_count()}"
            f" | CUDA 版本={torch.version.cuda}"
            f" | bf16={'支持' if torch.cuda.is_bf16_supported() else '不支持，回退 fp16'}"
        )

    if cuda_available:
        problems.append(
            "DEVICE=cpu 但当前 torch 报告 CUDA 可用，环境里装的是 CUDA 版 torch。"
            "CPU 部署应装 CPU 版：请用 autoInstall.sh 重装环境（DEVICE=cpu）"
        )
        return ""
    return f"CPU 核数={os.cpu_count()} | torch 为 CPU 构建"


def load_server_env() -> ServerEnv:
    """校验启动配置，通过则缓存为运行时唯一事实源，不通过抛 StartupConfigError。"""
    problems: list[str] = []

    host = _read("HOST", DEFAULT_HOST)
    port = _check_port(_read("PORT", str(DEFAULT_PORT)), problems)
    device = _check_device(_read("DEVICE", DEFAULT_DEVICE), problems)
    asr_model_path = _read("ASR_MODEL_PATH", DEFAULT_ASR_MODEL_PATH)
    fa_model_path = _read("FA_MODEL_PATH", DEFAULT_FA_MODEL_PATH)
    dashscope_apikey = _read_api_key()
    logging_file_raw = _read("LOGGING_FILE", DEFAULT_LOGGING_FILE)

    if not host:
        problems.append(f"HOST 为空，请检查 {ENV_FILE_HINT}")
    if not dashscope_apikey:
        problems.append(
            f"缺少百炼 API Key：请在 {ENV_FILE_HINT} 中设置 "
            + " 或 ".join(DASHSCOPE_API_KEY_VARS)
            + "（当前为无条件必填：不配则任何模式都起不来，即使只用本地 ASR）"
        )
    _check_python(problems)
    _check_model_path("ASR_MODEL_PATH", asr_model_path, problems)
    _check_model_path("FA_MODEL_PATH", fa_model_path, problems)
    logging_file = _resolve_log_path(logging_file_raw, problems)

    if problems:
        raise StartupConfigError(problems)

    global _SERVER_ENV
    _SERVER_ENV = ServerEnv(
        device=device,
        host=host,
        port=port,
        asr_model_path=asr_model_path,
        fa_model_path=fa_model_path,
        dashscope_apikey=dashscope_apikey,
        logging_file=logging_file,
    )
    return _SERVER_ENV


def server_env() -> ServerEnv:
    """取已通过校验的配置。启动入口先 load_server_env()，走到这里必然已就绪。"""
    if _SERVER_ENV is None:
        raise StartupConfigError(["load_server_env() 尚未执行：启动校验必须先于业务模块读取配置"])
    return _SERVER_ENV
