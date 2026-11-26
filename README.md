# Long Audio Aligner

长音频强制对齐服务。给定音频与参考文本，输出逐句时间戳。

- **对齐模式**（`Qwen/Qwen3-ASR-0.6B + Qwen/Qwen3-ForcedAligner-0.6B`）：先用 ASR 定位文本边界，再用 FA 出时间戳。
- **仅对齐**（`Qwen/Qwen3-ForcedAligner-0.6B`）：已有文本边界，只补时间戳。
- **纯 ASR**（`Qwen/Qwen3-ASR-0.6B-hf`）：不产出时间戳。
- **云端热词**（`qwen-audio-3.1-asr-flash-filetrans + Qwen/Qwen3-ForcedAligner-0.6B`）：走百炼 API 做 ASR，本地 FA 出时间戳，需要 `.env` 里的百炼 Key。

## 快速开始

```bash
# GPU 环境（默认）
./autoInstall.sh
./start_alignServer.sh

# CPU 环境
DEVICE=cpu ./autoInstall.sh
./start_alignServer.sh
```

安装脚本全流程无交互，所有选择走环境变量，可直接在 CI 或容器里重跑：

| 变量 | 默认 | 说明 |
|---|---|---|
| `DEVICE` | `cuda` | `cpu` 或 `cuda`，决定 torch 变体 |
| `MODEL_ROOT_DIR` | `$HOME/autodl-fs/models` | 模型落盘根目录 |
| `DOWNLOAD_SOURCE` | `hf` | `hf`（HuggingFace）或 `ms`（魔搭） |
| `HF_TOKEN` | 空 | 提高 HF 限额，公开模型可不给 |

### 包源

`uv.lock` 里写死了包源（当前是**清华镜像**），`uv sync --frozen` 照它下载，所以安装脚本**没有** `PYPI_INDEX` 之类的开关——给了也不生效。境外部署需重新生成锁文件：

```bash
uv lock --index-url https://pypi.org/simple
```

## 配置

复制 `.env.example` 为 `.env` 后修改。字段说明见该文件内的注释。

所有字段都在启动期校验：取值不合法会**一次性报完所有问题**并拒绝启动，不会拖到某个请求中途才炸。已存在的 `.env` 不会被安装脚本覆盖。

## 启动期检查

`main.py` 在加载模型前跑三层检查，任一项不过就打印全部问题并退出（exit 1）：

1. **配置层** — `PORT` 范围、`DEVICE` 枚举、模型路径形态、Python 版本下限。
2. **依赖锁层** — 比对**已装包**的版本与 `uv.lock`。绕开锁跑过一次 `uv pip install <包名>` 会在这里被抓住，而不是几小时后某个请求上炸。
3. **硬件层** — 双向核对 `DEVICE` 与 torch 变体：`DEVICE=cuda` 但 CUDA 不可用（装成了 CPU 版），或 `DEVICE=cpu` 但 CUDA 可用（装反了），都会报错。

依赖锁层是**单向**比对：只查已装包的版本是否与锁一致，不查「锁里有但没装」。后者是安装问题，`uv sync --frozen` 保证它；真漏了会在 import 时报出明确的 `ModuleNotFoundError`，比启动期笼统说「缺 N 个包」好定位。

## 依赖版本纪律

**直接依赖一律写死 `==`，整棵依赖树由 `uv.lock` 钉死。装环境只允许 `uv sync --frozen`。**

裸跑 `uv pip install <包名>` 会绕过锁、静默改版本。历史上被 `silero-vad>=5.1.0` 实际装到 6.x 这类跨大版本变更坑过——6.x API 恰好兼容，纯属运气。

升级流程：

```bash
# 1. 改 pyproject.toml 的版本号
# 2. 重新解析并更新锁文件
uv lock --index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 3. 装到本地并跑测试
uv sync --frozen
uv run pytest -q
# 4. 提交 pyproject.toml 与 uv.lock 两个文件
```

跨大版本（`>=` 语义下会自动发生的那些）先查 API 变更再升。本项目历史上一次跨大版本事故：`silero-vad` 5.x → 6.x。

### qwen-asr 钉在 commit 而非版本号

```toml
"qwen-asr @ https://github.com/QwenLM/Qwen3-ASR/archive/7c6daf77....zip"
```

原声明指向 `refs/heads/main.zip`，即 main 分支的实时快照，每次安装拿到的代码都可能不同，且无法离线复现。上游仓库没有 tag，故钉到 commit sha（`7c6daf77`，2026-06-26）。

升级时改这个 sha 并重跑 `uv lock`。`test_env_config.py` 里有断言守着它不会被改回滚动分支。

### torch 的 CPU / CUDA 变体

锁文件能钉版本，钉不了二进制变体——CPU 与 CUDA 构建版本号相同，`uv.lock` 表达不了「按硬件选变体」。

- **本锁文件是 GPU 基线**（默认 index 上的 CUDA 构建），是主部署形态。
- CPU 环境由安装脚本在装依赖后按锁里的版本号改装 CPU 构建，装完由启动期的硬件探针验证。

**注意**：`start_alignServer.sh` 不做任何依赖同步。CPU 环境装的是 `torch 2.14.1+cpu`，锁里记的是 `2.14.1`，`uv sync` 见到版本号不等会把它重装回 CUDA 构建——CPU 部署静默失效，只是慢一个数量级，没有任何一层会报错。

试过用 `[tool.uv.sources]` 的 marker 按 extra 路由两种变体，**实测失败**：uv 一旦为 torch 声明 source，lock 会把两个变体都解析到该 index，marker 失效。

### 关于 Flash Attention 2

GPU 上的可选加速。`pipeline.py` 有 `except ImportError` 降级到 `sdpa`，装不上不影响可用性，故不进锁文件。

它没有与 torch 版本稳定对应的预编译 wheel（原有脚本钉的是 `cu12torch2.5`，与当前 torch 2.14 已不匹配）。需要时装匹配版本：

```bash
uv pip install flash-attn --no-build-isolation
```

无预编译 wheel 时会退化成源码编译，耗时较长。

## 目录结构

```
alignServer/
├── main.py                    FastAPI 入口与启动期检查
├── pipeline.py                对齐流水线（VAD → ASR → FA）
├── cloud_hotword.py           云端热词分支
├── env_config.py              配置 / 依赖锁 / 硬件三层检查
├── conftest.py                让 test/ 下的测试能导入项目根模块
├── autoInstall.sh             安装（交互式）
├── start_alignServer.sh       启动
├── pyproject.toml             依赖声明（钉版本）
├── uv.lock                    整棵依赖树的精确版本
├── .env.example               配置模板
└── test/
    ├── test_env_config.py     单测
    ├── test_fa_timestamp_validation.py
    ├── test_hotword_matching.py
    ├── smoke_api.sh           端到端冒烟（需服务已起）
    ├── smoke_local.sh
    └── smoke_async.sh
```

源码平铺是 Python 惯例：模块间是平铺 import（`from pipeline import ...`），建 `src/` 会破坏它。测试则收进 `test/`，靠根目录的 `conftest.py` 把项目根补回 `sys.path`——否则直接跑 `pytest` 会 `ModuleNotFoundError`，只有 `python -m pytest` 能通（它顺手把 cwd 塞进了 `sys.path`，属于掩盖问题）。

## 测试

```bash
uv run pytest -q                              # 需要 dev group（uv 默认启用）
python -m unittest discover -s test -t . -p "test_*.py"
```

`test/test_env_config.py` 里的 `ShippedLockfileTests` 不 mock 任何东西，真读一次 `uv.lock` 与 `pyproject.toml`，守住「锁文件本身自洽」和「直接依赖都钉了版本」。
