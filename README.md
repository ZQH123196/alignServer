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

## 多语言约束

块级修复（`cloud_hotword.py` 的 `_decide_chunk_repairs` / `_repair_chunk_words`）与语言无关，靠下面五条约束成立。**改这两个函数之前必须先读完本节。**

1. **判据只比相对比例。** 参考语速 `R` 与坏块语速 `r` 同单位（有效 token / 秒），判据是比值 `r / R` 落在 `[CHUNK_RATE_BAND_LOW, CHUNK_RATE_BAND_HIGH]` 内。禁止改成绝对 token/秒阈值：绝对值随语言、音色、语速设置一起变，等于给每种语言配一套魔数。
2. **参考语速来自本次请求内未判坏的块。** `R = Σ(未判坏块的有效 token 数) / Σ(未判坏块的音频时长)`。一次请求 = 一个章节的一种语言，故 `R` 天然带语言与音色，不落盘、不预标定、不跨请求复用。没有未判坏的块时判 `no-good-chunk`，不做任何摊平。
3. **修复只重摊缺陷段，权重是 token 长度。** 缺陷段 = 有效且零时长的连续 token（无词覆盖的字已由 `_rebuild_words` 按最近锚点摊过，不再动）。重摊窗口取段前锚点的 end 与段后锚点的 start，块首/块尾段才用 `start_sec` / `end_sec` 兜。健康 token 的观测时间原样保留 —— 整块推平会为 18 个字毁掉同块 62 个可信时间（2026-10-28 第 8 章）。权重 = token 字符数，每 token 至少 `CHUNK_TOKEN_MIN_SPAN_SEC`。
4. **标点不占时间。** 标点继承前一个有效 token 的 end（块首标点继承块起点）。「有效」用 Unicode 类别判定，不靠 ASCII 白名单：任何脚本的字母与数字都算，符号、标点、空白都不算。
5. **token 切分只有一处。** `_tokenize_reference` 是唯一实现，rebuild、trim、块诊断、修复四处共用。中文逐字符，其他语言 `\w+|[^\w]`（单词 + 单个标点）。这套切法与客户端 `repairWordTimestamps` 同构，**改一侧必须同批改另一侧**。

判坏与修复的常量都在 `cloud_hotword.py` 顶部：`MAX_REPAIRED_CHUNKS_PER_REQUEST`（每请求可修复的块数上限）、`CHUNK_RATE_BAND_LOW/HIGH`（语速比值容差带）、`CHUNK_ZERO_RUN_MIN`（零时长有效 token 的连续段下限）、`CHUNK_UNCOVERED_MIN_CHARS`（单块无词覆盖有效字数的下限）、`CHUNK_COVERAGE_TOLERANCE_SEC`（词流对块窗口的覆盖率容差）、`CHUNK_TOKEN_MIN_SPAN_SEC`（重摊后单个 token 的最小跨度）、`CHUNK_ANCHOR_SEARCH_CHARS`（词锚定前跳上限）。阈值只比相对量，禁止改成绝对 token/秒。

`start_sec` / `end_sec`（前一块已输出的最大 end、下一块首个保留词的起点）只兜块首与块尾的缺陷段；块内缺口一律取邻居时间，否则整块会被平移。越界会让接缝处时间回退，那是 Fail-Fast 的锚点错位判据。

### 词锚定（`_rebuild_words`）

rebuild **不按序号配对**。FA 的输出单位是词，不是字：一串阿拉伯数字在 FA 词表里是一个词（`1500` 一段），参考文本却按字切成四份。按序号硬贴时，该词之后每个 token 都错位一个词，块尾若干 token 拿不到词、落成零时长。

规则：

1. 每个 FA 词的文本就是原文的一段连续字符（FA 只回声，不改写）。顺序定位后，按**有效字符数**把该词的时长摊给它覆盖的字符。
2. 没被任何词覆盖的字符（FA 真丢字）夹在前后两个已锚定字符之间按字数摊；头部贴块起点，尾部贴块终点。
3. 标点仍继承前一个 token 的 end。
4. 定位失败时**宁可不锚**：允许前跳 `CHUNK_ANCHOR_SEARCH_CHARS` 个字符，超出即放弃该词。错锚会让整块时间跳变，比留一个缺口危险。
5. 判据看「无词覆盖的有效字数」（`_diagnose_chunk`），**禁止再用「字符数 vs FA 词数」** —— 那会把所有数字块判坏。2026-10-27 事故：三章 54 块里 15 块因此被判 `fa-short`，进而整章作废并回退重跑 tts，而音频本身是对的。

自证式前提（不是运行时断言）：原文去掉标点与空白，应当等于所有 FA 词文本按顺序拼接。实测 ch8 / ch39 / ch45 三章完全相等，差 0 字。对不上时 `_locate_word` 返回 `None`，该段按缺口处理并在 `无词覆盖=` 里报数，不静默错位。

**无词覆盖不等于音频缺字。** 2026-10-28 第 8 章：同一章两次对齐，云端 ASR 覆盖率 98.6% / 98.4%（音频确实念了面板行），而 FA 在该块只给出 68 个词 / 118 字，三块分别有 18、5、15 字没有对上词。差额来自 FA 侧。阈值因此从 5 提到 `CHUNK_UNCOVERED_MIN_CHARS = 10`，与零时长判据的 `CHUNK_ZERO_RUN_MIN = 5` 拆成两个常量。

块被判坏时另打一行 `[块词序]`：FA 词文本、未对位词文本、块原文。判「FA 丢词」还是「锚定拒绝」只看这一行 —— 词表里没有该段的词是模型侧丢词；词在表里却未对位是 `_locate_word` 的边界判定拒绝。

**本地分支未适用词锚定**：`pipeline.py` 的 `_rebuild_words_from_ground_truth` 仍是「有效 token 按序号 1:1 贴 FA 词」，数字串会以完全相同的方式错位。本次只改云分支（`cloud_hotword.py`）。

## 错误响应

除 200 的对齐结果外，本服务只用两种失败形态。调用方按 `detail.error.type` 分派，不读消息文本。

| `type` | HTTP | 含义 | 调用方应有动作 |
|---|---|---|---|
| `invalid_request_error` | 400 | 请求参数问题（模型名、语言、缺 `prompt`、缺省校验） | 改请求。原样重发必然同样被拒 |
| `audio_defect` | 400 | 这份音频与参考文本不符（当前为整章有效字缺失段超过单段容差） | 作废本章产物并重生成音频，不要原样重发 |
| `alignment_threshold_error` | 400 | 首尾块对齐失败 | 查音频与参考文本。原样重发无意义 |
| `fa_timestamp_error` | 400 | FA 时间戳校验未通过 | 同上 |
| `server_error` | 500 | 服务自身故障（含 GPU OOM 耗尽） | 可重试 |

裁决体形状：

```json
{"detail": {"error": {"message": "…", "type": "audio_defect", "reason": "coverage-gap", "file": "merged_<书名>_<语言>_<序号>_<章标题>.opus"}}}
```

`message` 首行固定是 `章节文件：<文件名>`。本服务只收到音频与参考文本，章节身份的唯一载体是文件名。不带它，调用方看到的就是「某次请求缺了 116 字」，无法定位是哪一章。

`audio_defect` 与 `invalid_request_error` 共用 400，但处置相反，故必须是两个 type。覆盖率不足发生在切块之前，因此这类响应里没有时间戳。

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
    ├── test_chunk_repair.py            块级判据与词锚定
    ├── test_coverage_diagnosis.py      覆盖率诊断文案
    ├── test_env_config.py              配置 / 依赖锁 / 硬件三层检查
    ├── test_fa_reference_span.py       FA 参考文本的区间映射
    ├── test_fa_timestamp_validation.py FA 时间戳校验、接缝钳制与结构守门
    ├── test_fallback_chunks.py         兜底切块
    ├── test_hotword_matching.py        热词匹配
    ├── test_norm_contract.py           归一化契约
    ├── test_occurrence_matching.py     锚点匹配
    ├── test_term_selection.py          热词选词
    ├── test_utils.py                   工具函数
    ├── testCurl_api.sh                 端到端冒烟（需服务已起）
    ├── testCurlLocalFile.sh
    └── testCurlLocalFileAsync.sh
```

源码平铺是 Python 惯例：模块间是平铺 import（`from pipeline import ...`），建 `src/` 会破坏它。测试则收进 `test/`，靠根目录的 `conftest.py` 把项目根补回 `sys.path`——否则直接跑 `pytest` 会 `ModuleNotFoundError`，只有 `python -m pytest` 能通（它顺手把 cwd 塞进了 `sys.path`，属于掩盖问题）。

## 测试

```bash
uv run pytest -q                              # 需要 dev group（uv 默认启用）
python -m unittest discover -s test -t . -p "test_*.py"
```

`test/test_env_config.py` 里的 `ShippedLockfileTests` 不 mock 任何东西，真读一次 `uv.lock` 与 `pyproject.toml`，守住「锁文件本身自洽」和「直接依赖都钉了版本」。
