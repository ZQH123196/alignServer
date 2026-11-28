# Long Audio Aligner

长音频强制对齐服务。给定音频与参考文本，输出逐句时间戳。

- **对齐模式**（`Qwen/Qwen3-ASR-0.6B + Qwen/Qwen3-ForcedAligner-0.6B`）：先用 ASR 定位文本边界，再用 FA 出时间戳。
- **仅对齐**（`Qwen/Qwen3-ForcedAligner-0.6B`）：纯透传。整段音频与整章文本各一份，原样交给 FA，不跑 VAD、不推文本边界。音频超过 400s 时返回 400，不切块降级。见下文「纯透传上限」。
- **纯 ASR**（`Qwen/Qwen3-ASR-0.6B-hf`）：不产出时间戳。
- **云端热词**（`qwen-audio-3.1-asr-flash-filetrans + Qwen/Qwen3-ForcedAligner-0.6B`）：走百炼 API 做 ASR，本地 FA 出时间戳，需要 `.env` 里的百炼 Key。

## 环境与启动的唯一入口

**必须用 `autoInstall.sh` 装环境，必须用 `start_alignServer.sh` 启动服务。**

这两个脚本是唯一会被生产运行的脚本。任何绕过它们的做法都视为无效现场：

- 禁止手工 `uv venv` + `uv pip install` 拼环境。
- 禁止直接 `python main.py` 起服务。
- 手工装出的环境里，已装包的版本漂移会被 `check_lockfile` 在启动期拦下。但它只比对**锁里有的包**，锁外的包拦不住。
- 手工起服务没有 pm2 托管，日志位置与生产不同。

排障时可以读 `.venv`，但**修完必须回到这两个脚本重来一遍**，确认问题在脚本路径下也消失。

### 安装

```bash
# GPU 环境
DASHSCOPE_APIKEY=sk-xxx ./autoInstall.sh

# CPU 环境
DEVICE=cpu DASHSCOPE_APIKEY=sk-xxx ./autoInstall.sh
```

安装脚本全流程无交互，所有选择走环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `DEVICE` | `cuda` | `cpu` 或 `cuda`，决定 torch 二进制变体 |
| `DASHSCOPE_APIKEY` | 空 | **必填**。`main.py` 启动期无条件校验，缺失即拒绝启动 |
| `MODEL_ROOT_DIR` | `$HOME/autodl-fs/models` | 模型落盘根目录 |
| `DOWNLOAD_SOURCE` | `hf` | `hf`（HuggingFace）或 `ms`（魔搭） |
| `USE_DOMESTIC` | `y` | `y` 走清华镜像与 HF-Mirror，`n` 走官方源 |
| `UV_CACHE_DIR` | `$HOME/.cache/uv` | uv 缓存目录，只读挂载下需覆盖 |
| `HF_TOKEN` | 空 | 提高 HF 限额，公开模型可不给 |
| `INSTALL_FLASH_ATTN` | `0` | 仅 GPU 生效。设 `1` 时装 Flash Attention 2，失败不中断 |
| `LOGGING_FILE` | `./logs/alignServer.log` | 服务日志文件，写进 `.env`。相对路径按项目目录解析 |

安装脚本做的事，按顺序：

1. 校验 `DEVICE` / `DOWNLOAD_SOURCE` 取值，非法即退出。
2. 创建或复用 `.venv`。
3. `uv sync --frozen` 按 `uv.lock` 装全部依赖（跳过 torch 生态）。
4. 按 `DEVICE` 从 PyTorch 官方 index 装锁钉版本的 torch 变体（CPU 走 `whl/cpu`，CUDA 走 `whl/cu130`）。
5. 断言 torch 变体与 `DEVICE` 自洽：`DEVICE=cuda` 时 `torch.cuda.is_available()` 必须为真，反之亦然。
6. 可选装 Flash Attention 2（`INSTALL_FLASH_ATTN=1`，失败不中断）。
7. 校验 `qwen_asr` 可导入。
8. 下载两个 Qwen 模型与 Silero VAD 权重。
9. 生成 `.env`（已存在则跳过）。

脚本**不起服务**。装完执行 `./start_alignServer.sh`。

### CUDA 版本

CUDA 变体固定为 **CUDA 13.0**（`whl/cu130`）。依据是 `uv.lock` 里 torch 2.14.1 的依赖：`cuda-toolkit 13`、`nvidia-cudnn-cu13`、`nvidia-nccl-cu13`。

不能用 `whl/cu121`：该索引上最高只有 torch 2.5.1，取 2.14.1 必然 `No solution found`。

### 依赖与 torch 变体

基础依赖一律来自 `uv.lock`，脚本不跑 `uv pip install -r pyproject.toml`。

torch 的 CPU / CUDA 构建版本号相同，`uv.lock` 表达不了变体，故分两步：

1. `uv sync --frozen` 装其余全部依赖。
2. 按 `DEVICE` 从 PyTorch index 装同名版本的对应变体。

安装脚本实际执行的是 `uv sync --frozen --inexact --no-install-package torch --no-install-package torchaudio`：`--no-install-package` 让这一步不碰 torch，`--inexact` 让 uv 只补缺、不卸载已装的多余包。**装完 torch 之后不再跑同步。** 不带 `--no-install-package` 时 uv 会把 `torch==2.14.1+cpu` 判为「与锁不符」并卸载，第 2 步装好的变体又被抹掉，形成装好即卸载的死循环。

CPU 路径还须加 `--index-strategy unsafe-best-match` 并同时传 `--index-url` 与 `--extra-index-url`：前者放宽「只用第一个含该包的 index」的默认策略，后者让 PyTorch CPU index 与默认源一起参与解析。少了 `--extra-index-url`，`+cpu` 变体根本不在候选里。

安装脚本的每一步都有进度输出。它会创建或复用 `.venv`、按锁文件装依赖、下载模型、生成 `.env`。

### 启动

```bash
./start_alignServer.sh
```

启动脚本交给 pm2 托管。日志分两处：

| 文件 | 内容 |
|---|---|
| `logs/alignServer.log` | **业务日志，排查先看这里**。由服务自己异步写盘并轮转 |
| `~/.pm2/logs/alignServer-out.log` | 进程 stdout，与业务日志内容重复 |
| `~/.pm2/logs/alignServer-error.log` | stderr，Python 崩溃堆栈与非业务输出 |

## 日志

服务自己写日志文件，不依赖 pm2 收集 stdout。路径由 `.env` 的 `LOGGING_FILE` 决定，默认 `./logs/alignServer.log`。

写盘是异步的：调用线程只拼字符串入队，后台线程批量落盘。排队时长、计算耗时这些高频日志不会拖慢对齐。

| 行为 | 取值 | 说明 |
|---|---|---|
| 写盘时机 | 攒满 200 行或每 1 秒 | 两者取先到。WARN 及以上与 INFO 同队列，不等级别 |
| 队列上限 | 20000 条 | 满了丢弃 INFO 并计数，不阻塞调用线程 |
| 队列满时的 WARN 及以上 | 同步写 stderr | 关键日志不会因为队列堵住而消失 |
| 丢弃上报 | 有新增丢弃时 | 落一行 `[LOGGER] 队列已满，丢弃 N 条日志` |
| 轮转阈值 | 64 MB | 超过即改名成 `.1`，保留 `.1` ~ `.5`，不拷贝 |
| 单行上限 | 4000 字符 | 超长截断并标注原长。多行块逐行截断，堆栈尾部不受影响 |
| 连续写盘失败 | 3 次 | 第 3 次直接结束进程，由 pm2 拉起 |

判据：

1. 日志文件写不进去，启动期就失败（`StartupConfigError`）。服务不会在没有日志的情况下跑起来。
2. 配置校验通过前，只有 `boot()` 的启动进度走 stdout；校验失败时那几行拒绝原因也走 stdout，因为此时还没有日志文件可写。
3. 堆栈走同一条落盘通道，不会再出现「文件里有一行『下方为代码堆栈』，堆栈却在 stderr」。
4. 日志文件持续不可写时结束进程，不在没有日志的状态下继续处理请求。

### 日志运维

| 事项 | 做法 |
|---|---|
| 找日志 | `logs/alignServer.log`，按时间追加，重启不清空 |
| 看实时 | `tail -f logs/alignServer.log` |
| 看某请求 | `grep <req_id> logs/alignServer.log`，请求的每一行都带同一个 `req_id` |
| 看当前积压 | `grep 任务 logs/alignServer.log \| tail -3` |
| 删旧轮转 | `ls -t logs/alignServer.log.* \| tail -n +6 \| xargs -r rm`，禁止通配删目录 |
| 磁盘占用上限 | 单文件 64 MB，含轮转最多 384 MB |

### 日志行形态

```
2026-11-23 08:33:09.859 [INFO ] [d143ceeb] 📥 接收请求 | 模式: ForcedAligner | 文件: req4.wav | 前面 0 个 | 任务 2 | 运行1/3 等待1 | 运行: 1dce2df1 ForcedAligner 0.0s | 等待: d143ceeb ForcedAligner 0s
```

| 段 | 含义 |
|---|---|
| `[INFO ]` | 级别，`INFO` / `WARN` / `ERROR` / `FATAL` |
| `[d143ceeb]` | `req_id`。同一个请求的所有日志都带同一个值 |
| `排在 N 个之后` | 比本请求更早入队、且仍在运行或等待的请求数。与「排队 Xs」同口径 |
| `任务 N \| 运行1/3 等待1` | 当前任务数；`/3` 是并行上限 |
| `冻结中` | Frozen World 生效期间出现。此时所有等待都是显存清理导致的挂起，不是算力排队 |
| `运行: …` / `等待: …` | 逐条明细：`req_id 模式 已持续时长 文件名`，各最多 12 条 |

一次请求至少有 4 行：接收、开始计算、结束。结束行把耗时拆成排队、计算、总计三段。

流水线自己的诊断行（`pipeline.py` / `cloud_hotword.py` 里的分块日志、最终报告）带同一个 `req_id`，所以一次请求的日志必然多于 4 行。按 `req_id` grep 才能读全。

参数校验不通过的请求也会打入口行和队列快照（此时它已登记），随后打 `⛔ 请求被拒绝（未进入计算区）` 并立刻标记失败。这类请求不进计算区，不占后续等待。

## 配置

`.env` 由 `autoInstall.sh` 生成，字段见脚本第 8 节。脚本**不覆盖已存在的 `.env`**：要按新的环境变量重写，先手动删除 `.env`。

已存在的 `.env` 不会被补字段。升级后若缺 `LOGGING_FILE`，服务会用默认值 `./logs/alignServer.log`；启动日志的「启动配置」一段会打出实际生效的路径。

所有字段都在启动期校验：取值不合法会**一次性报完所有问题**并拒绝启动，不会拖到某个请求中途才炸。

## 启动期检查

`main.py` 在加载模型前跑三层检查，任一项不过就打印全部问题并退出（exit 1）：

1. **配置层** — `PORT` 范围、`DEVICE` 枚举、模型路径形态、Python 版本下限、日志文件可写。
2. **依赖锁层** — 比对**已装包**的版本与 `uv.lock`。绕开锁跑过一次 `uv pip install <包名>` 会在这里被抓住，而不是几小时后某个请求上炸。
3. **硬件层** — 双向核对 `DEVICE` 与 torch 变体：`DEVICE=cuda` 但 CUDA 不可用（装成了 CPU 版），或 `DEVICE=cpu` 但 CUDA 可用（装反了），都会报错。

依赖锁层是**单向**比对：只查已装包的版本是否与锁一致，不查「锁里有但没装」。后者是安装问题，`uv sync --frozen` 保证它；真漏了会在 import 时报出明确的 `ModuleNotFoundError`，比启动期笼统说「缺 N 个包」好定位。

## 依赖版本纪律

**直接依赖一律写死 `==`，整棵依赖树由 `uv.lock` 钉死。装环境只允许 `uv sync --frozen`。**

裸跑 `uv pip install <包名>` 会绕过锁、静默改版本。历史上被 `silero-vad>=5.1.0` 实际装到 6.x 这类跨大版本变更坑过——6.x API 恰好兼容，纯属运气。

唯一的例外是 torch 生态：版本号仍来自锁，二进制变体由 `DEVICE` 决定，两步装法与复验步骤见上文「依赖与 torch 变体」。这个例外只存在于 `autoInstall.sh` 里，手工复现不算数。

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

### torch 变体的设计与踩坑

锁文件能钉版本，钉不了二进制变体——CPU 与 CUDA 构建版本号相同，`uv.lock` 表达不了「按硬件选变体」。

- **本锁文件是 GPU 基线**（默认 index 上的 CUDA 构建），是主部署形态。
- CPU 环境由安装脚本在冻结同步之后按锁里的版本号改装 CPU 构建，装完由启动期的硬件探针验证。

试过用 `[tool.uv.sources]` 的 marker 按 extra 路由两种变体，**实测失败**：uv 一旦为 torch 声明 source，lock 会把两个变体都解析到该 index，marker 失效。

三个已在 `autoInstall.sh` 里固化的坑：

1. **同步必须跳过 torch。** `uv sync --frozen` 默认会用锁里那份把已装的 torch 换掉。加 `--no-install-package torch --no-install-package torchaudio` 后，同步不碰 torch，顺序才能是「先同步其余依赖，再指认 torch 变体」。
2. **那次同步还必须带 `--inexact`。** 默认 exact 模式下，uv 把 `torch==2.14.1+cpu` 判为「与锁里的 2.14.1 不符」，会把它卸掉。
3. **CPU 路径必须 `--index-strategy unsafe-best-match` 加 `--extra-index-url`。** uv 默认只用第一个含该包的 index，PyPI/清华源上的 torch 没有 `+cpu` 后缀，只传前者会把 PyTorch CPU index 上的变体挡在候选之外，报 `No solution found`。

**装完 torch 之后不要再跑同步。** 加了跳过参数才是安全的；不带跳过参数的复验正是把 CPU 变体抹掉的那一步，所以脚本里没有这一步，改用 torch 版本与 `DEVICE` 的自洽断言代替。

### 关于 Flash Attention 2

GPU 上的可选加速。`pipeline.py` 有 `except ImportError` 降级到 `sdpa`，装不上不影响可用性，故不进锁文件。

上游没有与 torch 2.14 + CUDA 13 匹配的预编译 wheel。需要时装：

```bash
INSTALL_FLASH_ATTN=1 DEVICE=cuda ./autoInstall.sh
```

或手工：

```bash
uv pip install flash-attn --no-build-isolation
```

无预编译 wheel 时会退化成源码编译，耗时较长。安装失败**不中断**安装流程。

## 已知限制：epitran 的 IPA 映射表

对齐模式用 epitran 把原文转成 IPA，再做音标寻路。各语言的映射表实际可用性不一致。

`config.toml` 为 `Qwen/Qwen3-ASR-0.6B + Qwen/Qwen3-ForcedAligner-0.6B` 声明了 12 种语言。实测（2026-11-23，本机 `flite 2.2-6build3`）：

| 语言 | epitran 映射表 | 状态 |
|---|---|---|
| 中文简/繁 | `cmn-Hans` / `cmn-Hant` | 可用 |
| 韩语 | `kor-Hang` | 可用 |
| 法/德/意/葡/俄/西 | 各 `Latn` / `Cyrl` | 可用 |
| 英语 | `eng-Latn` | **不可用**，缺 `lex_lookup` |
| 日语 | `jpn-Latn` | **不可用**，epitran 无此映射表文件 |

两个不可用的成因不同：

- **英语**：epitran 的英文 g2p 走 flite 的 `lex_lookup` 工具。Ubuntu 的 `flite` 包只提供 `t2p`，不含 `lex_lookup`；后者要在从源码编译 flite 时才生成。
- **日语**：epitran 自带 maps 目录里没有 `jpn-Latn`，与 flite 无关，装任何系统包都修不好。

结论：这 12 种语言里，本地对齐实际可用 10 种。英语与日语需另行处理（走云端热词分支，或换 IPA 方案）。

英语这一条是**功能边界，不是安装问题**：`autoInstall.sh` 装上系统 flite 已经算成功，不要再往「装别的 flite 包」方向排查。`autoInstall.sh` 的 flite 安装块顶部有同一条注释，改安装逻辑前先读它。

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

## 纯透传上限

仅对齐模式不切块。音频时长上限取模型权重本身：输出头 `classify_num=5000`，每档 `timestamp_segment_time=80ms`，故时间轴最大可表达长度为 `5000 × 0.08 = 400s`。超过即返回 400 `invalid_request_error`。

这个数字不是文档里的「5 分钟」口径，而是分类头能给出的档位数量。超过 400s 时 argmax 找不到合法档位，时间戳必然失真，故直接拒绝。

调用方需自行按 400s 切分音频与文本。切分边界归调用方所有，服务端不再推断。

## 错误响应

除 200 的对齐结果外，本服务只用两种失败形态。调用方按 `detail.error.type` 分派，不读消息文本。

| `type` | HTTP | 含义 | 调用方应有动作 |
|---|---|---|---|
| `invalid_request_error` | 400 | 请求参数问题（模型名、语言、缺 `prompt`、缺省校验） | 改请求。原样重发必然同样被拒 |
| `audio_defect` | 400 | 这份音频与参考文本不符（当前为整章有效字缺失段超过单段容差） | 作废本章产物并重生成音频，不要原样重发 |
| `alignment_threshold_error` | 400 | 首尾块对齐失败（来自对齐模式与云端热词模式；仅对齐是纯透传，不抛此错） | 查音频与参考文本。原样重发无意义 |
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
├── async_logging.py           异步日志：入队、批量落盘、轮转
├── conftest.py                让 test/ 下的测试能导入项目根模块
├── autoInstall.sh             安装（非交互，唯一安装入口）
├── start_alignServer.sh       启动（唯一启动入口，pm2 托管）
├── pyproject.toml             依赖声明（钉版本）
├── uv.lock                    整棵依赖树的精确版本
└── test/
    ├── test_async_logging.py           异步日志的落盘契约与日志文件校验
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
