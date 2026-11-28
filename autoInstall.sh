#!/bin/bash
# 遇到错误立即退出
set -e

# ==========================================
# 唯一安装入口。
# 生产环境只跑本脚本与 start_alignServer.sh，禁止手工拼环境。
# 设计判据：
#   1. 依赖一律来自 uv.lock（uv sync --frozen），不是 -r pyproject.toml。
#   2. torch 的 CPU/CUDA 变体锁文件表达不了，故在冻结同步之后单独装，
#      装完由启动期硬件探针验证（env_config.check_device_capability）。
#   3. 全部选择走环境变量，非交互可重跑。缺必填项立即失败，不猜默认值。
# ==========================================

echo "🚀 开始构建 Long Audio Aligner 运行环境..."

# ==========================================
# 0. 全局配置（全部走环境变量，非交互）
# ==========================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MyWorkDir="${MyWorkDir:-$SCRIPT_DIR}"
export MODEL_ROOT_DIR="${MODEL_ROOT_DIR:-$HOME/autodl-fs/models}"

# flite 存在性只在这里检查一次。提前告警，不提前退出：flite 可能由 apt 装依赖时带入。
# 真的缺 flite 由脚本末尾安装并回验，安装不上在那里 exit。
FLITE_PRESENT_AT_START=0
if command -v flite >/dev/null 2>&1; then
    FLITE_PRESENT_AT_START=1
    echo "✅ 检测到 flite：$(command -v flite)"
else
    echo "⚠️ 未检测到 flite（epitran 的系统依赖）。脚本末尾会尝试用 apt 安装 flite；"
    echo "   若当前无安装权限，脚本会在末尾失败。提前装：apt-get install -y flite"
fi

# DEVICE 决定 torch 变体。取值不合法立即退出，不回落默认值。
DEVICE="${DEVICE:-cuda}"
case "$DEVICE" in
    cpu|cuda) ;;
    *)
        echo "[FATAL] DEVICE=$DEVICE 不合法，只能是 cpu 或 cuda" >&2
        exit 1
        ;;
esac

# 国内镜像开关。y = 启用清华镜像 / n = 官方源
use_domestic="${USE_DOMESTIC:-y}"

# 模型下载方式：hf = HuggingFace / ms = 魔搭
DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE:-hf}"
case "$DOWNLOAD_SOURCE" in
    hf|ms) ;;
    *)
        echo "[FATAL] DOWNLOAD_SOURCE=$DOWNLOAD_SOURCE 不合法，只能是 hf 或 ms" >&2
        exit 1
        ;;
esac

# Hugging Face 令牌只从环境读取。Qwen 公开模型匿名可下，缺失不中断。
export HF_TOKEN="${HF_TOKEN:-}"
if [ -n "$HF_TOKEN" ]; then
    echo "✅ 已从环境变量读取 HF_TOKEN（${#HF_TOKEN} 位）"
else
    echo "⚠️ HF_TOKEN 未设置：匿名下载可能触发 HF 限流（公开模型可忽略）"
fi

# 百炼 Key。main.py 启动期无条件校验，缺失即拒绝启动，故这里必须拿到。
DASHSCOPE_APIKEY="${DASHSCOPE_APIKEY:-${dashscope_apikey:-}}"

echo "  工作目录 : $MyWorkDir"
echo "  模型根目录: $MODEL_ROOT_DIR"
echo "  设备     : $DEVICE"
echo "  下载源   : $DOWNLOAD_SOURCE"
echo "  国内加速 : $use_domestic"

mkdir -p "$MyWorkDir"
mkdir -p "$MODEL_ROOT_DIR"
cd "$MyWorkDir"

# 国内加速：uv.lock 的 wheel URL 已钉在清华镜像，核对时跟随 USE_DOMESTIC。
# 注意 uv.lock 的包源写死在 URL 里，装什么源由锁决定，不由这里决定。
if [[ "$use_domestic" == "y" || "$use_domestic" == "Y" ]]; then
    export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
    echo "⚡ 已启用国内加速（清华镜像）"
else
    export UV_DEFAULT_INDEX="https://pypi.org/simple"
    echo "🌐 直连官方源"
fi

# qwen-asr 的 commit 钉法只在 pyproject.toml 声明一次，本脚本不重复写死。

# ==========================================
# 1. 前置工具检查
# ==========================================
echo ""
if ! command -v uv &> /dev/null; then
    echo "[FATAL] 未检测到 uv。请先安装：python3 -m pip install uv" >&2
    exit 1
fi
echo "✅ 检测到 uv 已安装，跳过此步骤。"

# uv 缓存目录。默认 ~/.cache/uv 在只读挂载下不可写，允许用 UV_CACHE_DIR 覆盖。
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"

# ==========================================
# 2. 创建并激活虚拟环境
# ==========================================
echo ""
if [ -d ".venv" ]; then
    echo "♻️ 检测到已存在 .venv 文件夹，准备复用当前虚拟环境..."
else
    echo "[*] 正在使用 uv 创建纯净虚拟环境 .venv..."
    uv venv .venv -p 3.12 --allow-existing
fi
source .venv/bin/activate

# ==========================================
# 3. 按 uv.lock 冻结安装全部依赖
# ==========================================
echo ""
echo "[*] 正在执行 uv sync --frozen（依赖一律来自 uv.lock）..."

# torch / torchaudio 的二进制变体（CPU 与 CUDA 版本号相同）uv.lock 表达不了，
# 故先同步其余全部依赖，再按 DEVICE 单独装 torch 生态，最后用 --no-deps 复验。
# uv sync --frozen 会因为 torch 变体版本号不等而把它重装回锁里的 CUDA 构建，
# 这一步的顺序不能颠倒：先 frozen，再指认 torch。
# 复验：--inexact 让 uv 只补缺、不卸载已装的多余包。
# 不能用默认的 exact 模式：它会把 torch 2.14.1+cpu 当成「与锁不符」卸掉，
# 下一行装好的 CPU 变体又被抹掉，形成「装好即卸载」的死循环。
uv sync --frozen --inexact --no-install-package torch --no-install-package torchaudio

# qwen-asr 在锁里是 URL 依赖，uv sync --frozen 已按锁装好；此处不重装。

# ==========================================
# 4. 按 DEVICE 装 torch 生态（锁钉版本号，index 决定二进制变体）
# ==========================================
echo ""

# 版本号取自 pyproject.toml，与 uv.lock 同源；只从 index 决定 CPU / CUDA 变体。
TORCH_PIN="$(python -c "
import tomllib
deps = tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']
for d in deps:
    if d.startswith('torch=='):
        print(d.split('==',1)[1])
" )"
TORCHAUDIO_PIN="$(python -c "
import tomllib
deps = tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']
for d in deps:
    if d.startswith('torchaudio=='):
        print(d.split('==',1)[1])
" )"

if [ -z "$TORCH_PIN" ] || [ -z "$TORCHAUDIO_PIN" ]; then
    echo "[FATAL] pyproject.toml 里找不到 torch== / torchaudio== 的钉版本声明" >&2
    exit 1
fi
echo "  锁钉版本: torch=$TORCH_PIN torchaudio=$TORCHAUDIO_PIN"

# CUDA 变体的 index 必须与 uv.lock 里 torch 的 CUDA 依赖同源。
# 锁里 torch 2.14.1 依赖 cuda-toolkit 13 / nvidia-cudnn-cu13 / nccl-cu13，
# 对应 whl/cu130。whl/cu121 上最高只有 torch 2.5.1，取 2.14.1 必然 No solution。
CUDA_INDEX="https://download.pytorch.org/whl/cu130"
CPU_INDEX="https://download.pytorch.org/whl/cpu"

if [[ "$DEVICE" == "cuda" ]]; then
    echo "  -> 正在安装 GPU (CUDA 13.0) 版本的 Torch 生态..."
    uv pip install "torch==${TORCH_PIN}" "torchaudio==${TORCHAUDIO_PIN}" \
        --index-url "$CUDA_INDEX"
else
    echo "  -> 正在安装 CPU 版本的 Torch 生态..."
    # 必须显式 --index-strategy unsafe-best-match：uv 默认只用「第一个含该包的 index」，
    # 镜像源上的 torch 没有 +cpu 后缀，会挡住 PyTorch CPU index 上的变体。
    uv pip install "torch==${TORCH_PIN}+cpu" "torchaudio==${TORCHAUDIO_PIN}+cpu" \
        --index-strategy unsafe-best-match \
        --extra-index-url "$CPU_INDEX" \
        --index-url "$UV_DEFAULT_INDEX"
fi

# 断言 torch 变体真的对上了 DEVICE。这是唯一一处校验二进制变体的地方：
# uv.lock 钉不住变体（版本号相同），启动期探针只查「CUDA 可用性是否与 DEVICE 自洽」，
# 查不出「DEVICE=cuda 但装的是不含 CUDA 的 whl」以外的形态。
python -c "
import torch, sys
device = '${DEVICE}'
has_cuda = torch.cuda.is_available()
want_cuda = device == 'cuda'
if want_cuda != has_cuda:
    sys.exit(f'[FATAL] torch 变体与 DEVICE 不符: DEVICE={device} 但 cuda_available={has_cuda}')
print(f'  ✅ torch {torch.__version__} | cuda_available={has_cuda}')
"

# ==========================================
# 4.5 Flash Attention 2（仅 GPU，可选）
# ==========================================
# 它不是必需项：pipeline.py 有 `except ImportError` 落到 sdpa，功能不受影响。
# 故整节失败只告警、不中断。安装由用户显式打开（INSTALL_FLASH_ATTN=1），
# 默认关闭——上游没有与 torch 2.14 + CUDA 13 匹配的预编译 wheel。
if [[ "$DEVICE" == "cuda" && "${INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
    echo ""
    echo "[*] 正在为 GPU 环境安装 Flash Attention 2..."
    # 无预编译 wheel 时会退化为源码编译，耗时较长。
    # --no-build-isolation 是上游要求，否则会在隔离环境里重新解析 torch。
    if uv pip install flash-attn --no-build-isolation; then
        echo "    ✅ Flash Attention 2 安装成功"
    else
        echo "    ⚠️ Flash Attention 2 安装失败，pipeline 将回退到 sdpa（功能不受影响）"
    fi
fi

# ==========================================
# 5. Qwen3-ASR 核心包
# ==========================================
# 它已在 uv.lock 里作为 URL 依赖钉到某个 commit，uv sync --frozen 已装。
# 此处只校验装到的确实存在，不在脚本里重新装一遍（重装会绕过锁）。
# sha 不在本脚本重复写死：唯一声明处是 pyproject.toml 的 URL 依赖。
echo ""
echo "[*] 校验 Qwen3-ASR 核心包..."
if ! python -c "import qwen_asr" 2>/dev/null; then
    LOCKED_QWEN_URL="$(python -c "
import tomllib
deps = tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']
print(next((d for d in deps if d.startswith('qwen-asr @')), '(未声明)'))
")"
    echo "[FATAL] qwen_asr 未安装。uv sync --frozen 未覆盖它，请检查 uv.lock 里的 URL 依赖。" >&2
    echo "        pyproject.toml 的声明：$LOCKED_QWEN_URL" >&2
    exit 1
fi
echo "    ✅ qwen_asr 已就绪"

# ==========================================
# 6. 本地化下载大模型 (Qwen) - 多选版
# ==========================================
echo ""
mkdir -p "$MODEL_ROOT_DIR/Qwen"

# 清理可能导致网络绕路的代理环境变量
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

# 定义下载逻辑的函数以减少代码冗余
download_model() {
    local repo_id=$1
    local local_dir=$2
    
    if [ -d "$local_dir" ] && [ "$(ls -A $local_dir 2>/dev/null)" ]; then
        echo "    ✅ 检测到模型 $repo_id 已存在于 $local_dir，跳过下载..."
    else
        echo "    -> 开始下载 $repo_id..."
        if [ "$DOWNLOAD_SOURCE" == "ms" ]; then
            modelscope download --model="$repo_id" --local-dir "$local_dir"
        else
            # 兼容别名及标准工具
            if command -v hf &> /dev/null; then
                hf download "$repo_id" --local-dir "$local_dir"
            else
                HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "$repo_id" --local-dir "$local_dir"
            fi
        fi
    fi
}

if [ "$DOWNLOAD_SOURCE" == "ms" ]; then
    echo "[*] 正在使用 ModelScope 魔搭社区进行下载..."
elif [[ "$use_domestic" == "y" || "$use_domestic" == "Y" ]]; then
    echo "[*] 已切换至 HF-Mirror 国内镜像源，准备满速拉取模型..."
    export HF_ENDPOINT=https://hf-mirror.com
else
    echo "[*] 正在从 Hugging Face 官方直连下载模型..."
fi

# 开始下载 Qwen3-ASR 及其强制对齐模型 (MS与HF仓库名一致)
download_model "Qwen/Qwen3-ASR-0.6B" "$MODEL_ROOT_DIR/Qwen/Qwen3-ASR-0.6B"
download_model "Qwen/Qwen3-ForcedAligner-0.6B" "$MODEL_ROOT_DIR/Qwen/Qwen3-ForcedAligner-0.6B"

# ==========================================
# 7. 预热并下载 VAD 小模型 (优雅的包触发)
# ==========================================
echo ""
echo "[*] 正在预先下载并缓存 Silero VAD 模型..."
# silero-vad 已在 uv.lock 里，uv sync --frozen 已装，此处只用它触发权重缓存。
python -c "
from silero_vad import load_silero_vad
load_silero_vad()
"
echo "✅ Silero VAD 模型已预热下载完毕！"

# ==========================================
# 8. 生成 Server 配置文件 
# ==========================================
echo ""
echo "[*] 开始配置 Web 服务信息 (默认绑定 127.0.0.1:6008)..."

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-6008}"
# 日志文件写在项目目录下，与 .env 同级。相对路径由 main.py 按自身目录解析。
LOGGING_FILE_VALUE="${LOGGING_FILE:-./logs/alignServer.log}"

# cloud ASR 分支（qwen-audio-3.1-asr-flash-filetrans）所需；main.py 启动期强校验，缺失即拒绝启动
if [ -z "$DASHSCOPE_APIKEY" ]; then
    echo "[FATAL] 缺少百炼 API Key。main.py 启动期无条件校验，缺失即拒绝启动。" >&2
    echo "        请在运行前 export DASHSCOPE_APIKEY=sk-... 或 export dashscope_apikey=sk-..." >&2
    exit 1
fi

echo "[*] 正在生成 .env 环境变量配置文件..."

# 已存在的 .env 不覆盖：运维手工改过的 HOST / PORT / Key 必须保留。
# 要按这次的环境变量重写，先手动删掉 .env 再跑本脚本。
if [ -f ".env" ]; then
    echo "⚠️ .env 已存在，跳过生成。如需按本次环境变量重写，请先删除 .env。"
else
    cat <<EOF > .env
# 硬件加速模式
DEVICE=$DEVICE

# 本地化模型绝对路径
ASR_MODEL_PATH=$MODEL_ROOT_DIR/Qwen/Qwen3-ASR-0.6B
FA_MODEL_PATH=$MODEL_ROOT_DIR/Qwen/Qwen3-ForcedAligner-0.6B

# 百炼 API Key（cloud ASR 分支必需，启动时校验）
dashscope_apikey=$DASHSCOPE_APIKEY

# FastAPI 服务配置
HOST=$SERVER_HOST
PORT=$SERVER_PORT

# 本地日志文件（相对路径按 main.py 所在目录解析）。启动期会实测可写，写不进去即拒绝启动
LOGGING_FILE=${LOGGING_FILE_VALUE:-./logs/alignServer.log}
EOF
    echo "    ✅ 已写入 .env"
fi

# swapoff 需要 root 权限。非 root 下失败不应中断安装。
echo "正在尝试关闭 swap 内存交互，避免数据从内存卸载到硬盘..."
swapoff -a || true

# flite 是 epitran 的系统依赖（epitran 是 pyproject 的硬依赖），由 apt 提供。
# 缺失时对齐模式不可用，故这里装不上就退出，不把问题留到运行时。
#
# 提权方式按当前身份决定，不写死 sudo：AutoDL 等 root 容器镜像不带 sudo，
# 写死 sudo 时这些镜像必然 `sudo: command not found`。
# 已存在的 flite 一律直接复用，不因装上的是旧版就重装：硬下限只要求命令存在。
#
# 注意英语按设计不可用：Ubuntu 的 flite 包只提供 t2p，不含 lex_lookup，
# 而 epitran 的 eng-Latn 映射表依赖 lex_lookup。装了 flite 也跑不了英语。
# 这是功能边界，不是安装问题，本脚本不修。详见 README「已知限制：epitran 的 IPA 映射表」。
if command -v flite >/dev/null 2>&1; then
    echo "✅ flite 已存在：$(command -v flite)"
else
    if [ "$FLITE_PRESENT_AT_START" -eq 1 ]; then
        echo "⚠️ flite 在启动时存在，现在找不到了。PATH 或文件可能被改动。"
    fi
    echo "[*] 正在安装 flite（epitran 的系统依赖）..."
    # 精简镜像可能连 apt-get 都没有。先查它，让失败信息停在这里，
    # 而不是退回到要从 `apt-get: command not found` 反推。
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "[FATAL] flite 缺失，且系统无 apt-get（非 Debian/Ubuntu 系，或精简镜像）。" >&2
        echo "        请用本发行版的包管理器手工安装 flite 后重跑本脚本。" >&2
        exit 1
    fi
    if [ "$(id -u)" -eq 0 ]; then
        echo "    当前身份是 root，直接调用 apt-get"
        apt-get install -y flite
    elif command -v sudo >/dev/null 2>&1; then
        echo "    当前身份非 root，经 sudo 调用 apt-get"
        sudo apt-get install -y flite
    else
        echo "[FATAL] flite 缺失，且当前身份非 root、系统无 sudo，无法自动安装。" >&2
        echo "        请用 root 执行：apt-get install -y flite" >&2
        echo "        或先安装 sudo，再重跑本脚本。" >&2
        exit 1
    fi
    if ! command -v flite >/dev/null 2>&1; then
        echo "[FATAL] flite 安装后 PATH 中仍找不到。epitran 无法工作。" >&2
        exit 1
    fi
    echo "✅ flite 安装完成：$(command -v flite)"
fi

echo ""
echo "✅ 环境配置与所有模型（Qwen & VAD）全量下载完毕！"
echo ""
echo "下一步：./start_alignServer.sh"
echo "本脚本不起服务。启动的唯一入口是 start_alignServer.sh。"