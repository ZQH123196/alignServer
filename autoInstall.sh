#!/bin/bash
# 遇到错误立即退出
set -e

echo "🚀 开始构建 Long Audio Aligner 运行环境..."
# 如果系统存在 .bashrc 则加载，避免遗漏已有环境变量
[ -f ~/.bashrc ] && source ~/.bashrc

# ==========================================
# 0. 全局网络环境检测 (最先确认，影响全局策略)
# ==========================================
echo ""
read -p "❓ 当前是否处于国内网络环境? (y: 启用镜像加速与 HF-Mirror / n: 默认直连) [默认: y]: " use_domestic
use_domestic=${use_domestic:-y}

# 预设 PyPI 镜像、GitHub 代理和源码 ZIP 下载链接
PYPI_MIRROR=""
GH_PROXY=""
QWEN_ASR_ZIP="https://github.com/QwenLM/Qwen3-ASR/archive/refs/heads/main.zip"

if [[ "$use_domestic" == "y" || "$use_domestic" == "Y" ]]; then
    echo "⚡ 已开启国内加速模式"
    echo "请选择要使用的 PyPI 镜像源 (输入数字序号):"
    echo "  1) 清华大学 (默认)"
    echo "  2) 阿里云"
    echo "  3) 腾讯云"
    echo "  4) 中科大"
    echo "  5) 华为云"
    read -p "请输入序号 [默认: 1]: " mirror_choice
    mirror_choice=${mirror_choice:-1}

    case $mirror_choice in
        2) 
            PYPI_MIRROR="-i https://mirrors.aliyun.com/pypi/simple/" 
            echo "✅ 已选择: 阿里云镜像源"
            ;;
        3) 
            PYPI_MIRROR="-i https://mirrors.cloud.tencent.com/pypi/simple/" 
            echo "✅ 已选择: 腾讯云镜像源"
            ;;
        4) 
            PYPI_MIRROR="-i https://pypi.mirrors.ustc.edu.cn/simple/" 
            echo "✅ 已选择: 中科大镜像源"
            ;;
        5) 
            PYPI_MIRROR="-i https://repo.huaweicloud.com/repository/pypi/simple/" 
            echo "✅ 已选择: 华为云镜像源"
            ;;
        *) 
            PYPI_MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple" 
            echo "✅ 已选择: 清华大学镜像源 (默认)"
            ;;
    esac

    GH_PROXY="https://ghproxy.net/"
    # 拼接国内代理 ZIP 链接，体积最小，速度最快
    QWEN_ASR_ZIP="${GH_PROXY}${QWEN_ASR_ZIP}"
else
    echo "🌐 已开启海外直连模式"
fi

# ==========================================
# 1. 前置工具检查与交互式安装
# ==========================================
echo ""
if ! command -v uv &> /dev/null; then
    echo "⚠️  [警告] 未检测到包管理工具 uv。"
    read -p "❓ 是否允许脚本自动为您安装 uv? (y: 使用 pip 安装 / n: 退出并手动安装) [默认: y]: " install_uv
    install_uv=${install_uv:-y}
    
    if [[ "$install_uv" == "y" || "$install_uv" == "Y" ]]; then
        echo "[*] 正在执行安装: python -m pip install uv $PYPI_MIRROR ..."
        # 使用 python -m pip 避免多环境 pip 冲突
        python3 -m pip install uv $PYPI_MIRROR || pip install uv $PYPI_MIRROR
        
        # 尝试刷新环境变量
        [ -f ~/.bashrc ] && source ~/.bashrc
        if ! command -v uv &> /dev/null; then
            echo "[FATAL ERROR] uv 安装完成，但在当前环境变量 PATH 中未找到。可能安装到了用户目录（如 ~/.local/bin），请将其加入 PATH 或重启终端后再运行。"
            exit 1
        fi
        echo "✅ uv 安装成功！"
    else
        echo "[INFO] 您取消了自动安装。请手动安装 uv 后再运行此脚本。"
        exit 1
    fi
else
    echo "✅ 检测到 uv 已安装，跳过此步骤。"
fi

# ==========================================
# 2. 创建并激活虚拟环境 (修改为支持复用)
# ==========================================
echo ""
if [ -d ".venv" ]; then
    echo "♻️ 检测到已存在 .venv 文件夹，准备复用当前虚拟环境..."
else
    echo "[*] 正在使用 uv 创建纯净虚拟环境 .venv..."
    uv venv
fi
source .venv/bin/activate 

# ==========================================
# 3. 硬件环境与依赖安装
# ==========================================
echo ""
read -p "❓ 当前机器是否支持并需使用 NVIDIA GPU (CUDA)? (y: 使用 GPU / n: 使用 CPU) [默认: y]: " use_gpu
use_gpu=${use_gpu:-y}

INSTALL_TARGET="."
INDEX_URL=""

if [[ "$use_gpu" == "y" || "$use_gpu" == "Y" ]]; then
    DEVICE_ENV="cuda"
    INDEX_URL="--extra-index-url https://download.pytorch.org/whl/cu121"
    # 现在统一走 HF 生态，不再区分 ms 或 hf
    INSTALL_TARGET=".[GPU,hf]"
else
    DEVICE_ENV="cpu"
    INDEX_URL="--extra-index-url https://download.pytorch.org/whl/cpu"
    INSTALL_TARGET=".[CPU,hf]"
fi

# ==========================================
# 4. 严格按照 pyproject.toml 声明安装基础依赖
# ==========================================
echo ""
echo "[*] 正在执行规范安装: uv pip install -e \"$INSTALL_TARGET\" $PYPI_MIRROR $INDEX_URL ..."
# uv pip install 是幂等的，如果环境里已经有了符合要求的包，它会极速跳过
uv pip install -e "$INSTALL_TARGET" $PYPI_MIRROR $INDEX_URL

# ==========================================
# 4.5 独立安装 Flash Attention 2 (仅限 GPU)
# ==========================================
if [[ "$use_gpu" == "y" || "$use_gpu" == "Y" ]]; then
    echo ""
    echo "[*] 正在为 GPU 环境安装 Flash Attention 2..."
    echo "    -> 准备拉取预构建包 (Wheel)"

    # 如果是国内环境，尝试开启终端代理下载 GitHub 资源
    if [[ "$use_domestic" == "y" || "$use_domestic" == "Y" ]]; then
        if [ -f /etc/network_turbo ]; then
            echo "    -> 🌐 检测到国内环境，开启系统代理 (/etc/network_turbo)..."
            source /etc/network_turbo
        else
            echo "    -> ⚠️ 未检测到代理配置文件，尝试直连下载..."
        fi
    fi

    # 下载并安装
    wget -nc https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
    uv pip install flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

    # 安装完成后，如果是国内环境，关闭代理以免影响后续国内源（如 HF-Mirror/清华源）的访问
    if [[ "$use_domestic" == "y" || "$use_domestic" == "Y" ]]; then
        if [ -f /etc/network_turbo ]; then
            echo "    -> 🛑 关闭系统代理..."
            unset http_proxy && unset https_proxy
        fi
    fi
fi

# ==========================================
# 5. 自动补全 Qwen3-ASR 源码模块 (极速 ZIP 版)
# ==========================================
echo ""
echo "[*] 正在极速拉取并安装 Qwen3-ASR 核心包与底层依赖..."
uv pip install "$QWEN_ASR_ZIP" $PYPI_MIRROR $INDEX_URL

# ==========================================
# 6. 本地化下载大模型 (Qwen) - HF-Mirror 版
# ==========================================
echo ""
mkdir -p ./models/Qwen3-ASR-0.6B
mkdir -p ./models/Qwen3-ForcedAligner-0.6B

# 清理可能导致网络绕路的代理环境变量
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

echo "[*] 正在更新并安装官方 Hugging Face CLI 及加速模块..."
# 使用 pip 升级/安装 huggingface_hub，同时安装 hf-transfer 以开启多线程加速
pip install --upgrade huggingface_hub hf-transfer $PYPI_MIRROR

# 开启底层 hf-transfer 极速下载机制
export HF_HUB_ENABLE_HF_TRANSFER=1

if [[ "$use_domestic" == "y" || "$use_domestic" == "Y" ]]; then
    echo "[*] 已切换至 HF-Mirror 国内镜像源，准备满速拉取模型..."
    export HF_ENDPOINT=https://hf-mirror.com
else
    echo "[*] 正在从 Hugging Face 官方直连下载模型..."
fi

# 使用官方标准的 hf 命令执行下载
echo "    -> 开始下载 Qwen3-ASR-0.6B..."
hf download Qwen/Qwen3-ASR-0.6B --local-dir ./models/Qwen3-ASR-0.6B

echo "    -> 开始下载 Qwen3-ForcedAligner-0.6B..."
hf download Qwen/Qwen3-ForcedAligner-0.6B --local-dir ./models/Qwen3-ForcedAligner-0.6B

# ==========================================
# 7. 预热并下载 VAD 小模型 (解决 PyTorch Hub 交互式卡顿)
# ==========================================
echo ""
echo "[*] 正在预先下载并信任 Silero VAD 模型..."
# 使用 yes 命令自动向 Python 脚本输入 'y'，解决下载时询问 "Do you trust this repository" 的停顿问题
yes y | python -c "
import torch
try:
    print('  -> 正在通过 PyTorch Hub 拉取 Silero VAD...')
    torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True, verbose=False)
except TypeError:
    torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', verbose=False)
" > /dev/null 2>&1
echo "✅ Silero VAD 模型已预热并下载完毕！"

# ==========================================
# 8. 生成 Server 配置文件 (已固定默认值)
# ==========================================
echo ""
echo "[*] 开始配置 Web 服务信息 (默认绑定 127.0.0.1:6006)..."

SERVER_HOST="127.0.0.1"
SERVER_PORT="6006"

echo "[*] 正在生成 .env 环境变量配置文件..."

PROJECT_ROOT=$(pwd)

cat <<EOF > .env
# 硬件加速模式
DEVICE=$DEVICE_ENV

# VAD 模型配置
VAD_REPO=snakers4/silero-vad
VAD_MODEL_NAME=silero_vad

# 本地化模型绝对路径
ASR_MODEL_PATH=$PROJECT_ROOT/models/Qwen3-ASR-0.6B
FA_MODEL_PATH=$PROJECT_ROOT/models/Qwen3-ForcedAligner-0.6B

# FastAPI 服务配置
HOST=$SERVER_HOST
PORT=$SERVER_PORT
EOF

# 注意：swapoff 需要 root 权限，如果在非 root 环境执行会报错导致脚本退出（因为上方开启了 set -e）
# 如果用户没有 root 权限，可以加上 || true 防止脚本意外中断
echo "正在尝试关闭 swap 内存交互，避免数据从内存卸载到硬盘..."
sudo swapoff -a || true 

echo ""
echo "✅ 环境配置与所有模型（Qwen & VAD）全量下载完毕！"
echo "🚀 正在启动服务..."
source .venv/bin/activate && python main.py