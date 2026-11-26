#!/bin/bash
# 遇到错误立即退出
set -e

echo "🚀 开始构建 Long Audio Aligner 运行环境..."

# ==========================================
# 0. 全局配置与网络环境检测 (最先确认，影响全局策略)
# ==========================================
echo ""
export MyWorkDir="/root/autodl-tmp/alignServer"
export HF_TOKEN="REDACTED_HF_TOKEN"
export MODEL_ROOT_DIR="/root/autodl-fs/models"

# 创建目录
mkdir -p "$MyWorkDir"
mkdir -p "$MODEL_ROOT_DIR"
cd "$MyWorkDir"

read -p "❓ 当前是否处于国内网络环境? (y: 启用镜像加速 / n: 默认直连) [默认: y]: " use_domestic
use_domestic=${use_domestic:-y}

# 预设 PyPI 镜像、GitHub 代理和源码 ZIP 下载链接
PYPI_MIRROR=""
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

    QWEN_ASR_ZIP="${QWEN_ASR_ZIP}"
else
    echo "🌐 已开启海外直连模式"
fi

echo ""
echo "📦 请选择大模型下载方式 (输入数字序号):"
echo "  1) Hugging Face 官方直连"
echo "  2) Hugging Face 国内镜像 (HF-Mirror) [默认]"
echo "  3) ModelScope 魔搭社区"
read -p "请输入序号 [默认: 2]: " dl_choice
dl_choice=${dl_choice:-2}

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
        [ -f $HOME/.bashrc ] && source $HOME/.bashrc
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
    uv venv .venv -p 3.12 --allow-existing
fi
source .venv/bin/activate 

# ==========================================
# 3. 硬件环境与依赖安装
# ==========================================
echo ""
read -p "❓ 当前机器是否支持并需使用 NVIDIA GPU (CUDA)? (y: 使用 GPU / n: 使用 CPU) [默认: y]: " use_gpu
use_gpu=${use_gpu:-y}

INDEX_URL=""
EXTRA_ARGS="" # 改用 EXTRA_ARGS 传参

if [[ "$use_gpu" == "y" || "$use_gpu" == "Y" ]]; then
    DEVICE_ENV="cuda"
    INDEX_URL="--extra-index-url https://download.pytorch.org/whl/cu121"
    EXTRA_ARGS="--extra GPU --extra hf"
else
    DEVICE_ENV="cpu"
    INDEX_URL="--extra-index-url https://download.pytorch.org/whl/cpu"
    EXTRA_ARGS="--extra CPU --extra hf"
fi

# ==========================================
# 4. 严格按照 pyproject.toml 声明安装基础依赖
# ==========================================
echo ""
echo "[*] 正在执行规范安装..."

if [[ "$use_gpu" == "y" || "$use_gpu" == "Y" ]]; then
    # 💡 GPU 模式下，强行直接指定从 PyTorch 官方源下载对应的 CUDA 12.1 版本，防止被国内镜像源的 CPU 版覆盖
    echo "  -> 正在强制锁定安装 GPU (CUDA 12.1) 版本的 Torch 生态..."
    uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "  -> 正在安装 CPU 版本的 Torch 生态..."
    uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 安装剩余的常规依赖（如 fastapi, silero-vad 等）
echo "  -> 正在安装 pyproject.toml 中的其他配套依赖..."
uv pip install -r pyproject.toml $EXTRA_ARGS $PYPI_MIRROR

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
    FLASH_ATTN_WHL="flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
    FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/${FLASH_ATTN_WHL}"

    if [ -f "$FLASH_ATTN_WHL" ]; then
        echo "    -> ✅ 检测到 ${FLASH_ATTN_WHL} 已存在，跳过下载..."
    else
        echo "    -> 准备下载 ${FLASH_ATTN_WHL}..."

        # 自动安装 aria2c
        if ! command -v aria2c &> /dev/null; then
            echo "       ⚠️ 未检测到 aria2c，自动安装中..."
            apt-get install -y aria2
        fi

        echo "       使用 aria2c (多连接加速)..."
        aria2c -x 16 -s 16 -k 1M \
               --auto-file-renaming=false \
               --console-log-level=warn \
               -o "$FLASH_ATTN_WHL" "$FLASH_ATTN_URL"
    fi
    uv pip install "$FLASH_ATTN_WHL"

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
        if [ "$dl_choice" == "3" ]; then
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

if [ "$dl_choice" == "3" ]; then
    echo "[*] 正在使用 ModelScope 魔搭社区进行下载..."
    pip install modelscope $PYPI_MIRROR
elif [ "$dl_choice" == "2" ]; then
    echo "[*] 已切换至 HF-Mirror 国内镜像源，准备满速拉取模型..."
    export HF_ENDPOINT=https://hf-mirror.com
    pip install --upgrade huggingface_hub hf-transfer $PYPI_MIRROR
else
    echo "[*] 正在从 Hugging Face 官方直连下载模型..."
    pip install --upgrade huggingface_hub hf-transfer $PYPI_MIRROR
fi

# 开始下载 Qwen3-ASR 及其强制对齐模型 (MS与HF仓库名一致)
download_model "Qwen/Qwen3-ASR-0.6B" "$MODEL_ROOT_DIR/Qwen/Qwen3-ASR-0.6B"
download_model "Qwen/Qwen3-ForcedAligner-0.6B" "$MODEL_ROOT_DIR/Qwen/Qwen3-ForcedAligner-0.6B"

# ==========================================
# 7. 预热并下载 VAD 小模型 (优雅的包触发)
# ==========================================
echo ""
echo "[*] 正在预先下载并缓存 Silero VAD 模型..."
# 不再需要 yes y | 管道拦截
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

SERVER_HOST="127.0.0.1"
SERVER_PORT="6008"

echo "[*] 正在生成 .env 环境变量配置文件..."

cat <<EOF > .env
# 硬件加速模式
DEVICE=$DEVICE_ENV

# 本地化模型绝对路径
ASR_MODEL_PATH=$MODEL_ROOT_DIR/Qwen/Qwen3-ASR-0.6B
FA_MODEL_PATH=$MODEL_ROOT_DIR/Qwen/Qwen3-ForcedAligner-0.6B

# FastAPI 服务配置
HOST=$SERVER_HOST
PORT=$SERVER_PORT
EOF

# 注意：swapoff 需要 root 权限，如果在非 root 环境执行会报错导致脚本退出（因为上方开启了 set -e）
# 如果用户没有 root 权限，可以加上 || true 防止脚本意外中断
echo "正在尝试关闭 swap 内存交互，避免数据从内存卸载到硬盘..."
swapoff -a || true 

# IPA 依赖，epitran 库所需
apt-get install -y flite
echo ""
echo "✅ 环境配置与所有模型（Qwen & VAD）全量下载完毕！"
echo "🚀 正在启动服务..."
# uv run --no-project python main.py
source .venv/bin/activate && python main.py