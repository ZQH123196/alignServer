#!/bin/bash
# 遇到错误立即退出
set -e

# ==========================================
# 1. 变量配置
# ==========================================
AUDIO_URL="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav"
AUDIO_FILE="test_asr_zh.wav"

# 注意：这里默认你的服务运行在本地 6008 端口。如果有变，请修改这里。
API_URL="http://127.0.0.1:6008/v1/audio/transcriptions"

# 💡【核心新增】：对齐模式必须指定合法的 ForcedAligner 模型名称
MODEL="Qwen/Qwen3-ForcedAligner-0.6B"
PROMPT_TEXT="甚至出现，交易几乎停滞的情况。"

# ==========================================
# 2. 下载测试音频
# ==========================================
echo "⬇️ 正在从阿里云 OSS 下载测试音频..."
curl -# -o "$AUDIO_FILE" "$AUDIO_URL"
echo "✅ 下载完成: $AUDIO_FILE"
echo ""

# ==========================================
# 3. 发送 API 请求 (使用 curl)
# ==========================================
echo "🚀 正在向本地 Server 发起【对齐模式】请求..."
echo "  -> 目标地址: $API_URL"
echo "  -> 选定模型: $MODEL"
echo "  -> 参考原文: $PROMPT_TEXT"
echo "  -> 响应格式: verbose_json"
echo ""

curl -X POST "$API_URL" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@$AUDIO_FILE" \
  -F "model=$MODEL" \
  -F "prompt=$PROMPT_TEXT" \
  -F "response_format=verbose_json" \
  -F "language=Chinese"

echo ""
echo ""
echo "✅ 请求执行完毕！"