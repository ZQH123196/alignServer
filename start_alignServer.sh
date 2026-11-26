#!/usr/bin/env bash
# 启动长音频强制对齐服务，交给 pm2 托管。
#
# 这是生产环境唯一的启动入口。禁止直接 `python main.py` 起服务。
# 前置条件：环境必须由 autoInstall.sh 装出（.venv 与 .env 都在本目录下）。
set -euo pipefail

cd "$(dirname "$0")"

# 启动前置校验：缺任何一项就停在这里，不让 pm2 起一个必然崩的进程。
if [ ! -x ".venv/bin/python" ]; then
    echo "[FATAL] 找不到 .venv/bin/python。环境必须用 ./autoInstall.sh 装。" >&2
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "[FATAL] 找不到 .env。环境必须用 ./autoInstall.sh 装，它会生成 .env。" >&2
    exit 1
fi

# 依赖锁校验在 main.py 启动期做（check_lockfile），此处只做文件存在性检查。
command -v pm2 >/dev/null || npm install -g pm2

# 日志轮转不是启动的必要条件。装上失败只告警，不让服务起不来。
if ! pm2 describe pm2-logrotate >/dev/null 2>&1; then
    pm2 install pm2-logrotate >/dev/null 2>&1 \
        || echo "⚠️ pm2-logrotate 安装失败，日志不会自动轮转"
fi

pm2 delete alignServer >/dev/null 2>&1 || true
pm2 start main.py --interpreter "$PWD/.venv/bin/python" --name alignServer --cwd "$PWD"

echo
echo "业务日志：$PWD/logs/alignServer.log（由 LOGGING_FILE 指定，服务自行异步写盘与轮转）"
echo "pm2 原始输出：~/.pm2/logs/alignServer-out.log（stdout）/ alignServer-error.log（stderr）"
echo
pm2 logs alignServer
