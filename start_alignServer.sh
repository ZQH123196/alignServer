#!/usr/bin/env bash
# 启动长音频强制对齐服务，交给 pm2 托管。
set -euo pipefail

cd "$(dirname "$0")"

# 依赖：pm2 与 pm2-logrotate。node 若由 apt 安装、无写权限，需先手动
# sudo npm install -g pm2。
command -v pm2 >/dev/null || npm install -g pm2
pm2 describe pm2-logrotate >/dev/null 2>&1 || pm2 install pm2-logrotate

# source .venv/bin/activate && python main.py
pm2 delete alignServer >/dev/null 2>&1 || true
pm2 start main.py --interpreter "$PWD/.venv/bin/python" --name alignServer --cwd "$PWD"

echo
echo "日志：~/.pm2/logs/alignServer-out.log    stdout，业务错误也在此文件"
echo "      ~/.pm2/logs/alignServer-error.log  stderr，Python 崩溃堆栈"
echo
pm2 logs alignServer
