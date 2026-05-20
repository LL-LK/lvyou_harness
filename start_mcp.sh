#!/bin/bash
# LvyouHarness MCP Server 启动脚本
# ================================

cd /home/l2140/lvyou_harness

# 设置Python路径
export PYTHONPATH=/home/l2140/lvyou_harness

# 可选：设置API密钥
# export HEFENG_API_KEY="your-key"
# export AMAP_KEY="your-key"
# export EXCHANGE_RATE_API_KEY="your-key"

# 默认启动HTTP模式
MODE=${1:-http}
PORT=${2:-8765}

if [ "$MODE" = "stdio" ]; then
    echo "启动Stdio模式 MCP Server..."
    python -m mcp_server.server
elif [ "$MODE" = "http" ]; then
    echo "启动HTTP模式 MCP Server (端口 $PORT)..."
    python -m mcp_server.server --http --port $PORT
else
    echo "用法: $0 [stdio|http] [port]"
    echo "  stdio - MCP协议通过标准输入输出"
    echo "  http  - MCP协议通过HTTP (默认, 端口8765)"
fi
