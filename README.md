# LvyouHarness 旅游AGI系统

旅游领域垂直AGI系统，基于多智能体架构，支持RAG检索、MCP协议扩展。

## 特性

- **多智能体架构**: Researcher/Modeler/Coder/Writer Agent协作
- **RAG检索**: BGE-m3嵌入 + Milvus向量库 + BM25混合检索 + BGE-reranker重排
- **MCP Server**: 标准MCP协议，支持stdio和HTTP双模式
- **工具扩展**: 天气查询、地理编码、汇率转换等

## 架构

```
lvyou_harness/
├── adapters/          # 适配器层
│   ├── rag_adapter_v2.py    # RAG适配器（BM25+向量+重排）
│   ├── weather_adapter.py    # 天气适配器
│   ├── geocoder_adapter.py   # 地理编码适配器
│   └── finance_adapter.py    # 汇率适配器
├── mcp_server/       # MCP Server
│   └── server.py      # 标准MCP Server (FastMCP)
├── embedding/         # 嵌入模型
├── pipeline/          # 处理流水线
├── agents/            # Agent实现
└── data/              # 数据目录
```

## MCP工具

### 天气工具
- `get_current_weather` - 获取实时天气
- `get_weather_forecast` - 获取天气预报
- `get_air_quality` - 获取空气质量

### 地理编码工具
- `geocode_address` - 地址转坐标
- `reverse_geocode` - 坐标转地址
- `batch_geocode` - 批量地理编码

### 汇率工具
- `get_exchange_rate` - 获取汇率
- `convert_currency` - 货币转换
- `get_all_exchange_rates` - 获取所有汇率

### RAG检索工具
- `query_scenic` - 景点查询
- `retrieve` - 通用RAG检索
- `plan_route` - 行程规划
- `write_guide` - 攻略写作
- `optimize_budget` - 预算优化

## 启动

```bash
cd /home/l2140/lvyou_harness

# Stdio模式 (MCP协议)
PYTHONPATH=/home/l2140 python -m mcp_server.server

# HTTP模式
PYTHONPATH=/home/l2140 python -m mcp_server.server --http --port 8765

# 使用启动脚本
./start_mcp.sh http 8765
```

## 环境变量

```bash
# 数据库
LVYOU_DB=/home/l2140/milvus_rag.db

# 天气API (可选)
HEFENG_API_KEY=your-key

# 高德地图API (可选)
AMAP_KEY=your-key

# 汇率API (可选)
EXCHANGE_RATE_API_KEY=your-key

# 重排模型 (可选)
RERANK_MODEL_PATH=/mnt/f/LLM/models/bge-reranker-large
```

## 依赖

```bash
pip install -r requirements.txt
```

核心依赖:
- `fastapi`, `uvicorn` - HTTP服务
- `mcp-server-python`, `fastmcp` - MCP协议
- `pymilvus`, `milvus-lite` - 向量数据库
- `sentence-transformers` - 重排模型
- `rank-bm25` - BM25检索
- `jieba` - 中文分词
