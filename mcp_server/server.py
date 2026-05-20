"""
LvyouHarness MCP Server - 标准MCP Server (stdio + HTTP双模式)
=============================================================

支持两种运行模式:
1. Stdio模式 (MCP Protocol over stdio) - 默认，用于Claude Desktop等MCP客户端
2. HTTP模式 (MCP Protocol over HTTP/Streamable) - 通过--http参数启用

依赖:
    pip install mcp-server-python fastmcp

运行方式:
    # Stdio模式 (MCP协议通过标准输入输出)
    cd /home/l2140/lvyou_harness
    PYTHONPATH=/home/l2140 python -m mcp_server.server

    # HTTP模式 (通过HTTP提供MCP协议，支持SSE)
    cd /home/l2140/lvyou_harness
    PYTHONPATH=/home/l2140 python -m mcp_server.server --http --port 8765

    # 启用调试日志
    python -m mcp_server.server --http --port 8765 --log-level DEBUG
"""
from __future__ import annotations

import os
import sys
import logging
import asyncio
import argparse
from typing import Any, Dict, List, Optional

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP

# =============================================================================
# 日志配置
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# MCP Server 实例
# =============================================================================

mcp = FastMCP(
    name="LvyouHarness",
    instructions="旅游领域AGI系统的MCP Server，提供天气查询、地理编码、汇率换算等工具",
)


# =============================================================================
# 适配器初始化
# =============================================================================

_weather_adapter = None
_geocoder_adapter = None
_exchange_adapter = None


def get_weather_adapter():
    """获取天气适配器 (延迟初始化)"""
    global _weather_adapter
    if _weather_adapter is None:
        from lvyou_harness.adapters.weather_adapter import create_weather_adapter
        _weather_adapter = create_weather_adapter("hefeng")
        _weather_adapter.initialize()
        logger.info("天气适配器初始化完成")
    return _weather_adapter


def get_geocoder_adapter():
    """获取地理编码适配器 (延迟初始化)"""
    global _geocoder_adapter
    if _geocoder_adapter is None:
        from lvyou_harness.adapters.geocoder_adapter import GeocoderAdapter
        _geocoder_adapter = GeocoderAdapter()
        _geocoder_adapter.initialize()
        logger.info("地理编码适配器初始化完成")
    return _geocoder_adapter


def get_exchange_adapter():
    """获取汇率适配器 (延迟初始化)"""
    global _exchange_adapter
    if _exchange_adapter is None:
        from lvyou_harness.adapters.finance_adapter import create_exchange_rate_adapter
        _exchange_adapter = create_exchange_rate_adapter("mock")
        _exchange_adapter.initialize()
        logger.info("汇率适配器初始化完成")
    return _exchange_adapter


# =============================================================================
# 天气类工具 (Weather Tools)
# =============================================================================

@mcp.tool()
def get_current_weather(location: str) -> Dict[str, Any]:
    """
    获取指定地点的当前天气信息。

    Args:
        location: 城市名称或坐标，如 "桂林"、"阳朔" 或 "110.29,25.28"

    Returns:
        包含温度、天气状况、风力、湿度等信息的字典
    """
    try:
        adapter = get_weather_adapter()
        result = adapter.get_current_weather(location)
        return {
            "success": True,
            "location": location,
            "weather": result,
        }
    except Exception as e:
        logger.error(f"获取天气失败: {e}")
        return {"success": False, "error": str(e), "location": location}


@mcp.tool()
def get_weather_forecast(location: str, days: int = 7) -> Dict[str, Any]:
    """
    获取指定地点的天气预报。

    Args:
        location: 城市名称或坐标，如 "桂林"
        days: 预报天数，默认7天，最大7天

    Returns:
        包含未来几天天气预报的字典
    """
    try:
        adapter = get_weather_adapter()
        days = max(1, min(days, 7))
        result = adapter.get_forecast(location, days)
        return {
            "success": True,
            "location": location,
            "days": days,
            "forecast": result,
        }
    except Exception as e:
        logger.error(f"获取预报失败: {e}")
        return {"success": False, "error": str(e), "location": location}


@mcp.tool()
def get_air_quality(location: str) -> Dict[str, Any]:
    """
    获取指定地点的空气质量。

    Args:
        location: 城市名称或坐标

    Returns:
        包含AQI、PM2.5、PM10等空气质量数据的字典
    """
    try:
        adapter = get_weather_adapter()
        result = adapter.get_air_quality(location)
        return {
            "success": True,
            "location": location,
            "air_quality": result,
        }
    except Exception as e:
        logger.error(f"获取空气质量失败: {e}")
        return {"success": False, "error": str(e), "location": location}


# =============================================================================
# 地理编码类工具 (Geocoder Tools)
# =============================================================================

@mcp.tool()
def geocode_address(address: str) -> Dict[str, Any]:
    """
    将地址转换为地理坐标。

    Args:
        address: 地址字符串，如 "桂林象鼻山"、"阳朔西街"

    Returns:
        包含经纬度、省份、城市、区县等信息的字典
    """
    try:
        adapter = get_geocoder_adapter()
        result = adapter.geocode(address)
        if result:
            return {
                "success": True,
                "address": address,
                "location": result.to_dict(),
            }
        else:
            return {
                "success": False,
                "error": f"未找到地址: {address}",
                "address": address,
            }
    except Exception as e:
        logger.error(f"地理编码失败: {e}")
        return {"success": False, "error": str(e), "address": address}


@mcp.tool()
def reverse_geocode(lat: float, lng: float) -> Dict[str, Any]:
    """
    将地理坐标转换为地址。

    Args:
        lat: 纬度，如 25.2628
        lng: 经度，如 110.2974

    Returns:
        包含地址、省份、城市、区县等信息的字典
    """
    try:
        adapter = get_geocoder_adapter()
        result = adapter.reverse_geocode(lat, lng)
        if result:
            return {
                "success": True,
                "lat": lat,
                "lng": lng,
                "address_info": result.to_dict(),
            }
        else:
            return {
                "success": False,
                "error": f"未找到坐标 ({lat}, {lng}) 对应的地址",
                "lat": lat,
                "lng": lng,
            }
    except Exception as e:
        logger.error(f"逆地理编码失败: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def batch_geocode(addresses: List[str]) -> Dict[str, Any]:
    """
    批量将地址转换为地理坐标。

    Args:
        addresses: 地址字符串列表

    Returns:
        包含批量转换结果的字典
    """
    try:
        adapter = get_geocoder_adapter()
        results = adapter.batch_geocode(addresses)
        locations = []
        for addr, loc in zip(addresses, results):
            if loc:
                locations.append({"address": addr, "location": loc.to_dict()})
            else:
                locations.append({"address": addr, "location": None, "error": "未找到"})
        return {
            "success": True,
            "count": len(addresses),
            "results": locations,
        }
    except Exception as e:
        logger.error(f"批量地理编码失败: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# 汇率/金融类工具 (Finance Tools)
# =============================================================================

@mcp.tool()
def get_exchange_rate(from_currency: str, to_currency: str) -> Dict[str, Any]:
    """
    获取两种货币之间的汇率。

    Args:
        from_currency: 源货币代码，如 "USD"、"EUR"
        to_currency: 目标货币代码，如 "CNY"、"JPY"

    Returns:
        包含汇率值的字典
    """
    try:
        adapter = get_exchange_adapter()
        rate = adapter.get_exchange_rate(from_currency.upper(), to_currency.upper())
        if rate is not None:
            return {
                "success": True,
                "from_currency": from_currency.upper(),
                "to_currency": to_currency.upper(),
                "rate": rate,
            }
        else:
            return {
                "success": False,
                "error": f"无法获取 {from_currency} -> {to_currency} 的汇率",
            }
    except Exception as e:
        logger.error(f"获取汇率失败: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def convert_currency(amount: float, from_currency: str, to_currency: str) -> Dict[str, Any]:
    """
    货币换算。

    Args:
        amount: 金额
        from_currency: 源货币代码
        to_currency: 目标货币代码

    Returns:
        包含换算结果的字典
    """
    try:
        adapter = get_exchange_adapter()
        result = adapter.convert(amount, from_currency.upper(), to_currency.upper())
        rate = adapter.get_exchange_rate(from_currency.upper(), to_currency.upper())
        if result is not None:
            return {
                "success": True,
                "amount": amount,
                "from_currency": from_currency.upper(),
                "to_currency": to_currency.upper(),
                "converted_amount": round(result, 2),
                "rate": rate,
            }
        else:
            return {
                "success": False,
                "error": f"无法换算 {amount} {from_currency} -> {to_currency}",
            }
    except Exception as e:
        logger.error(f"货币换算失败: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def get_all_exchange_rates(base_currency: str = "USD") -> Dict[str, Any]:
    """
    获取相对于基准货币的所有汇率。

    Args:
        base_currency: 基准货币代码，默认 "USD"

    Returns:
        包含所有汇率的字典
    """
    try:
        adapter = get_exchange_adapter()
        rates = adapter.get_all_rates(base_currency.upper())
        if rates is not None:
            return {
                "success": True,
                "base_currency": base_currency.upper(),
                "rates": rates,
            }
        else:
            return {
                "success": False,
                "error": f"无法获取以 {base_currency} 为基准的汇率",
            }
    except Exception as e:
        logger.error(f"获取所有汇率失败: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def get_supported_currencies() -> Dict[str, Any]:
    """
    获取支持的货币列表。

    Returns:
        包含支持的货币代码列表的字典
    """
    try:
        adapter = get_exchange_adapter()
        currencies = adapter.get_supported_currencies()
        return {
            "success": True,
            "currencies": currencies,
            "count": len(currencies),
        }
    except Exception as e:
        logger.error(f"获取货币列表失败: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# RAG和旅游业务类工具
# =============================================================================

_rag_adapter = None
_llm_adapter = None


def get_rag_adapter():
    """获取RAG适配器 (延迟初始化)"""
    global _rag_adapter
    if _rag_adapter is None:
        from lvyou_harness.adapters.rag_adapter_v2 import MilvusRAGAdapter
        _rag_adapter = MilvusRAGAdapter()
        _rag_adapter.initialize()
        logger.info(f"RAG适配器初始化完成，文档数: {_rag_adapter.count()}")
    return _rag_adapter


def get_llm_adapter():
    """获取LLM适配器 (延迟初始化)"""
    global _llm_adapter
    if _llm_adapter is None:
        try:
            from lvyou_harness.adapters.llm_adapter import MiniMaxAdapter
            _llm_adapter = MiniMaxAdapter()
            _llm_adapter.initialize()
            logger.info("LLM适配器初始化完成")
        except Exception as e:
            logger.warning(f"LLM适配器初始化失败: {e}")
    return _llm_adapter


@mcp.tool()
def query_scenic(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    查询景点相关信息（RAG检索）。

    Args:
        query: 查询文本，如 "桂林象鼻山"
        top_k: 返回结果数量，默认5条

    Returns:
        包含景点信息的结果列表
    """
    try:
        adapter = get_rag_adapter()
        results = adapter.retrieve(query, top_k=top_k)
        return {
            "success": True,
            "query": query,
            "results": [
                {
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0),
                    "source": r.get("source", ""),
                }
                for r in results
            ],
            "count": len(results),
        }
    except Exception as e:
        logger.error(f"景点查询失败: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def retrieve(query: str, top_k: int = 5, data_type: Optional[str] = None) -> Dict[str, Any]:
    """
    RAG检索（通用）。

    Args:
        query: 查询文本
        top_k: 返回结果数量，默认5条
        data_type: 数据类型过滤，如 "attraction", "hotel", "restaurant"

    Returns:
        检索结果列表
    """
    try:
        adapter = get_rag_adapter()
        results = adapter.retrieve(query, top_k=top_k * 2)  # 多检索一些用于过滤

        # 按data_type过滤
        if data_type:
            results = [r for r in results if r.get("data_type") == data_type]

        return {
            "success": True,
            "query": query,
            "results": results[:top_k],
            "count": len(results[:top_k]),
        }
    except Exception as e:
        logger.error(f"RAG检索失败: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def plan_route(destination: str, days: int, budget: Optional[int] = None, style: str = "普通", travelers: int = 2) -> Dict[str, Any]:
    """
    规划旅行行程。

    Args:
        destination: 目的地，如 "桂林"
        days: 旅行天数
        budget: 预算（元），可选
        style: 旅行风格，如 "普通"、"亲子"、"情侣"、"探险"
        travelers: 出行人数，默认2人

    Returns:
        行程规划结果
    """
    try:
        # 检索相关景点
        rag = get_rag_adapter()
        scenic_query = f"{destination}{days}天旅游热门景点"
        results = rag.retrieve(scenic_query, top_k=10)

        scenic_context = "\n".join([
            f"- {r.get('title', '')}: {r.get('content', '')[:100]}"
            for r in results
        ])

        # 构建行程
        route_plan = f"根据您的需求，为您规划{days}天{destination}之旅：\n\n"
        route_plan += f"推荐游览的景点包括：\n{scenic_context}\n\n"
        route_plan += "（注：完整行程规划需要LLM生成）"

        return {
            "success": True,
            "destination": destination,
            "days": days,
            "style": style,
            "travelers": travelers,
            "budget": budget,
            "route_plan": route_plan,
        }
    except Exception as e:
        logger.error(f"行程规划失败: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def write_guide(destination: str, days: int, route_plan: Optional[str] = None) -> Dict[str, Any]:
    """
    撰写旅行攻略。

    Args:
        destination: 目的地
        days: 旅行天数
        route_plan: 已有行程规划（可选）

    Returns:
        旅行攻略内容
    """
    try:
        guide = f"# {destination}旅行攻略\n\n"
        guide += f"## 行程概览\n{days}天{destination}之旅\n\n"
        guide += "## 详细攻略\n（攻略内容需要LLM生成）\n"
        if route_plan:
            guide += f"\n## 推荐行程\n{route_plan}\n"
        guide += "\n## 实用Tips\n- 提前查看天气预报\n- 旺季提前预订酒店\n- 携带防晒用品"

        return {
            "success": True,
            "destination": destination,
            "days": days,
            "guide": guide,
        }
    except Exception as e:
        logger.error(f"攻略写作失败: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def optimize_budget(destination: str, days: int, total_budget: int, style: str = "普通") -> Dict[str, Any]:
    """
    优化旅行预算分配。

    Args:
        destination: 目的地
        days: 旅行天数
        total_budget: 总预算（元）
        style: 旅行风格

    Returns:
        预算分配方案
    """
    try:
        # 标准预算分配
        allocation = {
            "住宿": {"比例": "35%", "金额": int(total_budget * 0.35)},
            "餐饮": {"比例": "22%", "金额": int(total_budget * 0.22)},
            "交通": {"比例": "18%", "金额": int(total_budget * 0.18)},
            "门票": {"比例": "15%", "金额": int(total_budget * 0.15)},
            "购物": {"比例": "10%", "金额": int(total_budget * 0.10)},
        }

        tips = [
            "提前预订机票/火车票可节省20-30%",
            "选择淡季出行，酒店价格更低",
            "尝试当地小吃，比网红餐厅更实惠",
            "购买景点联票比单买更划算",
        ]

        return {
            "success": True,
            "destination": destination,
            "days": days,
            "total_budget": total_budget,
            "style": style,
            "allocation": allocation,
            "tips": tips,
        }
    except Exception as e:
        logger.error(f"预算优化失败: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def get_weather_and_location(city: str) -> Dict[str, Any]:
    """
    获取城市的天气和地理信息 (组合工具)。

    Args:
        city: 城市名称，如 "桂林"、"阳朔"

    Returns:
        包含天气信息和地理位置的字典
    """
    weather_result = get_current_weather(city)
    geo_result = geocode_address(city)

    return {
        "city": city,
        "weather": weather_result,
        "geolocation": geo_result,
    }


# =============================================================================
# 主入口
# =============================================================================

def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description="LvyouHarness MCP Server - 标准MCP Server (stdio + HTTP双模式)"
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="启用HTTP模式 (MCP over HTTP/Streamable)，默认启用stdio模式",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("LVYOU_PORT", "8765")),
        help="HTTP模式端口号 (默认: 8765)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("LVYOU_HOST", "0.0.0.0"),
        help="HTTP模式主机地址 (默认: 0.0.0.0)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    args = parser.parse_args()

    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.http:
        # HTTP模式 - 使用uvicorn运行streamable_http_app
        import uvicorn
        logger.info(f"启动HTTP模式 MCP Server: http://{args.host}:{args.port}")
        logger.info("使用 streamable-http 传输协议")
        app = mcp.streamable_http_app()
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level.lower(),
        )
    else:
        # Stdio模式 (默认)
        logger.info("启动Stdio模式 MCP Server")
        logger.info("通过标准输入输出与MCP客户端通信")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
