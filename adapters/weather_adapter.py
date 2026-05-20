"""
天气适配器
==========

支持和风天气API和模拟数据模式的天气适配器

支持:
1. 和风天气API (HeFeng Weather API)
2. 模拟数据模式 (用于测试/开发)
"""
from __future__ import annotations

import os
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base import BaseAdapter

logger = logging.getLogger(__name__)


@dataclass
class WeatherConfig:
    """天气适配器配置"""
    # 和风天气API配置
    hefeng_api_key: str = os.environ.get("HEFENG_API_KEY", "")
    hefeng_base_url: str = "https://devapi.qweather.com/v7/weather"
    
    # 模拟数据配置
    use_mock: bool = os.environ.get("WEATHER_USE_MOCK", "true").lower() == "true"
    mock_city: str = "桂林"
    
    # 缓存配置
    cache_ttl: int = 1800  # 缓存有效期(秒), 默认30分钟
    
    # API请求配置
    timeout: int = 10


class MockWeatherData:
    """模拟天气数据生成器"""
    
    # 天气状况映射
    WEATHER_CONDITIONS = [
        {"text": "晴", "code": "100"},
        {"text": "多云", "code": "101"},
        {"text": "阴", "code": "104"},
        {"text": "小雨", "code": "301"},
        {"text": "中雨", "code": "302"},
        {"text": "大雨", "code": "303"},
        {"text": "雷阵雨", "code": "308"},
        {"text": "小雪", "code": "401"},
        {"text": "中雪", "code": "402"},
        {"text": "大雾", "code": "501"},
    ]
    
    # 风力等级
    WIND_DIRECTIONS = ["北风", "东北风", "东风", "东南风", "南风", "西南风", "西风", "西北风"]
    WIND_LEVELS = ["0级", "1级", "2级", "3级", "4级", "5级"]
    
    @classmethod
    def generate_weather(cls, city: str = "桂林") -> Dict[str, Any]:
        """生成模拟天气数据"""
        now = datetime.now()
        
        # 随机选择天气状况
        weather = random.choice(cls.WEATHER_CONDITIONS)
        wind_dir = random.choice(cls.WIND_DIRECTIONS)
        wind_level = random.choice(cls.WIND_LEVELS)
        
        # 生成温度 (根据天气状况有所区别)
        if weather["code"] in ["100", "101"]:  # 晴/多云
            temp_base = random.randint(18, 32)
        elif weather["code"] in ["301", "302", "303", "308"]:  # 雨
            temp_base = random.randint(12, 25)
        elif weather["code"] in ["401", "402"]:  # 雪
            temp_base = random.randint(-5, 8)
        else:  # 其他
            temp_base = random.randint(15, 28)
        
        # 生成空气质量
        aqi = random.randint(20, 150)
        aqi_level = "优" if aqi <= 50 else "良" if aqi <= 100 else "轻度污染"
        
        return {
            "code": "200",
            "updateTime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "fxLink": f"https://www.qweather.com/weather/{city.lower()}",
            "now": {
                "obsTime": now.strftime("%Y-%m-%d %H:%M:%S"),
                "temp": str(temp_base),
                "feelsLike": str(temp_base - random.randint(1, 3)),
                "icon": weather["code"],
                "text": weather["text"],
                "windDir": wind_dir,
                "windLevel": wind_level.replace("级", ""),
                "windSpeed": str(random.randint(5, 25)),
                "humidity": str(random.randint(40, 95)),
                "precip": str(round(random.uniform(0, 50), 1)),
                "pressure": str(random.randint(980, 1050)),
                "vis": str(random.randint(5, 30)),
                "cloud": str(random.randint(0, 100)),
                "dew": str(temp_base - random.randint(5, 10)),
            },
            "refer": {
                "sources": ["heweather.com"],
                "license": ["CC BY-SA 4.0"],
            },
        }
    
    @classmethod
    def generate_forecast(cls, city: str = "桂林", days: int = 7) -> List[Dict[str, Any]]:
        """生成模拟预报数据"""
        forecasts = []
        now = datetime.now()
        
        for i in range(days):
            target_date = now.replace(hour=8)  # 固定为8点
            weather = random.choice(cls.WEATHER_CONDITIONS)
            wind_dir = random.choice(cls.WIND_DIRECTIONS)
            
            # 温度随天数略有变化
            temp_base = random.randint(15, 30) + (i % 3 - 1) * 3
            
            forecasts.append({
                "fxDate": target_date.strftime("%Y-%m-%d"),
                "sunrise": "06:12",
                "sunset": "18:45",
                "moonrise": "19:23",
                "moonset": "06:15",
                "moonPhase": random.choice(["满月", "残月", "新月", "上弦月", "下弦月"]),
                "tempMax": str(temp_base + random.randint(2, 6)),
                "tempMin": str(temp_base - random.randint(5, 10)),
                "weatherDay": weather["text"],
                "weatherNight": random.choice(cls.WEATHER_CONDITIONS)["text"],
                "iconDay": weather["code"],
                "iconNight": random.choice(cls.WEATHER_CONDITIONS)["code"],
                "windDirDay": wind_dir,
                "windDirNight": random.choice(cls.WIND_DIRECTIONS),
                "windLevelDay": random.choice(cls.WIND_LEVELS).replace("级", ""),
                "windLevelNight": random.choice(cls.WIND_LEVELS).replace("级", ""),
                "humidity": str(random.randint(40, 90)),
            })
        
        return forecasts


class HeFengWeatherAdapter(BaseAdapter):
    """
    和风天气适配器
    
    支持:
    - 实时天气查询
    - 天气预报查询
    - 空气质量查询
    
    使用和风天气API v7版本
    文档: https://dev.qweather.com/docs/api/
    """
    
    def __init__(self, config: Optional[WeatherConfig] = None):
        super().__init__()
        self.config = config or WeatherConfig()
        self._cache: Dict[str, tuple[Any, float]] = {}  # {cache_key: (data, timestamp)}
    
    def initialize(self) -> bool:
        """初始化适配器"""
        if self._initialized:
            return True
        
        if not self.config.use_mock and not self.config.hefeng_api_key:
            logger.error("和风天气API密钥未设置，请设置 HEFENG_API_KEY 环境变量")
            return False
        
        self._initialized = True
        logger.info(f"天气适配器初始化完成，模式: {'模拟数据' if self.config.use_mock else '和风API'}")
        return True
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if datetime.now().timestamp() - timestamp < self.config.cache_ttl:
                return data
            del self._cache[key]
        return None
    
    def _set_cache(self, key: str, data: Any):
        """设置缓存"""
        self._cache[key] = (data, datetime.now().timestamp())
    
    def _http_get(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """发送HTTP GET请求"""
        try:
            import urllib.parse
            import urllib.request
            
            full_url = f"{url}?{urllib.parse.urlencode(params)}"
            logger.debug(f"请求URL: {full_url}")
            
            req = urllib.request.Request(full_url)
            req.add_header("User-Agent", "LvyouHarness/1.0")
            
            with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
                
        except Exception as e:
            logger.error(f"HTTP请求失败: {e}")
            return None
    
    def get_current_weather(self, location: str) -> Dict[str, Any]:
        """
        获取实时天气
        
        Args:
            location: 城市名或坐标 (如 "桂林" 或 "110.29,25.28")
            
        Returns:
            天气数据字典，包含:
            - code: 状态码
            - now: 实时天气信息
            - updateTime: 更新时间
        """
        self.ensure_initialized()
        
        cache_key = f"current_{location}"
        cached = self._get_cached(cache_key)
        if cached:
            logger.debug(f"返回缓存的天气数据: {location}")
            return cached
        
        if self.config.use_mock:
            result = MockWeatherData.generate_weather(location)
        else:
            params = {
                "key": self.config.hefeng_api_key,
                "location": location,
            }
            result = self._http_get(f"{self.config.hefeng_base_url}/now", params)
        
        if result:
            self._set_cache(cache_key, result)
        
        return result
    
    def get_forecast(self, location: str, days: int = 7) -> Dict[str, Any]:
        """
        获取天气预报
        
        Args:
            location: 城市名或坐标
            days: 预报天数 (1-7天)
            
        Returns:
            预报数据字典
        """
        self.ensure_initialized()
        
        # 限制天数
        days = max(1, min(days, 7))
        
        cache_key = f"forecast_{location}_{days}"
        cached = self._get_cached(cache_key)
        if cached:
            logger.debug(f"返回缓存的预报数据: {location}")
            return cached
        
        if self.config.use_mock:
            forecasts = MockWeatherData.generate_forecast(location, days)
            result = {
                "code": "200",
                "updateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "fxLink": f"https://www.qweather.com/weather/{location.lower()}",
                "daily": forecasts,
                "refer": {
                    "sources": ["heweather.com"],
                    "license": ["CC BY-SA 4.0"],
                },
            }
        else:
            params = {
                "key": self.config.hefeng_api_key,
                "location": location,
                "days": days,
            }
            result = self._http_get(f"{self.config.hefeng_base_url}/{days}d", params)
        
        if result:
            self._set_cache(cache_key, result)
        
        return result
    
    def get_air_quality(self, location: str) -> Dict[str, Any]:
        """
        获取空气质量
        
        Args:
            location: 城市名或坐标
            
        Returns:
            空气质量数据
        """
        self.ensure_initialized()
        
        cache_key = f"air_{location}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        if self.config.use_mock:
            aqi = random.randint(20, 150)
            result = {
                "code": "200",
                "updateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "now": {
                    "aqi": str(aqi),
                    "level": "优" if aqi <= 50 else "良" if aqi <= 100 else "轻度污染" if aqi <= 150 else "中度污染",
                    "category": "空气质量" + ("较好" if aqi <= 50 else "良好" if aqi <= 100 else "一般"),
                    "pm2p5": str(random.randint(5, 80)),
                    "pm10": str(random.randint(10, 150)),
                    "so2": str(random.randint(0, 50)),
                    "no2": str(random.randint(0, 80)),
                    "co": str(round(random.uniform(0.2, 1.5), 2)),
                    "o3": str(random.randint(50, 200)),
                },
            }
        else:
            params = {
                "key": self.config.hefeng_api_key,
                "location": location,
            }
            result = self._http_get("https://devapi.qweather.com/v7/air/now", params)
        
        if result:
            self._set_cache(cache_key, result)
        
        return result
    
    def get_weather_summary(self, location: str) -> str:
        """
        获取天气摘要信息 (用于直接展示)
        
        Args:
            location: 城市名
            
        Returns:
            格式化的天气描述字符串
        """
        weather = self.get_current_weather(location)
        
        if weather.get("code") != "200":
            return f"获取天气失败: {weather.get('code', 'unknown')}"
        
        now = weather.get("now", {})
        temp = now.get("temp", "N/A")
        feels_like = now.get("feelsLike", "N/A")
        text = now.get("text", "未知")
        wind_dir = now.get("windDir", "")
        wind_level = now.get("windLevel", "")
        humidity = now.get("humidity", "N/A")
        
        summary = (
            f"{location}当前天气：{text}，温度{temp}°C（体感{feels_like}°C），"
            f"{wind_dir}{wind_level}级，湿度{humidity}%"
        )
        
        return summary
    
    def clear_cache(self):
        """清除所有缓存"""
        self._cache.clear()
        logger.debug("天气缓存已清除")


class MockWeatherAdapter(BaseAdapter):
    """
    纯模拟天气适配器 (更彻底的模拟)
    
    适用于:
    - 单元测试
    - 无网络环境
    - 开发调试
    """
    
    def __init__(self, config: Optional[WeatherConfig] = None):
        super().__init__()
        self.config = config or WeatherConfig()
    
    def initialize(self) -> bool:
        """初始化适配器"""
        self._initialized = True
        logger.info("模拟天气适配器初始化完成")
        return True
    
    def get_current_weather(self, location: str) -> Dict[str, Any]:
        """获取模拟实时天气"""
        return MockWeatherData.generate_weather(location or self.config.mock_city)
    
    def get_forecast(self, location: str, days: int = 7) -> Dict[str, Any]:
        """获取模拟预报"""
        forecasts = MockWeatherData.generate_forecast(
            location or self.config.mock_city, days
        )
        return {
            "code": "200",
            "daily": forecasts,
        }
    
    def get_air_quality(self, location: str) -> Dict[str, Any]:
        """获取模拟空气质量"""
        aqi = random.randint(30, 100)
        return {
            "code": "200",
            "now": {
                "aqi": str(aqi),
                "level": "优" if aqi <= 50 else "良",
            },
        }
    
    def get_weather_summary(self, location: str) -> str:
        """获取模拟天气摘要"""
        weather = self.get_current_weather(location)
        now = weather.get("now", {})
        return (
            f"【模拟数据】{location}当前天气：{now.get('text', '未知')}，"
            f"温度{now.get('temp', 'N/A')}°C"
        )
    
    def clear_cache(self):
        """清除缓存 (模拟器无需缓存)"""
        pass


# =============================================================================
# 工厂函数
# =============================================================================

def create_weather_adapter(
    adapter_type: str = "hefeng",
    **kwargs,
) -> BaseAdapter:
    """
    创建天气适配器
    
    Args:
        adapter_type: 适配器类型
            - "hefeng": 和风天气API (默认)
            - "mock": 纯模拟数据
        **kwargs: 传递给适配器的配置
        
    Returns:
        BaseAdapter实例
        
    Example:
        >>> # 使用和风API
        >>> adapter = create_weather_adapter("hefeng")
        >>> weather = adapter.get_current_weather("桂林")
        
        >>> # 使用模拟数据
        >>> mock_adapter = create_weather_adapter("mock")
        >>> mock_weather = mock_adapter.get_current_weather("桂林")
    """
    if adapter_type == "hefeng":
        config = WeatherConfig(**kwargs)
        return HeFengWeatherAdapter(config)
    elif adapter_type == "mock":
        config = WeatherConfig(use_mock=True, **kwargs)
        return MockWeatherAdapter(config)
    else:
        raise ValueError(f"未知的天气适配器类型: {adapter_type}")


# =============================================================================
# 便捷函数
# =============================================================================

# 全局默认适配器实例
_default_adapter: Optional[BaseAdapter] = None


def get_default_adapter() -> BaseAdapter:
    """获取默认天气适配器"""
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = create_weather_adapter()
        _default_adapter.initialize()
    return _default_adapter


def get_weather(location: str) -> Dict[str, Any]:
    """便捷函数: 获取实时天气"""
    return get_default_adapter().get_current_weather(location)


def get_forecast(location: str, days: int = 7) -> Dict[str, Any]:
    """便捷函数: 获取天气预报"""
    return get_default_adapter().get_forecast(location, days)


def get_weather_summary(location: str) -> str:
    """便捷函数: 获取天气摘要"""
    return get_default_adapter().get_weather_summary(location)
