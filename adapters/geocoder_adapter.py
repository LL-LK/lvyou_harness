"""
地理编码适配器
============

支持高德地图API和模拟数据的地理编码适配器

功能:
1. 地址 -> 坐标 (geocode)
2. 坐标 -> 地址 (reverse_geocode)
3. 批量地理编码

支持模式:
- amap: 高德地图API (需要配置API Key)
- mock: 模拟数据模式 (用于测试/开发)
"""

from __future__ import annotations

import os
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class GeocoderMode(Enum):
    """地理编码器模式"""
    AMAP = "amap"      # 高德地图API
    MOCK = "mock"      # 模拟数据


@dataclass
class GeocoderConfig:
    """地理编码器配置"""
    mode: str = os.environ.get("GEOCODER_MODE", "mock")
    amap_key: str = os.environ.get("AMAP_KEY", "")
    amap_secret: str = os.environ.get("AMAP_SECRET", "")
    timeout: int = 10
    retry_times: int = 3
    cache_dir: Optional[str] = os.environ.get("GEOCODER_CACHE_DIR", None)


@dataclass
class GeoLocation:
    """地理坐标"""
    lat: float
    lng: float
    province: str = ""
    city: str = ""
    district: str = ""
    address: str = ""
    name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lat": self.lat,
            "lng": self.lng,
            "province": self.province,
            "city": self.city,
            "district": self.district,
            "address": self.address,
            "name": self.name,
        }


class MockGeocoder:
    """
    模拟地理编码器

    使用预定义的地理位置数据进行模拟，适用于测试和开发环境
    """

    # 预定义的地理位置数据 (主要用于桂林及周边景区)
    MOCK_LOCATIONS: Dict[str, GeoLocation] = {
        # 桂林市区
        "桂林": GeoLocation(lat=25.2736, lng=110.2900, province="广西", city="桂林市", district="秀峰区", address="桂林市", name="桂林市"),
        "桂林市": GeoLocation(lat=25.2736, lng=110.2900, province="广西", city="桂林市", district="秀峰区", address="桂林市", name="桂林市"),
        "桂林火车站": GeoLocation(lat=25.2799, lng=110.2967, province="广西", city="桂林市", district="象山区", address="桂林站", name="桂林火车站"),
        "桂林两江国际机场": GeoLocation(lat=25.2136, lng=110.2005, province="广西", city="桂林市", district="临桂区", address="两江国际机场", name="桂林两江国际机场"),

        # 桂林景区
        "象鼻山": GeoLocation(lat=25.2628, lng=110.2974, province="广西", city="桂林市", district="象山区", address="象山景区", name="象鼻山"),
        "两江四湖": GeoLocation(lat=25.2816, lng=110.2868, province="广西", city="桂林市", district="象山区", address="两江四湖", name="两江四湖"),
        "靖江王府": GeoLocation(lat=25.2759, lng=110.2906, province="广西", city="桂林市", district="秀峰区", address="靖江王城景区", name="靖江王府"),
        "叠彩山": GeoLocation(lat=25.2894, lng=110.2927, province="广西", city="桂林市", district="叠彩区", address="叠彩山", name="叠彩山"),
        "伏波山": GeoLocation(lat=25.2856, lng=110.2951, province="广西", city="桂林市", district="叠彩区", address="伏波山", name="伏波山"),
        "七星公园": GeoLocation(lat=25.2775, lng=110.3112, province="广西", city="桂林市", district="七星区", address="七星公园", name="七星公园"),
        "芦笛岩": GeoLocation(lat=25.3029, lng=110.2801, province="广西", city="桂林市", district="秀峰区", address="芦笛岩景区", name="芦笛岩"),

        # 阳朔
        "阳朔": GeoLocation(lat=24.7759, lng=110.4960, province="广西", city="桂林市", district="阳朔县", address="阳朔县", name="阳朔县"),
        "阳朔西街": GeoLocation(lat=24.7727, lng=110.4967, province="广西", city="桂林市", district="阳朔县", address="阳朔西街", name="阳朔西街"),
        "漓江": GeoLocation(lat=25.0967, lng=110.4833, province="广西", city="桂林市", district="阳朔县", address="漓江", name="漓江"),
        "遇龙河": GeoLocation(lat=24.8178, lng=110.4500, province="广西", city="桂林市", district="阳朔县", address="遇龙河", name="遇龙河"),
        "十里画廊": GeoLocation(lat=24.7892, lng=110.4656, province="广西", city="桂林市", district="阳朔县", address="十里画廊", name="十里画廊"),
        "印象刘三姐": GeoLocation(lat=24.7689, lng=110.5017, province="广西", city="桂林市", district="阳朔县", address="印象刘三姐剧场", name="印象刘三姐"),
        "世外桃源": GeoLocation(lat=24.7583, lng=110.5333, province="广西", city="桂林市", district="阳朔县", address="世外桃源景区", name="世外桃源"),
        "银子岩": GeoLocation(lat=24.8167, lng=110.4167, province="广西", city="桂林市", district="荔浦县", address="银子岩景区", name="银子岩"),

        # 龙胜
        "龙脊梯田": GeoLocation(lat=25.7586, lng=110.0889, province="广西", city="桂林市", district="龙胜各族自治县", address="龙脊梯田", name="龙脊梯田"),
        "龙胜温泉": GeoLocation(lat=25.6917, lng=110.0833, province="广西", city="桂林市", district="龙胜各族自治县", address="龙胜温泉", name="龙胜温泉"),

        # 其他
        "灵渠": GeoLocation(lat=25.6081, lng=110.8131, province="广西", city="桂林市", district="兴安县", address="灵渠景区", name="灵渠"),
        "乐满地": GeoLocation(lat=25.4333, lng=110.1333, province="广西", city="桂林市", district="兴安县", address="乐满地度假世界", name="乐满地"),
    }

    def __init__(self, config: Optional[GeocoderConfig] = None):
        self.config = config or GeocoderConfig()
        self._cache: Dict[str, GeoLocation] = {}

    def _get_cache_key(self, query: str) -> str:
        """生成缓存key"""
        return hashlib.md5(query.encode()).hexdigest()

    def geocode(self, address: str) -> Optional[GeoLocation]:
        """
        地址转坐标

        Args:
            address: 地址字符串

        Returns:
            GeoLocation对象，失败返回None
        """
        if not address:
            return None

        # 精确匹配
        addr_key = address.strip()
        if addr_key in self.MOCK_LOCATIONS:
            return self.MOCK_LOCATIONS[addr_key]

        # 模糊匹配 - 遍历查找包含关键字的
        for key, loc in self.MOCK_LOCATIONS.items():
            if key in addr_key or addr_key in key:
                logger.debug(f"Mock geocode模糊匹配: {addr_key} -> {key}")
                return loc

        # 尝试提取关键城市/景区名
        keywords = ["桂林", "阳朔", "龙胜", "兴安", "荔浦", "漓江", "象鼻山", "两江四湖"]
        for kw in keywords:
            if kw in addr_key:
                # 返回关键词对应的位置
                if kw in self.MOCK_LOCATIONS:
                    return self.MOCK_LOCATIONS[kw]

        logger.warning(f"Mock geocode未找到: {address}")
        return None

    def reverse_geocode(self, lat: float, lng: float) -> Optional[GeoLocation]:
        """
        坐标转地址

        Args:
            lat: 纬度
            lng: 经度

        Returns:
            GeoLocation对象，失败返回None
        """
        # 在预定义位置中查找最近的
        min_dist = float('inf')
        nearest = None

        for name, loc in self.MOCK_LOCATIONS.items():
            dist = ((lat - loc.lat) ** 2 + (lng - loc.lng) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest = loc

        # 如果距离小于0.1度，认为匹配成功
        if min_dist < 0.1:
            return nearest

        # 返回一个近似位置
        if nearest:
            return GeoLocation(
                lat=lat,
                lng=lng,
                province=nearest.province,
                city=nearest.city,
                district=nearest.district,
                address=f"附近({lat:.4f}, {lng:.4f})",
                name="未知位置"
            )

        return None


class AmapGeocoder:
    """
    高德地图地理编码器

    使用高德地图Web服务API进行地理编码
    API文档: https://lbs.amap.com/api/webservice/guide/api/georegeo
    """

    AMAP_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"
    AMAP_REGEOCODE_URL = "https://restapi.amap.com/v3/geocode/regeo"

    def __init__(self, config: Optional[GeocoderConfig] = None):
        self.config = config or GeocoderConfig()
        self._cache: Dict[str, GeoLocation] = {}
        self._session = None

    def _get_session(self):
        """获取HTTP会话"""
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def _get_cache_key(self, query: str) -> str:
        """生成缓存key"""
        return hashlib.md5(query.encode()).hexdigest()

    def _parse_geocode_result(self, result: Dict[str, Any]) -> Optional[GeoLocation]:
        """解析地理编码响应"""
        try:
            if result.get("status") != "1":
                logger.warning(f"高德地理编码失败: {result.get('info', 'unknown')}")
                return None

            geocodes = result.get("geocodes", [])
            if not geocodes:
                return None

            geo = geocodes[0]
            location = geo.get("location", "").split(",")
            if len(location) != 2:
                return None

            return GeoLocation(
                lat=float(location[1]),
                lng=float(location[0]),
                province=geo.get("province", ""),
                city=geo.get("city", ""),
                district=geo.get("district", ""),
                address=geo.get("formatted_address", ""),
                name=geo.get("building", ""),
            )
        except Exception as e:
            logger.error(f"解析高德地理编码结果失败: {e}")
            return None

    def _parse_regeocode_result(self, result: Dict[str, Any]) -> Optional[GeoLocation]:
        """解析逆地理编码响应"""
        try:
            if result.get("status") != "1":
                logger.warning(f"高德逆地理编码失败: {result.get('info', 'unknown')}")
                return None

            regeocode = result.get("regeocode", {})
            if not regeocode:
                return None

            address_component = regeocode.get("addressComponent", {})
            location = regeocode.get("location", "").split(",")
            if len(location) != 2:
                return None

            return GeoLocation(
                lat=float(location[1]),
                lng=float(location[0]),
                province=address_component.get("province", ""),
                city=address_component.get("city", ""),
                district=address_component.get("district", ""),
                address=regeocode.get("formatted_address", ""),
                name="",
            )
        except Exception as e:
            logger.error(f"解析高德逆地理编码结果失败: {e}")
            return None

    def geocode(self, address: str) -> Optional[GeoLocation]:
        """
        地址转坐标

        Args:
            address: 地址字符串

        Returns:
            GeoLocation对象，失败返回None
        """
        if not address:
            return None

        if not self.config.amap_key:
            logger.error("高德API Key未配置")
            return None

        # 检查缓存
        cache_key = self._get_cache_key(address)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            import requests

            params = {
                "key": self.config.amap_key,
                "address": address,
                "output": "json",
            }

            # 如果有安全密钥，添加sig
            if self.config.amap_secret:
                import hashlib
                import urllib.parse
                sorted_params = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
                sig_str = sorted_params + self.config.amap_secret
                sig = hashlib.md5(sig_str.encode()).hexdigest()
                params["sig"] = sig

            session = self._get_session()
            response = session.get(
                self.AMAP_GEOCODE_URL,
                params=params,
                timeout=self.config.timeout
            )
            result = response.json()

            location = self._parse_geocode_result(result)
            if location:
                self._cache[cache_key] = location

            return location

        except Exception as e:
            logger.error(f"高德地理编码请求失败: {e}")
            return None

    def reverse_geocode(self, lat: float, lng: float) -> Optional[GeoLocation]:
        """
        坐标转地址

        Args:
            lat: 纬度
            lng: 经度

        Returns:
            GeoLocation对象，失败返回None
        """
        if not self.config.amap_key:
            logger.error("高德API Key未配置")
            return None

        # 检查缓存
        cache_key = self._get_cache_key(f"{lat},{lng}")
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            import requests

            params = {
                "key": self.config.amap_key,
                "location": f"{lng},{lat}",  # 高德API格式：经度,纬度
                "output": "json",
                "extensions": "all",
            }

            # 如果有安全密钥，添加sig
            if self.config.amap_secret:
                import hashlib
                sorted_params = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
                sig_str = sorted_params + self.config.amap_secret
                sig = hashlib.md5(sig_str.encode()).hexdigest()
                params["sig"] = sig

            session = self._get_session()
            response = session.get(
                self.AMAP_REGEOCODE_URL,
                params=params,
                timeout=self.config.timeout
            )
            result = response.json()

            location = self._parse_regeocode_result(result)
            if location:
                self._cache[cache_key] = location

            return location

        except Exception as e:
            logger.error(f"高德逆地理编码请求失败: {e}")
            return None


class GeocoderAdapter:
    """
    地理编码适配器

    统一接口，支持高德地图API和模拟数据两种模式
    """

    def __init__(self, config: Optional[GeocoderConfig] = None):
        self.config = config or GeocoderConfig()
        self._geocoder: Optional[Any] = None
        self._initialized = False

    def initialize(self) -> bool:
        """初始化地理编码器"""
        if self._initialized:
            return True

        try:
            mode = GeocoderMode(self.config.mode)
        except ValueError:
            logger.warning(f"未知的地理编码器模式: {self.config.mode}，使用mock模式")
            mode = GeocoderMode.MOCK

        if mode == GeocoderMode.AMAP:
            self._geocoder = AmapGeocoder(self.config)
            logger.info("使用高德地图地理编码器")
        else:
            self._geocoder = MockGeocoder(self.config)
            logger.info("使用模拟地理编码器")

        self._initialized = True
        return True

    def geocode(self, address: str) -> Optional[GeoLocation]:
        """
        地址转坐标

        Args:
            address: 地址字符串

        Returns:
            GeoLocation对象，失败返回None
        """
        if not self._initialized:
            self.initialize()
        return self._geocoder.geocode(address)

    def reverse_geocode(self, lat: float, lng: float) -> Optional[GeoLocation]:
        """
        坐标转地址

        Args:
            lat: 纬度
            lng: 经度

        Returns:
            GeoLocation对象，失败返回None
        """
        if not self._initialized:
            self.initialize()
        return self._geocoder.reverse_geocode(lat, lng)

    def batch_geocode(self, addresses: List[str]) -> List[Optional[GeoLocation]]:
        """
        批量地址转坐标

        Args:
            addresses: 地址列表

        Returns:
            GeoLocation对象列表
        """
        if not self._initialized:
            self.initialize()

        return [self.geocode(addr) for addr in addresses]

    def batch_reverse_geocode(self, locations: List[Tuple[float, float]]) -> List[Optional[GeoLocation]]:
        """
        批量坐标转地址

        Args:
            locations: [(lat, lng), ...] 坐标列表

        Returns:
            GeoLocation对象列表
        """
        if not self._initialized:
            self.initialize()

        return [self.reverse_geocode(lat, lng) for lat, lng in locations]

    def close(self):
        """关闭连接/清理资源"""
        self._geocoder = None
        self._initialized = False


# =============================================================================
# 工厂函数
# =============================================================================

def create_geocoder_adapter(
    mode: str = "mock",
    amap_key: Optional[str] = None,
    amap_secret: Optional[str] = None,
    **kwargs,
) -> GeocoderAdapter:
    """
    创建地理编码适配器

    Args:
        mode: 模式 ("amap" 或 "mock")
        amap_key: 高德地图API Key
        amap_secret: 高德地图安全密钥
        **kwargs: 其他配置参数

    Returns:
        GeocoderAdapter实例
    """
    config = GeocoderConfig(
        mode=mode,
        amap_key=amap_key or os.environ.get("AMAP_KEY", ""),
        amap_secret=amap_secret or os.environ.get("AMAP_SECRET", ""),
        **kwargs,
    )
    return GeocoderAdapter(config)


# =============================================================================
# 便捷函数
# =============================================================================

_default_geocoder: Optional[GeocoderAdapter] = None


def get_default_geocoder() -> GeocoderAdapter:
    """获取默认地理编码器实例"""
    global _default_geocoder
    if _default_geocoder is None:
        _default_geocoder = create_geocoder_adapter()
        _default_geocoder.initialize()
    return _default_geocoder


def geocode(address: str) -> Optional[GeoLocation]:
    """便捷函数：地址转坐标"""
    return get_default_geocoder().geocode(address)


def reverse_geocode(lat: float, lng: float) -> Optional[GeoLocation]:
    """便捷函数：坐标转地址"""
    return get_default_geocoder().reverse_geocode(lat, lng)
