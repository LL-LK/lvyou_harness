"""
汇率适配器
==========

支持汇率API和模拟数据的适配器

功能:
1. 真实汇率API调用 (exchangerate-api.com)
2. 模拟数据模式 (用于测试/开发)
3. 多种货币对转换
"""
from __future__ import annotations

import os
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# =============================================================================
# 汇率端口接口
# =============================================================================

class ExchangeRatePort(ABC):
    """
    汇率获取端口接口

    所有汇率实现必须实现此接口:
    - 真实API调用
    - 模拟数据
    """

    @abstractmethod
    def initialize(self) -> bool:
        """初始化连接"""
        ...

    @abstractmethod
    def get_exchange_rate(
        self,
        from_currency: str,
        to_currency: str,
    ) -> Optional[float]:
        """
        获取汇率

        Args:
            from_currency: 源货币代码 (如 "USD")
            to_currency: 目标货币代码 (如 "CNY")

        Returns:
            汇率值，如果获取失败返回None
        """
        ...

    @abstractmethod
    def convert(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
    ) -> Optional[float]:
        """
        货币转换

        Args:
            amount: 金额
            from_currency: 源货币代码
            to_currency: 目标货币代码

        Returns:
            转换后的金额，失败返回None
        """
        ...

    @abstractmethod
    def get_all_rates(self, base_currency: str = "USD") -> Optional[Dict[str, float]]:
        """
        获取相对于基准货币的所有汇率

        Args:
            base_currency: 基准货币代码

        Returns:
            货币->汇率的字典，失败返回None
        """
        ...

    @abstractmethod
    def get_supported_currencies(self) -> List[str]:
        """获取支持的货币列表"""
        ...


# =============================================================================
# 配置
# =============================================================================

@dataclass
class ExchangeRateConfig:
    """汇率配置"""
    # API模式: "api" (真实API) 或 "mock" (模拟数据)
    mode: str = os.environ.get("EXCHANGE_RATE_MODE", "mock")

    # 真实API配置
    api_key: str = os.environ.get("EXCHANGE_RATE_API_KEY", "")
    api_base_url: str = os.environ.get(
        "EXCHANGE_RATE_API_URL",
        "https://api.exchangerate-api.com/v4/latest"
    )

    # 缓存配置 (秒)
    cache_ttl: int = 3600  # 默认1小时缓存

    # API请求超时 (秒)
    request_timeout: int = 10

    # 模拟数据配置
    mock_rates: Dict[str, float] = field(default_factory=lambda: {
        "USD": 1.0,
        "CNY": 7.24,
        "EUR": 0.92,
        "GBP": 0.79,
        "JPY": 149.50,
        "HKD": 7.82,
        "KRW": 1320.0,
        "SGD": 1.34,
        "THB": 35.20,
        "AUD": 1.53,
        "CAD": 1.36,
        "CHF": 0.88,
        "NZD": 1.64,
        "MYR": 4.72,
        "PHP": 56.50,
        "VND": 24500.0,
        "TWD": 31.50,
        "MOP": 8.08,
    })


# =============================================================================
# 缓存助手
# =============================================================================

class RateCache:
    """简单的时间戳缓存"""

    def __init__(self, ttl: int = 3600):
        self._cache: Dict[str, tuple[float, float]] = {}  # key -> (rate, timestamp)
        self._ttl = ttl

    def get(self, key: str) -> Optional[float]:
        if key in self._cache:
            rate, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return rate
            del self._cache[key]
        return None

    def set(self, key: str, rate: float):
        self._cache[key] = (rate, time.time())

    def clear(self):
        self._cache.clear()


# =============================================================================
# 真实API适配器
# =============================================================================

class ExchangeRateAPIAdapter(ExchangeRatePort):
    """
    真实汇率API适配器

    使用 exchangerate-api.com 的免费API
    """

    def __init__(self, config: Optional[ExchangeRateConfig] = None):
        self.config = config or ExchangeRateConfig()
        self._cache = RateCache(ttl=self.config.cache_ttl)
        self._initialized = False

    def initialize(self) -> bool:
        """初始化API适配器"""
        if self._initialized:
            return True

        # 验证配置
        if not self.config.api_base_url:
            logger.warning("未配置API URL，使用模拟模式")
            return False

        self._initialized = True
        logger.info("汇率API适配器初始化成功")
        return True

    def get_exchange_rate(
        self,
        from_currency: str,
        to_currency: str,
    ) -> Optional[float]:
        """获取汇率"""
        self.ensure_initialized()

        if from_currency == to_currency:
            return 1.0

        # 尝试从缓存获取
        cache_key = f"{from_currency}_{to_currency}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug(f"使用缓存汇率: {cache_key}={cached}")
            return cached

        # 获取所有汇率然后计算
        all_rates = self.get_all_rates(from_currency)
        if all_rates is None:
            return None

        rate = all_rates.get(to_currency)
        if rate is not None:
            self._cache.set(cache_key, rate)

        return rate

    def convert(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
    ) -> Optional[float]:
        """货币转换"""
        rate = self.get_exchange_rate(from_currency, to_currency)
        if rate is None:
            return None
        return amount * rate

    def get_all_rates(self, base_currency: str = "USD") -> Optional[Dict[str, float]]:
        """获取基准货币的所有汇率"""
        self.ensure_initialized()

        try:
            import requests

            url = f"{self.config.api_base_url}/{base_currency}"
            logger.info(f"请求汇率API: {url}")

            response = requests.get(
                url,
                timeout=self.config.request_timeout,
            )
            response.raise_for_status()

            data = response.json()
            rates = data.get("rates", {})

            logger.info(f"成功获取{base_currency}的{len(rates)}种货币汇率")
            return rates

        except ImportError:
            logger.error("需要安装 requests 库: pip install requests")
            return None
        except Exception as e:
            logger.error(f"获取汇率失败: {e}")
            return None

    def get_supported_currencies(self) -> List[str]:
        """获取支持的货币列表"""
        # 常用货币列表
        return [
            "USD", "CNY", "EUR", "GBP", "JPY", "HKD", "KRW", "SGD",
            "THB", "AUD", "CAD", "CHF", "NZD", "MYR", "PHP", "VND",
            "TWD", "MOP", "AED", "SAR", "INR", "RUB", "BRL", "ZAR",
        ]

    def ensure_initialized(self):
        """确保已初始化"""
        if not self._initialized:
            self.initialize()


# =============================================================================
# 模拟数据适配器
# =============================================================================

class MockExchangeRateAdapter(ExchangeRatePort):
    """
    模拟汇率适配器

    用于测试和开发环境
    """

    def __init__(self, config: Optional[ExchangeRateConfig] = None):
        self.config = config or ExchangeRateConfig()
        self._cache = RateCache(ttl=self.config.cache_ttl)
        self._initialized = False

    def initialize(self) -> bool:
        """初始化模拟适配器"""
        if self._initialized:
            return True

        logger.info(f"模拟汇率适配器初始化成功，使用以下汇率:")
        for currency, rate in sorted(self.config.mock_rates.items()):
            logger.info(f"  1 USD = {rate} {currency}")

        self._initialized = True
        return True

    def get_exchange_rate(
        self,
        from_currency: str,
        to_currency: str,
    ) -> Optional[float]:
        """获取汇率"""
        self.ensure_initialized()

        if from_currency == to_currency:
            return 1.0

        from_upper = from_currency.upper()
        to_upper = to_currency.upper()

        # 检查是否支持
        if from_upper not in self.config.mock_rates:
            logger.warning(f"不支持的货币: {from_currency}")
            return None
        if to_upper not in self.config.mock_rates:
            logger.warning(f"不支持的货币: {to_currency}")
            return None

        # 转换为 USD 再转换为目标货币
        # 汇率定义为: 1 USD = X 货币
        from_to_usd = 1.0 / self.config.mock_rates[from_upper]  # X -> USD
        usd_to_to = self.config.mock_rates[to_upper]  # USD -> Y
        rate = from_to_usd * usd_to_to

        logger.debug(f"模拟汇率: 1 {from_currency} = {rate} {to_currency}")
        return rate

    def convert(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
    ) -> Optional[float]:
        """货币转换"""
        rate = self.get_exchange_rate(from_currency, to_currency)
        if rate is None:
            return None
        return amount * rate

    def get_all_rates(self, base_currency: str = "USD") -> Optional[Dict[str, float]]:
        """获取基准货币的所有汇率"""
        self.ensure_initialized()

        base_upper = base_currency.upper()
        if base_upper not in self.config.mock_rates:
            logger.warning(f"不支持的基准货币: {base_currency}")
            return None

        base_rate = self.config.mock_rates[base_upper]
        result = {}

        for currency, rate in self.config.mock_rates.items():
            # 转换: base_currency -> currency
            result[currency] = rate / base_rate

        return result

    def get_supported_currencies(self) -> List[str]:
        """获取支持的货币列表"""
        return list(self.config.mock_rates.keys())

    def ensure_initialized(self):
        """确保已初始化"""
        if not self._initialized:
            self.initialize()

    def set_mock_rate(self, currency: str, rate: float):
        """
        设置模拟汇率 (用于测试)

        Args:
            currency: 货币代码
            rate: 1 USD = rate 该货币
        """
        self.config.mock_rates[currency.upper()] = rate
        self._cache.clear()
        logger.info(f"设置模拟汇率: 1 USD = {rate} {currency}")


# =============================================================================
# 工厂函数
# =============================================================================

def create_exchange_rate_adapter(
    adapter_type: str = "mock",
    **kwargs,
) -> ExchangeRatePort:
    """
    创建汇率适配器

    Args:
        adapter_type: 适配器类型 ("api", "mock")
        **kwargs: 传递给适配器的配置

    Returns:
        ExchangeRatePort接口实例
    """
    # 如果指定了环境变量覆盖
    mode = os.environ.get("EXCHANGE_RATE_MODE", adapter_type)

    if mode == "api":
        config_data = {}
        for key, default in [
            ("api_key", ""),
            ("api_base_url", "https://api.exchangerate-api.com/v4/latest"),
            ("cache_ttl", 3600),
            ("request_timeout", 10),
        ]:
            config_data[key] = kwargs.get(key, os.environ.get(f"EXCHANGE_RATE_{key.upper()}", default))

        config = ExchangeRateConfig(mode="api", **config_data)
        return ExchangeRateAPIAdapter(config)

    else:  # mock
        # 支持通过kwargs传入自定义模拟汇率
        mock_rates = kwargs.get("mock_rates")
        config = ExchangeRateConfig(mode="mock")

        if mock_rates:
            config.mock_rates.update(mock_rates)

        return MockExchangeRateAdapter(config)


# =============================================================================
# 便捷函数
# =============================================================================

def get_exchange_rate(
    from_currency: str,
    to_currency: str,
    adapter_type: str = "mock",
    **kwargs,
) -> Optional[float]:
    """
    便捷函数: 获取汇率

    Args:
        from_currency: 源货币
        to_currency: 目标货币
        adapter_type: 适配器类型
        **kwargs: 其他配置

    Returns:
        汇率值
    """
    adapter = create_exchange_rate_adapter(adapter_type, **kwargs)
    return adapter.get_exchange_rate(from_currency, to_currency)


def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
    adapter_type: str = "mock",
    **kwargs,
) -> Optional[float]:
    """
    便捷函数: 货币转换

    Args:
        amount: 金额
        from_currency: 源货币
        to_currency: 目标货币
        adapter_type: 适配器类型
        **kwargs: 其他配置

    Returns:
        转换后的金额
    """
    adapter = create_exchange_rate_adapter(adapter_type, **kwargs)
    return adapter.convert(amount, from_currency, to_currency)


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 60)
    print("汇率适配器测试")
    print("=" * 60)

    # 1. 模拟模式测试
    print("\n[1] 模拟模式测试")
    mock_adapter = create_exchange_rate_adapter("mock")
    mock_adapter.initialize()

    print(f"支持的货币: {mock_adapter.get_supported_currencies()}")
    print(f"USD -> CNY 汇率: {mock_adapter.get_exchange_rate('USD', 'CNY')}")
    print(f"100 USD -> CNY: {mock_adapter.convert(100, 'USD', 'CNY')}")
    print(f"1000 JPY -> USD: {mock_adapter.convert(1000, 'JPY', 'USD')}")

    # 2. 设置自定义汇率测试
    print("\n[2] 自定义汇率测试")
    mock_adapter.set_mock_rate("BTC", 0.000016)  # 1 USD = 0.000016 BTC
    print(f"1000 USD -> BTC: {mock_adapter.convert(1000, 'USD', 'BTC')}")

    # 3. API模式测试 (需要网络)
    print("\n[3] API模式测试 (如配置了API Key)")
    try:
        api_adapter = create_exchange_rate_adapter("api")
        if api_adapter.initialize():
            print(f"USD -> EUR 汇率: {api_adapter.get_exchange_rate('USD', 'EUR')}")
        else:
            print("API初始化失败，使用模拟模式")
    except Exception as e:
        print(f"API模式错误: {e}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
