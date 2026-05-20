"""
LvyouUtils - 旅游领域通用工具
==================================

提供:
- 日志配置
- 文件操作
- 格式化输出
- 知识库操作
"""
from __future__ import annotations

import os
import sys
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict


# =============================================================================
# 日志配置
# =============================================================================

def setup_logging(
    name: str = "lvyou",
    level: int = logging.INFO,
    log_file: str = None,
) -> logging.Logger:
    """
    配置日志

    Args:
        name: logger名称
        level: 日志级别
        log_file: 日志文件路径 (可选)

    Returns:
        配置好的logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 格式化
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # 文件handler (可选)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


# =============================================================================
# 文件操作
# =============================================================================

def ensure_dir(path: str | Path) -> Path:
    """确保目录存在"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Dict[str, Any]:
    """读取JSON文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Dict, indent: int = 2) -> None:
    """写入JSON文件"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_text(path: str | Path) -> str:
    """读取文本文件"""
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, content: str) -> None:
    """写入文本文件"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def append_text(path: str | Path, content: str) -> None:
    """追加文本到文件"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(content)


# =============================================================================
# 格式化输出
# =============================================================================

def format_currency(amount: float, currency: str = "CNY") -> str:
    """格式化货币"""
    if currency == "CNY":
        return f"¥{amount:.2f}"
    elif currency == "USD":
        return f"${amount:.2f}"
    elif currency == "EUR":
        return f"€{amount:.2f}"
    else:
        return f"{amount:.2f} {currency}"


def format_duration(minutes: int) -> str:
    """格式化时长"""
    if minutes < 60:
        return f"{minutes}分钟"
    hours = minutes // 60
    mins = minutes % 60
    if mins == 0:
        return f"{hours}小时"
    return f"{hours}小时{mins}分钟"


def format_distance(km: float) -> str:
    """格式化距离"""
    if km < 1:
        return f"{km * 1000:.0f}米"
    return f"{km:.1f}公里"


def format_time(hour: int, minute: int = 0) -> str:
    """格式化时间"""
    return f"{hour:02d}:{minute:02d}"


def format_date(date: datetime, fmt: str = "%Y-%m-%d") -> str:
    """格式化日期"""
    return date.strftime(fmt)


def parse_date(date_str: str) -> datetime:
    """解析日期字符串"""
    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d", "%m月%d日"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"无法解析日期: {date_str}")


# =============================================================================
# 文本处理
# =============================================================================

def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """截断文本"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """清理文本"""
    import re
    # 移除多余空白
    text = re.sub(r"\s+", " ", text)
    # 移除特殊字符
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    return text.strip()


def md5(text: str) -> str:
    """计算MD5"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# =============================================================================
# 数据验证
# =============================================================================

def validate_days(days: Any) -> int:
    """验证天数"""
    try:
        d = int(days)
        if d < 1 or d > 30:
            raise ValueError("天数必须在1-30之间")
        return d
    except (TypeError, ValueError) as e:
        raise ValueError(f"无效天数: {days}") from e


def validate_budget(budget: Any) -> float:
    """验证预算"""
    try:
        b = float(budget)
        if b < 0:
            raise ValueError("预算不能为负")
        return b
    except (TypeError, ValueError) as e:
        raise ValueError(f"无效预算: {budget}") from e


# =============================================================================
# 缓存装饰器
# =============================================================================

def simple_cache(ttl_seconds: int = 300):
    """
    简单缓存装饰器 (基于时间)

    用于RAG查询等不需要精确缓存的场景
    """
    cache: Dict[str, tuple[Any, float]] = {}

    def decorator(func):
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            now = time.time()

            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < ttl_seconds:
                    return result

            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result

        return wrapper

    return decorator


# =============================================================================
# 时间工具
# =============================================================================

def get_today() -> datetime:
    """获取今天日期"""
    return datetime.now()


def add_days(date: datetime, days: int) -> datetime:
    """日期加减"""
    return date + timedelta(days=days)


def get_date_range(start: datetime, days: int) -> List[datetime]:
    """获取日期范围"""
    return [start + timedelta(days=i) for i in range(days)]


# =============================================================================
# 配置合并
# =============================================================================

def merge_config(base: Dict, override: Dict) -> Dict:
    """合并配置字典"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


# =============================================================================
# 版本信息
# =============================================================================

VERSION = "1.0.0"
AUTHOR = "LvyouHarness Team"


def get_version() -> str:
    """获取版本"""
    return VERSION


# 延迟导入time模块 (避免命名冲突)
import time
