"""
LvyouHarness 配置管理
========================
旅游领域Harness的默认配置、RAG连接参数、Agent角色定义。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import os


@dataclass
class LvyouHarnessConfig:
    """旅游Harness统一配置"""

    # --- 基础路径 ---
    workspace: Path = Path("/home/l2140/lvyou_harness/workspace")
    data_dir: Path = Path("/home/l2140/lvyou_harness/data")

    # --- RAG配置 ---
    rag_harness_path: Path = Path("/home/l2140/RAG-Harness")
    collection_name: str = "lvyou_guilin"  # 默认旅游知识库集合
    embedding_model: str = "BAAI/bge-m3"
    top_k: int = 5
    reranker_model: str = "BAAI/bge-reranker-base"

    # --- LLM配置 ---
    llm_provider: str = "minimax-cn"
    llm_model: str = "MiniMax-M2.7"
    llm_api_key: Optional[str] = None
    llm_timeout: int = 120

    # --- Agent配置 ---
    max_parallel_agents: int = 3
    enable_rag: bool = True
    enable_multimodal: bool = True  # 支持景点图片理解

    # --- 行程规划参数 ---
    default_days: int = 3
    max_daily_spots: int = 6
    preferred_start_time: str = "08:00"
    preferred_end_time: str = "20:00"
    transfer_time_minutes: int = 30  # 景点间转移时间

    # --- 预算参数 ---
    default_currency: str = "CNY"
    budget_per_day: float = 500.0
    include_food: bool = True
    include_accommodation: bool = True

    # --- 知识库覆盖 ---
    scenic_regions: List[str] = field(default_factory=lambda: [
        "桂林", "阳朔", "龙脊梯田", "漓江", "象山", "两江四湖",
        "北海", "涠洲岛", "南宁", "柳州", "玉林", "梧州"
    ])

    @classmethod
    def for_guilin(cls) -> "LvyouHarnessConfig":
        """桂林旅游专用配置"""
        cfg = cls()
        cfg.collection_name = "lvyou_guilin"
        cfg.scenic_regions = ["桂林", "阳朔", "龙脊梯田", "漓江", "象山", "两江四湖"]
        cfg.default_days = 4
        cfg.budget_per_day = 600.0
        return cfg

    @classmethod
    def for_beibuwan(cls) -> "LvyouHarnessConfig":
        """北部湾(北海/涠洲岛)专用配置"""
        cfg = cls()
        cfg.collection_name = "lvyou_beibuwan"
        cfg.scenic_regions = ["北海", "涠洲岛", "银滩", "侨港", "冠头岭"]
        cfg.default_days = 3
        cfg.budget_per_day = 550.0
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace": str(self.workspace),
            "data_dir": str(self.data_dir),
            "rag": {
                "collection": self.collection_name,
                "embedding": self.embedding_model,
                "reranker": self.reranker_model,
                "top_k": self.top_k,
            },
            "llm": {
                "provider": self.llm_provider,
                "model": self.llm_model,
                "timeout": self.llm_timeout,
            },
            "agent": {
                "max_parallel": self.max_parallel_agents,
                "enable_rag": self.enable_rag,
            },
            "route": {
                "default_days": self.default_days,
                "max_daily_spots": self.max_daily_spots,
                "transfer_time_minutes": self.transfer_time_minutes,
            },
            "budget": {
                "currency": self.default_currency,
                "per_day": self.budget_per_day,
            },
        }

    def ensure_workspace(self) -> None:
        """确保工作目录存在"""
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


# 环境变量覆盖
def load_from_env(cfg: LvyouHarnessConfig) -> LvyouHarnessConfig:
    """从环境变量覆盖配置"""
    if os.getenv("LVYOU_RAG_COLLECTION"):
        cfg.collection_name = os.getenv("LVYOU_RAG_COLLECTION")
    if os.getenv("LVYOU_LLM_MODEL"):
        cfg.llm_model = os.getenv("LVYOU_LLM_MODEL")
    if os.getenv("LVYOU_API_KEY"):
        cfg.llm_api_key = os.getenv("LVYOU_API_KEY")
    if os.getenv("LVYOU_BUDGET_DAY"):
        cfg.budget_per_day = float(os.getenv("LVYOU_BUDGET_DAY"))
    return cfg
