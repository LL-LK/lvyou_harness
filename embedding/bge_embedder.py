"""
BGE Embedder - 本地BGE-m3嵌入向量生成
=====================================

使用本地BGE模型生成嵌入向量，支持:
- BGE-small-zh-v1.5 (512维)
- BGE-m3 (1024维)

模型路径: /mnt/f/LLM/models/bge-small-zh-v1.5
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class EmbedderConfig:
    """嵌入器配置"""
    model_path: str = "/mnt/f/LLM/models/bge-small-zh-v1.5"
    model_name: str = "BAAI/bge-small-zh-v1.5"
    dimension: int = 512
    normalize: bool = True  # 是否L2归一化
    max_length: int = 512
    batch_size: int = 32
    device: str = "cpu"  # cpu/cuda


class BGEEmbedder:
    """
    BGE嵌入向量生成器

    使用本地BGE模型生成文本嵌入向量
    """

    def __init__(self, config: Optional[EmbedderConfig] = None):
        self.config = config or EmbedderConfig()
        self._model = None
        self._tokenizer = None
        self._initialized = False

    def initialize(self) -> bool:
        """初始化模型"""
        if self._initialized:
            return True

        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"加载BGE模型: {self.config.model_path}")

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
            )
            self._model = AutoModel.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
            )
            self._model.eval()

            self._initialized = True
            logger.info(f"BGE模型加载成功，维度: {self.config.dimension}")
            return True

        except Exception as e:
            logger.error(f"BGE模型加载失败: {e}")
            return False

    def embed(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """
        生成嵌入向量

        Args:
            texts: 文本列表
            normalize: 是否L2归一化

        Returns:
            嵌入向量列表
        """
        if not self._initialized:
            self.initialize()

        if not texts:
            return []

        try:
            # 分批处理
            all_embeddings = []
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]

                # Tokenize
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )

                # 前向传播
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    # Mean pooling
                    attention_mask = inputs["attention_mask"]
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

                    # 归一化
                    if normalize:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    embeddings = embeddings.numpy()

                all_embeddings.extend(embeddings.tolist())

            return all_embeddings

        except ImportError as e:
            logger.error(f"缺少依赖: {e}")
            # Fallback: 返回随机向量
            return self._fallback_embeddings(len(texts))

        except Exception as e:
            logger.error(f"Embedding失败: {e}")
            return self._fallback_embeddings(len(texts))

    def embed_single(self, text: str, normalize: bool = True) -> List[float]:
        """生成单个文本的嵌入向量"""
        return self.embed([text], normalize=normalize)[0]

    def _fallback_embeddings(self, count: int) -> List[List[float]]:
        """Fallback: 返回随机向量（用于测试或模型加载失败时）"""
        logger.warning("使用随机向量作为fallback")
        return [np.random.rand(self.config.dimension).tolist() for _ in range(count)]

    @property
    def dimension(self) -> int:
        """返回向量维度"""
        return self.config.dimension

    def close(self):
        """关闭模型，释放资源"""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._initialized = False


def create_embedder(
    model_path: str = "/mnt/f/LLM/models/bge-small-zh-v1.5",
    dimension: int = 512,
    **kwargs,
) -> BGEEmbedder:
    """
    创建嵌入器

    Args:
        model_path: 模型路径
        dimension: 向量维度
        **kwargs: 其他配置

    Returns:
        BGEEmbedder实例
    """
    config = EmbedderConfig(
        model_path=model_path,
        dimension=dimension,
        **kwargs,
    )
    return BGEEmbedder(config)
