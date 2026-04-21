#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: 基础评估器抽象类，提供评估器的基类接口
日期: 2026年1月23日
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from collections import defaultdict
import time


class BaseEvaluator(ABC):
    """
    基础评估器抽象类

    所有评估器应继承此类并实现必要的方法

    Attributes:
        metric_list: 要计算的指标列表
        class_names: 类别名称列表
        ignore_index: 忽略的类别索引
    """

    def __init__(
        self,
        metric_list: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        ignore_index: int = 255,
    ):
        """
        初始化基础评估器

        Args:
            metric_list: 要计算的指标列表，如 ['IoU', 'mIoU', 'Accuracy']
            class_names: 类别名称列表
            ignore_index: 忽略的像素类别索引
        """
        self.metric_list = metric_list or ['mIoU', 'Accuracy']
        self.class_names = class_names or []
        self.ignore_index = ignore_index
        self._reset()

    def _reset(self):
        """
        重置评估器状态
        """
        self.total_samples = 0
        self.total_time = 0.0
        self.results = {}

    def reset(self):
        """
        重置评估器状态
        """
        self._reset()

    def add_predictions(
        self,
        predictions: Any,
        targets: Any,
    ):
        """
        添加预测结果和标签

        Args:
            predictions: 模型预测结果
            targets: 真实标签
        """
        start_time = time.time()
        self._add_predictions(predictions, targets)
        self.total_time += time.time() - start_time
        self.total_samples += self._get_batch_size(targets)

    @abstractmethod
    def _add_predictions(
        self,
        predictions: Any,
        targets: Any,
    ):
        """
        内部方法：添加预测结果和标签

        Args:
            predictions: 模型预测结果
            targets: 真实标签
        """
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """
        计算并返回所有指标

        Returns:
            metrics: 包含各指标的字典
        """
        pass

    def get_metric_table(self) -> str:
        """
        获取格式化的指标表格

        Returns:
            str: 格式化的表格字符串
        """
        metrics = self.evaluate()
        lines = []
        lines.append("=" * 50)
        lines.append("Evaluation Results")
        lines.append("=" * 50)

        for key, value in metrics.items():
            if isinstance(value, dict):
                lines.append(f"\n{key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"  {sub_key}: {sub_value:.4f}")
            else:
                lines.append(f"{key}: {value:.4f}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def _get_batch_size(self, targets: Any) -> int:
        """
        获取批次大小

        Args:
            targets: 目标张量

        Returns:
            int: 批次大小
        """
        if hasattr(targets, 'shape'):
            return targets.shape[0]
        return 1

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"metrics={self.metric_list}, "
            f"samples={self.total_samples}"
            f")"
        )
