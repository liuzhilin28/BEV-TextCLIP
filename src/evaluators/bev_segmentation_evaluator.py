#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: BEV 分割评估器，提供 BEV 分割任务的评估指标计算
日期: 2026年1月23日
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np

from .base_evaluator import BaseEvaluator


class BEVSegmentationEvaluator(BaseEvaluator):
    """
    BEV 分割评估器

    计算 BEV 分割任务的各类评估指标

    Supported Metrics:
        - IoU: 各类别的 Intersection over Union
        - mIoU: 平均 IoU
        - Accuracy: 像素准确率
        - Precision: 精确率
        - Recall: 召回率
        - F1: F1 分数
        - Confusion Matrix: 混淆矩阵
    """

    def __init__(
        self,
        num_classes: int,
        metric_list: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        ignore_index: int = 255,
        device: str = 'cpu',
    ):
        """
        初始化 BEV 分割评估器

        Args:
            num_classes: 类别数量
            metric_list: 要计算的指标列表
            class_names: 类别名称列表
            ignore_index: 忽略的类别索引
            device: 计算设备
        """
        class_names = class_names or [f'class_{i}' for i in range(num_classes)]

        self.num_classes = num_classes
        self.device = device
        self.confusion_matrix: torch.Tensor = torch.zeros(
            self.num_classes, self.num_classes, device=self.device
        )

        super().__init__(metric_list, class_names, ignore_index)

    def _init_confusion_matrix(self):
        """
        初始化混淆矩阵
        """
        self.confusion_matrix = torch.zeros(
            self.num_classes, self.num_classes, device=self.device
        )

    def _reset(self):
        """
        重置评估器状态
        """
        super()._reset()
        self.confusion_matrix = torch.zeros(
            self.num_classes, self.num_classes, device=self.device
        )

    def _add_predictions(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        添加预测结果和标签到混淆矩阵

        Args:
            predictions: 预测结果 [B, H, W] 或 [B, C, H, W]
            targets: 真实标签 [B, H, W]
        """
        if predictions.dim() == 4:
            predictions = predictions.argmax(dim=1)

        predictions = predictions.long()
        targets = targets.long()

        mask = (targets != self.ignore_index)
        predictions = predictions[mask]
        targets = targets[mask]

        if targets.numel() == 0:
            return

        valid_mask = (targets >= 0) & (targets < self.num_classes) & (predictions >= 0) & (predictions < self.num_classes)
        targets = targets[valid_mask]
        predictions = predictions[valid_mask]

        indices = targets * self.num_classes + predictions
        counts = torch.bincount(indices, minlength=self.num_classes ** 2)
        confusion = counts.reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += confusion.to(self.device)

    def evaluate(self) -> Dict[str, float]:
        """
        计算所有指标

        Returns:
            metrics: 包含各指标的字典
        """
        metrics = {}

        cm = self.confusion_matrix.float()
        intersection = torch.diag(cm)
        union = cm.sum(1) + cm.sum(0) - intersection

        class_iou = intersection / (union + 1e-10)
        valid_classes = union > 0
        class_iou = class_iou.masked_fill(~valid_classes, 0.0)

        if 'IoU' in self.metric_list or 'mIoU' in self.metric_list:
            metrics['class_IoU'] = {}
            for i, name in enumerate(self.class_names):
                if i < len(class_iou):
                    metrics['class_IoU'][name] = class_iou[i].item()

        if 'mIoU' in self.metric_list:
            metrics['mIoU'] = class_iou.mean().item()

        if 'Accuracy' in self.metric_list:
            metrics['Accuracy'] = intersection.sum().item() / (cm.sum() + 1e-10)

        if 'Precision' in self.metric_list:
            precision = intersection / (cm.sum(0) + 1e-10)
            precision = precision.masked_fill(~valid_classes, 0.0)
            metrics['Precision'] = precision.mean().item()

        if 'Recall' in self.metric_list:
            recall = intersection / (cm.sum(1) + 1e-10)
            recall = recall.masked_fill(~valid_classes, 0.0)
            metrics['Recall'] = recall.mean().item()

        if 'F1' in self.metric_list:
            precision = intersection / (cm.sum(0) + 1e-10)
            recall = intersection / (cm.sum(1) + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            f1 = f1.masked_fill(~valid_classes, 0.0)
            metrics['F1'] = f1.mean().item()

        metrics['total_samples'] = self.total_samples
        metrics['evaluation_time'] = self.total_time

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """
        获取混淆矩阵

        Returns:
            numpy 格式的混淆矩阵
        """
        return self.confusion_matrix.cpu().numpy()

    def plot_confusion_matrix(
        self,
        save_path: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        绘制混淆矩阵

        Args:
            save_path: 保存路径
            normalize: 是否归一化
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            cm = self.confusion_matrix.cpu().numpy().astype(float)
            if normalize:
                cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-10)

            plt.figure(figsize=(12, 10))
            sns.heatmap(
                cm,
                annot=True,
                fmt='.2f' if normalize else 'd',
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
            )
            plt.xlabel('Predicted')
            plt.ylabel('Ground Truth')
            plt.title('Confusion Matrix')

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Confusion matrix saved to {save_path}")

            plt.close()

        except ImportError:
            print("Warning: matplotlib/seaborn not installed, skipping plot")

    def get_classification_report(self) -> Dict[str, Dict[str, float]]:
        """
        获取分类报告

        Returns:
            包含每个类别的 precision, recall, f1 的字典
        """
        cm = self.confusion_matrix.float()
        intersection = torch.diag(cm)
        union = cm.sum(1) + cm.sum(0) - intersection

        precision = intersection / (cm.sum(0) + 1e-10)
        recall = intersection / (cm.sum(1) + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        valid = union > 0
        precision = precision.masked_fill(~valid, 0.0)
        recall = recall.masked_fill(~valid, 0.0)
        f1 = f1.masked_fill(~valid, 0.0)

        report = {}
        for i, name in enumerate(self.class_names):
            if i < len(precision):
                report[name] = {
                    'precision': precision[i].item(),
                    'recall': recall[i].item(),
                    'f1': f1[i].item(),
                    'support': int(cm[i].sum().item()),
                }

        return report

    def __repr__(self) -> str:
        return (
            f"BEVSegmentationEvaluator("
            f"classes={self.num_classes}, "
            f"metrics={self.metric_list}, "
            f"samples={self.total_samples}"
            f")"
        )


def compute_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 IoU

    Args:
        predictions: 预测结果 [B, H, W]
        targets: 真实标签 [B, H, W]
        num_classes: 类别数量
        ignore_index: 忽略的类别索引

    Returns:
        iou: 各类别 IoU [num_classes]
        valid: 有效类别掩码
    """
    predictions = predictions.long()
    targets = targets.long()

    mask = (targets != ignore_index)
    predictions = predictions[mask]
    targets = targets[mask]

    cm = torch.zeros(num_classes, num_classes, device=predictions.device)
    indices = targets * num_classes + predictions
    counts = torch.bincount(indices, minlength=num_classes ** 2)
    cm = counts.reshape(num_classes, num_classes)

    intersection = torch.diag(cm)
    union = cm.sum(1) + cm.sum(0) - intersection
    iou = intersection / (union + 1e-10)
    valid = union > 0

    return iou, valid


def compute_miou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    """
    计算平均 IoU

    Args:
        predictions: 预测结果 [B, H, W]
        targets: 真实标签 [B, H, W]
        num_classes: 类别数量
        ignore_index: 忽略的类别索引

    Returns:
        miou: 平均 IoU 值
    """
    iou, valid = compute_iou(predictions, targets, num_classes, ignore_index)
    iou = iou.masked_fill(~valid, 0.0)
    return iou.mean().item()


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 255,
) -> float:
    """
    计算像素准确率

    Args:
        predictions: 预测结果 [B, H, W]
        targets: 真实标签 [B, H, W]
        ignore_index: 忽略的类别索引

    Returns:
        accuracy: 像素准确率
    """
    predictions = predictions.long()
    targets = targets.long()

    mask = (targets != ignore_index)
    correct = ((predictions == targets) & mask).sum()
    total = mask.sum()

    return (correct / (total + 1e-10)).item()
