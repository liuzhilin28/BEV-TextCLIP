#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEV-TextCLIP 可视化模块

提供分割结果可视化功能：
1. BEV空间可视化
2. 文本-注意力可视化
3. 分割掩码可视化
4. 多视角对比可视化

日期: 2026-01-26（更新）
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import os
from datetime import datetime


class SegmentationVisualizer:
    """分割结果可视化工具类"""

    def __init__(self, class_names: List[str], colormap: Optional[np.ndarray] = None, output_dir: str = "visualization_results"):
        """
        初始化可视化工具

        Args:
            class_names: 类别名称列表
            colormap: 自定义颜色映射表，shape为[num_classes, 3]，值范围[0, 1]
            output_dir: 输出目录
        """
        self.class_names = class_names
        self.num_classes = len(class_names)

        if colormap is None:
            self.colormap = self._generate_colormap()
        else:
            self.colormap = colormap

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _generate_colormap(self) -> np.ndarray:
        """生成类别颜色映射表"""
        colors = plt.cm.get_cmap('tab20', self.num_classes)
        colormap = np.zeros((self.num_classes, 3))
        for i in range(self.num_classes):
            colormap[i] = colors(i)[:3]
        if self.num_classes > 0:
            colormap[self.num_classes - 1] = [0.7, 0.7, 0.7]
        return colormap

    def visualize_segmentation(
        self,
        predictions: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "Segmentation Result",
        figsize: Tuple[int, int] = (12, 10),
        show_legend: bool = True,
        dpi: int = 150
    ) -> plt.Figure:
        """
        可视化分割结果

        Args:
            predictions: 预测掩码，shape为[H, W]，值为类别索引
            save_path: 保存路径，None表示不保存
            title: 图像标题
            figsize: 图像大小
            show_legend: 是否显示图例
            dpi: 图像分辨率

        Returns:
            matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        im = ax.imshow(predictions, cmap=ListedColormap(self.colormap), vmin=0, vmax=self.num_classes - 1)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')

        if show_legend:
            legend_patches = [mpatches.Patch(color=self.colormap[i], label=self.class_names[i])
                              for i in range(self.num_classes)]
            ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.35, 1), fontsize=8)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"图像已保存至: {save_path}")

        return fig

    def visualize_probability_map(
        self,
        probabilities: np.ndarray,
        class_idx: int,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 150
    ) -> plt.Figure:
        """
        可视化特定类别的概率图

        Args:
            probabilities: 概率图，shape为[num_classes, H, W]
            class_idx: 要可视化的类别索引
            save_path: 保存路径
            title: 图像标题
            figsize: 图像大小
            dpi: 图像分辨率

        Returns:
            matplotlib.figure.Figure
        """
        if title is None:
            title = f"Probability Map: {self.class_names[class_idx]}"

        prob_map = probabilities[class_idx]

        fig, axes = plt.subplots(1, 2, figsize=(figsize[0], figsize[1] / 2))

        im1 = axes[0].imshow(prob_map, cmap='viridis', vmin=0, vmax=1)
        axes[0].set_title(title, fontsize=12, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        axes[1].hist(prob_map.flatten(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"Class {class_idx}"
        axes[1].set_title(f"Probability Distribution: {class_name}", fontsize=12)
        axes[1].set_xlabel("Probability")
        axes[1].set_ylabel("Frequency")
        axes[1].axvline(x=prob_map.mean(), color='red', linestyle='--', label=f"Mean: {prob_map.mean():.3f}")
        axes[1].legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"概率图已保存至: {save_path}")

        return fig

    def visualize_all_probability_maps(
        self,
        probabilities: np.ndarray,
        save_dir: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 150
    ) -> Dict[str, str]:
        """
        可视化所有类别的概率图

        Args:
            probabilities: 概率图，shape为[num_classes, H, W]
            save_dir: 保存目录
            figsize: 图像大小
            dpi: 图像分辨率

        Returns:
            包含保存路径的字典
        """
        if save_dir is None:
            save_dir = self.output_dir

        saved_files = {}
        for i in range(min(self.num_classes, probabilities.shape[0])):
            class_name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            save_path = os.path.join(save_dir, f"probability_map_{class_name}.png")
            self.visualize_probability_map(probabilities, i, save_path, figsize=figsize, dpi=dpi)
            saved_files[class_name] = save_path

        return saved_files

    def visualize_bev_features(
        self,
        bev_features: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "BEV Features",
        figsize: Tuple[int, int] = (16, 4),
        dpi: int = 150
    ) -> plt.Figure:
        """
        可视化BEV特征图（选择指定通道）

        Args:
            bev_features: BEV特征，shape为[C, H, W]
            save_path: 保存路径
            title: 图像标题
            figsize: 图像大小
            dpi: 图像分辨率

        Returns:
            matplotlib.figure.Figure
        """
        num_channels = bev_features.shape[0]
        if num_channels >= 4:
            selected_channels = [0, num_channels // 4, num_channels // 2, 3 * num_channels // 4]
        else:
            selected_channels = list(range(min(num_channels, 4)))

        fig, axes = plt.subplots(1, len(selected_channels), figsize=figsize)

        if len(selected_channels) == 1:
            axes = [axes]

        for idx, channel in enumerate(selected_channels):
            im = axes[idx].imshow(bev_features[channel], cmap='viridis')
            axes[idx].set_title(f"Channel {channel}", fontsize=10)
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"BEV特征图已保存至: {save_path}")

        return fig

    def visualize_attention_weights(
        self,
        attention_weights: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "Attention Weights",
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 150
    ) -> plt.Figure:
        """
        可视化注意力权重

        Args:
            attention_weights: 注意力权重，shape为[num_classes, H, W]
            save_path: 保存路径
            title: 图像标题
            figsize: 图像大小
            dpi: 图像分辨率

        Returns:
            matplotlib.figure.Figure
        """
        num_classes = attention_weights.shape[0]
        n_cols = min(4, num_classes)
        n_rows = (num_classes + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for i in range(num_classes):
            im = axes[i].imshow(attention_weights[i], cmap='hot', vmin=0, vmax=attention_weights[i].max())
            class_name = self.class_names[i] if i < len(self.class_names) else f"Class {i}"
            axes[i].set_title(class_name, fontsize=8)
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        for j in range(num_classes, len(axes)):
            axes[j].axis('off')

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"注意力权重图已保存至: {save_path}")

        return fig

    def visualize_text_embeddings(
        self,
        text_embeddings: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "Text Embeddings",
        figsize: Tuple[int, int] = (12, 6),
        dpi: int = 150
    ) -> plt.Figure:
        """
        可视化文本嵌入向量

        Args:
            text_embeddings: 文本嵌入，shape为[num_classes, embedding_dim]
            save_path: 保存路径
            title: 图像标题
            figsize: 图像大小
            dpi: 图像分辨率

        Returns:
            matplotlib.figure.Figure
        """
        try:
            from scipy.cluster.hierarchy import dendrogram, linkage
            from scipy.spatial.distance import pdist
            has_scipy = True
        except ImportError:
            has_scipy = False

        fig, axes = plt.subplots(1, 2 if has_scipy else 1, figsize=figsize)

        im1 = axes[0].imshow(text_embeddings, cmap='RdYlBu', aspect='auto')
        axes[0].set_title("Text Embedding Matrix", fontsize=12)
        axes[0].set_yticks(range(self.num_classes))
        axes[0].set_yticklabels(self.class_names)
        axes[0].set_xlabel("Embedding Dimension")
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        if has_scipy:
            linkage_matrix = linkage(text_embeddings, method='ward')
            dendrogram(linkage_matrix, labels=self.class_names, ax=axes[1], leaf_rotation=45)
            axes[1].set_title("Semantic Similarity (Hierarchical Clustering)", fontsize=12)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"文本嵌入可视化已保存至: {save_path}")

        return fig

    def visualize_comparison(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "Prediction vs Ground Truth",
        figsize: Tuple[int, int] = (18, 6),
        dpi: int = 150
    ) -> plt.Figure:
        """
        可视化预测结果与真值的对比

        Args:
            predictions: 预测掩码，shape为[H, W]
            ground_truth: 真值掩码，shape为[H, W]
            save_path: 保存路径
            title: 图像标题
            figsize: 图像大小
            dpi: 图像分辨率

        Returns:
            matplotlib.figure.Figure
        """
        iou = self._calculate_iou(predictions, ground_truth)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        im1 = axes[0].imshow(predictions, cmap=ListedColormap(self.colormap), vmin=0, vmax=self.num_classes - 1)
        axes[0].set_title("Prediction", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        im2 = axes[1].imshow(ground_truth, cmap=ListedColormap(self.colormap), vmin=0, vmax=self.num_classes - 1)
        axes[1].set_title("Ground Truth", fontsize=12, fontweight='bold')
        axes[1].axis('off')

        error_map = np.zeros_like(predictions, dtype=float)
        error_map[predictions != ground_truth] = 1.0
        im3 = axes[2].imshow(error_map, cmap='Reds', vmin=0, vmax=1)
        axes[2].set_title(f"Error Map (IoU: {iou:.3f})", fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        if self.num_classes > 0:
            legend_patches = [mpatches.Patch(color=self.colormap[i], label=self.class_names[i])
                              for i in range(self.num_classes)]
            fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=8, fontsize=8)

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"对比图已保存至: {save_path}")

        return fig

    def visualize_multi_sample(
        self,
        predictions_list: List[np.ndarray],
        sample_names: List[str],
        save_path: Optional[str] = None,
        title: str = "Multi-Sample Results",
        figsize: Optional[Tuple[int, int]] = None,
        dpi: int = 150
    ) -> plt.Figure:
        """
        可视化多个样本的分割结果

        Args:
            predictions_list: 预测掩码列表，每个shape为[H, W]
            sample_names: 样本名称列表
            save_path: 保存路径
            title: 图像标题
            figsize: 图像大小
            dpi: 图像分辨率

        Returns:
            matplotlib.figure.Figure
        """
        n_samples = len(predictions_list)
        if figsize is None:
            figsize = (5 * n_samples, 5)

        fig, axes = plt.subplots(1, n_samples, figsize=figsize)

        if n_samples == 1:
            axes = [axes]

        for idx, (pred, name) in enumerate(zip(predictions_list, sample_names)):
            im = axes[idx].imshow(pred, cmap=ListedColormap(self.colormap), vmin=0, vmax=self.num_classes - 1)
            axes[idx].set_title(name, fontsize=12, fontweight='bold')
            axes[idx].axis('off')

        if self.num_classes > 0:
            legend_patches = [mpatches.Patch(color=self.colormap[i], label=self.class_names[i])
                              for i in range(self.num_classes)]
            fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=8, fontsize=8)

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"多样本对比图已保存至: {save_path}")

        return fig

    def create_visualization_report(
        self,
        predictions: np.ndarray,
        ground_truth: Optional[np.ndarray],
        probabilities: np.ndarray,
        bev_features: Optional[np.ndarray],
        text_embeddings: np.ndarray,
        attention_weights: Optional[np.ndarray],
        save_prefix: str = "visualization",
        dpi: int = 150
    ) -> Dict[str, str]:
        """
        生成完整的可视化报告

        Args:
            predictions: 预测掩码
            ground_truth: 真值掩码
            probabilities: 概率图
            bev_features: BEV特征
            text_embeddings: 文本嵌入
            attention_weights: 注意力权重
            save_prefix: 文件名前缀
            dpi: 图像分辨率

        Returns:
            包含所有保存路径的字典
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.output_dir, f"{save_prefix}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        saved_files = {}

        save_path = os.path.join(save_dir, "segmentation_result.png")
        self.visualize_segmentation(predictions, save_path, "Segmentation Result", dpi=dpi)
        saved_files["segmentation"] = save_path

        for i in range(min(4, min(self.num_classes, probabilities.shape[0]))):
            class_name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            save_path = os.path.join(save_dir, f"probability_map_{class_name}.png")
            self.visualize_probability_map(probabilities, i, save_path, dpi=dpi)
            saved_files[f"probability_{class_name}"] = save_path

        if bev_features is not None:
            save_path = os.path.join(save_dir, "bev_features.png")
            self.visualize_bev_features(bev_features, save_path, dpi=dpi)
            saved_files["bev_features"] = save_path

        if attention_weights is not None:
            save_path = os.path.join(save_dir, "attention_weights.png")
            self.visualize_attention_weights(attention_weights, save_path, dpi=dpi)
            saved_files["attention_weights"] = save_path

        save_path = os.path.join(save_dir, "text_embeddings.png")
        self.visualize_text_embeddings(text_embeddings, save_path, dpi=dpi)
        saved_files["text_embeddings"] = save_path

        if ground_truth is not None:
            save_path = os.path.join(save_dir, "comparison.png")
            self.visualize_comparison(predictions, ground_truth, save_path, dpi=dpi)
            saved_files["comparison"] = save_path

        return saved_files

    def _calculate_iou(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """计算整体IoU"""
        valid_mask = ground_truth < self.num_classes
        if not np.any(valid_mask):
            return 0.0

        intersection = np.logical_and(predictions == ground_truth, valid_mask)
        union = np.logical_or(predictions == ground_truth, valid_mask)
        iou = intersection.sum() / (union.sum() + 1e-10)
        return iou

    def save_colormap_legend(
        self,
        save_path: str,
        figsize: Tuple[int, int] = (12, 3),
        dpi: int = 150
    ) -> plt.Figure:
        """
        保存颜色映射图例

        Args:
            save_path: 保存路径
            figsize: 图像大小
            dpi: 图像分辨率

        Returns:
            matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        legend_patches = [mpatches.Patch(color=self.colormap[i], label=self.class_names[i])
                          for i in range(self.num_classes)]
        ax.legend(handles=legend_patches, loc='center', ncol=4, fontsize=10)
        ax.axis('off')

        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"颜色图例已保存至: {save_path}")

        return fig
