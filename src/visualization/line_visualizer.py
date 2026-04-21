#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEV-TextCLIP 线条风格可视化模块

实现学术Paper风格的简洁线条可视化：
1. 从分割掩码提取轮廓线条
2. 不同类别使用不同颜色线条
3. 平滑处理去除噪声
4. 类似高精地图的俯视图效果

日期: 2026-03-16
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from typing import List, Dict, Optional, Tuple
from scipy import ndimage
from skimage import measure
import cv2
import os


class LineStyleVisualizer:
    """线条风格BEV可视化工具类"""
    
    # nuScenes标准颜色映射（线条颜色）
    NUSCENES_LINE_COLORS = {
        'vehicle': '#FF0000',           # 红色 - 车辆
        'pedestrian': '#FF00FF',        # 品红 - 行人
        'motorcycle': '#FF4500',        # 橙红 - 摩托车
        'bicycle': '#FFA500',           # 橙色 - 自行车
        'traffic cone': '#FFD700',      # 金色 - 交通锥
        'barrier': '#808080',           # 灰色 - 障碍物
        'driveable surface': '#0000FF', # 蓝色 - 可行驶区域
        'sidewalk': '#00FF00',          # 绿色 - 人行道
        'other flat': '#00CED1',        # 深青 - 其他平面
        'vegetation': '#228B22',        # 森林绿 - 植被
        'manmade': '#800080',           # 紫色 - 人造物
    }
    
    def __init__(self, class_names: List[str], output_dir: str = "visualization_results"):
        """
        初始化线条风格可视化工具
        
        Args:
            class_names: 类别名称列表
            output_dir: 输出目录
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 为每个类别分配颜色
        self.line_colors = self._assign_colors()
        
    def _assign_colors(self) -> Dict[int, str]:
        """为每个类别分配线条颜色"""
        colors = {}
        for i, name in enumerate(self.class_names):
            if name.lower() in self.NUSCENES_LINE_COLORS:
                colors[i] = self.NUSCENES_LINE_COLORS[name.lower()]
            else:
                # 默认颜色
                default_colors = ['#FF0000', '#0000FF', '#00FF00', '#FFD700', '#FF00FF', 
                                  '#00FFFF', '#FFA500', '#800080', '#808080', '#FF4500']
                colors[i] = default_colors[i % len(default_colors)]
        return colors
    
    def extract_contours(self, segmentation_mask: np.ndarray, 
                        class_idx: int, 
                        min_area: float = 10.0,
                        epsilon_factor: float = 0.01) -> List[np.ndarray]:
        """
        从分割掩码中提取指定类别的轮廓
        
        Args:
            segmentation_mask: 分割掩码 [H, W]
            class_idx: 类别索引
            min_area: 最小轮廓面积
            epsilon_factor: 轮廓简化因子
            
        Returns:
            轮廓列表，每个轮廓是[N, 2]的点数组
        """
        # 创建二值掩码
        binary_mask = (segmentation_mask == class_idx).astype(np.uint8)
        
        # 形态学操作去除噪声
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # 提取轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤小面积轮廓并简化
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                # 轮廓简化
                epsilon = epsilon_factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) >= 3:  # 至少3个点
                    filtered_contours.append(approx.reshape(-1, 2))
        
        return filtered_contours
    
    def smooth_contours(self, contours: List[np.ndarray], 
                       smoothing_factor: float = 0.1) -> List[np.ndarray]:
        """
        平滑轮廓线条
        
        Args:
            contours: 轮廓列表
            smoothing_factor: 平滑因子
            
        Returns:
            平滑后的轮廓列表
        """
        smoothed = []
        for contour in contours:
            if len(contour) < 4:
                smoothed.append(contour)
                continue
            
            # 使用B样条曲线平滑
            from scipy.interpolate import splprep, splev
            
            # 闭合轮廓
            if not np.array_equal(contour[0], contour[-1]):
                contour = np.vstack([contour, contour[0]])
            
            try:
                # 参数化样条
                tck, u = splprep([contour[:, 0], contour[:, 1]], s=smoothing_factor * len(contour), per=1)
                
                # 生成平滑点
                u_new = np.linspace(0, 1, len(contour) * 2)
                x_new, y_new = splev(u_new, tck)
                
                smoothed_contour = np.column_stack([x_new, y_new])
                smoothed.append(smoothed_contour)
            except:
                # 如果样条拟合失败，使用原始轮廓
                smoothed.append(contour)
        
        return smoothed
    
    def visualize_line_style(self,
                            segmentation_mask: np.ndarray,
                            save_path: Optional[str] = None,
                            title: str = "BEV Line Visualization",
                            figsize: Tuple[int, int] = (10, 10),
                            dpi: int = 300,
                            show_grid: bool = False,
                            line_width: float = 1.5,
                            show_legend: bool = True,
                            smoothing: bool = True) -> plt.Figure:
        """
        创建线条风格的BEV可视化
        
        Args:
            segmentation_mask: 分割掩码 [H, W]
            save_path: 保存路径
            title: 图像标题
            figsize: 图像大小
            dpi: 图像分辨率
            show_grid: 是否显示网格
            line_width: 线条宽度
            show_legend: 是否显示图例
            smoothing: 是否平滑轮廓
            
        Returns:
            matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # 设置白色背景
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # 遍历每个类别提取并绘制轮廓
        legend_handles = []
        
        for class_idx in range(self.num_classes):
            # 提取轮廓
            contours = self.extract_contours(segmentation_mask, class_idx)
            
            if len(contours) == 0:
                continue
            
            # 平滑轮廓
            if smoothing:
                contours = self.smooth_contours(contours)
            
            # 绘制轮廓
            color = self.line_colors.get(class_idx, '#000000')
            for contour in contours:
                if len(contour) >= 2:
                    ax.plot(contour[:, 0], contour[:, 1], 
                           color=color, 
                           linewidth=line_width,
                           alpha=0.9)
            
            # 添加到图例
            if show_legend and len(contours) > 0:
                legend_handles.append(
                    plt.Line2D([0], [0], color=color, linewidth=line_width, 
                              label=self.class_names[class_idx])
                )
        
        # 设置坐标轴
        ax.set_xlim(0, segmentation_mask.shape[1])
        ax.set_ylim(segmentation_mask.shape[0], 0)  # Y轴反转，符合图像坐标
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 添加网格（可选）
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # 添加标题
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        
        # 添加图例
        if show_legend and legend_handles:
            ax.legend(handles=legend_handles, 
                     loc='upper right', 
                     bbox_to_anchor=(1.15, 1),
                     fontsize=9,
                     frameon=True,
                     fancybox=True,
                     shadow=False)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"线条风格可视化已保存至: {save_path}")
        
        return fig
    
    def visualize_comparison_line_style(self,
                                      predictions: np.ndarray,
                                      ground_truth: np.ndarray,
                                      save_path: Optional[str] = None,
                                      title: str = "Line Style Comparison",
                                      figsize: Tuple[int, int] = (20, 10),
                                      dpi: int = 300) -> plt.Figure:
        """
        对比预测与真值的线条风格可视化
        
        Args:
            predictions: 预测掩码 [H, W]
            ground_truth: 真值掩码 [H, W]
            save_path: 保存路径
            title: 图像标题
            figsize: 图像大小
            dpi: 图像分辨率
            
        Returns:
            matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        for ax in axes:
            ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # 绘制预测结果
        legend_handles = []
        for class_idx in range(self.num_classes):
            contours = self.extract_contours(predictions, class_idx)
            if len(contours) == 0:
                continue
            
            color = self.line_colors.get(class_idx, '#000000')
            for contour in contours:
                if len(contour) >= 2:
                    axes[0].plot(contour[:, 0], contour[:, 1], 
                               color=color, linewidth=1.5, alpha=0.9)
            
            if len(contours) > 0:
                legend_handles.append(
                    plt.Line2D([0], [0], color=color, linewidth=1.5, 
                              label=self.class_names[class_idx])
                )
        
        axes[0].set_xlim(0, predictions.shape[1])
        axes[0].set_ylim(predictions.shape[0], 0)
        axes[0].set_aspect('equal')
        axes[0].axis('off')
        axes[0].set_title("Prediction", fontsize=14, fontweight='bold')
        
        # 绘制真值
        for class_idx in range(self.num_classes):
            contours = self.extract_contours(ground_truth, class_idx)
            if len(contours) == 0:
                continue
            
            color = self.line_colors.get(class_idx, '#000000')
            for contour in contours:
                if len(contour) >= 2:
                    axes[1].plot(contour[:, 0], contour[:, 1], 
                               color=color, linewidth=1.5, alpha=0.9)
        
        axes[1].set_xlim(0, ground_truth.shape[1])
        axes[1].set_ylim(ground_truth.shape[0], 0)
        axes[1].set_aspect('equal')
        axes[1].axis('off')
        axes[1].set_title("Ground Truth", fontsize=14, fontweight='bold')
        
        # 添加共享图例
        if legend_handles:
            fig.legend(handles=legend_handles, 
                      loc='upper center', 
                      bbox_to_anchor=(0.5, 0.02),
                      ncol=min(6, len(legend_handles)),
                      fontsize=9,
                      frameon=True)
        
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"对比可视化已保存至: {save_path}")
        
        return fig
    
    def visualize_multi_sample_lines(self,
                                    predictions_list: List[np.ndarray],
                                    sample_names: List[str],
                                    save_path: Optional[str] = None,
                                    title: str = "Multi-Sample Line Visualization",
                                    figsize: Optional[Tuple[int, int]] = None,
                                    dpi: int = 300,
                                    ncols: int = 4) -> plt.Figure:
        """
        可视化多个样本的线条风格结果（用于Paper展示）
        
        Args:
            predictions_list: 预测掩码列表
            sample_names: 样本名称列表
            save_path: 保存路径
            title: 图像标题
            figsize: 图像大小
            dpi: 图像分辨率
            ncols: 每行显示的样本数
            
        Returns:
            matplotlib.figure.Figure
        """
        n_samples = len(predictions_list)
        nrows = (n_samples + ncols - 1) // ncols
        
        if figsize is None:
            figsize = (4 * ncols, 4 * nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        if nrows == 1:
            axes = axes.reshape(1, -1)
        
        for ax in axes.flat:
            ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        legend_handles = []
        
        for idx, (pred, name) in enumerate(zip(predictions_list, sample_names)):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col]
            
            # 绘制该样本的所有类别轮廓
            for class_idx in range(self.num_classes):
                contours = self.extract_contours(pred, class_idx, min_area=5.0)
                if len(contours) == 0:
                    continue
                
                color = self.line_colors.get(class_idx, '#000000')
                for contour in contours:
                    if len(contour) >= 2:
                        ax.plot(contour[:, 0], contour[:, 1], 
                               color=color, linewidth=1.2, alpha=0.85)
                
                # 只在第一个样本添加图例
                if idx == 0 and len(contours) > 0:
                    legend_handles.append(
                        plt.Line2D([0], [0], color=color, linewidth=1.5, 
                                  label=self.class_names[class_idx])
                    )
            
            ax.set_xlim(0, pred.shape[1])
            ax.set_ylim(pred.shape[0], 0)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(name, fontsize=11, fontweight='bold')
        
        # 隐藏多余的子图
        for idx in range(n_samples, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].axis('off')
        
        # 添加图例
        if legend_handles:
            fig.legend(handles=legend_handles, 
                      loc='upper center', 
                      bbox_to_anchor=(0.5, 0.02),
                      ncol=min(6, len(legend_handles)),
                      fontsize=8,
                      frameon=True)
        
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"多样本线条可视化已保存至: {save_path}")
        
        return fig


# 便捷函数
def create_line_visualization(segmentation_mask: np.ndarray,
                              class_names: List[str],
                              save_path: Optional[str] = None,
                              **kwargs) -> plt.Figure:
    """
    便捷函数：创建线条风格可视化
    
    Args:
        segmentation_mask: 分割掩码 [H, W]
        class_names: 类别名称列表
        save_path: 保存路径
        **kwargs: 其他可视化参数
        
    Returns:
        matplotlib.figure.Figure
    """
    visualizer = LineStyleVisualizer(class_names)
    return visualizer.visualize_line_style(segmentation_mask, save_path, **kwargs)


if __name__ == "__main__":
    # 示例用法
    print("线条风格可视化模块")
    print("使用方法:")
    print("from src.visualization.line_visualizer import LineStyleVisualizer")
    print("")
    print("visualizer = LineStyleVisualizer(class_names)")
    print("fig = visualizer.visualize_line_style(segmentation_mask, save_path='output.png')")
