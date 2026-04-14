#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEV-TextCLIP HDMap风格可视化模块 (简化版)

实现高精地图风格的简洁线条可视化：
- 道路边界用蓝色粗线条
- 车辆用红色矩形框
- 人行道用绿色线条
- 简洁专业，适合Paper展示

日期: 2026-03-16
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Optional, Tuple
import cv2
import os


class HDMapVisualizer:
    """高精地图风格BEV可视化工具"""
    
    # HDMap风格颜色
    COLORS = {
        'driveable': '#0066CC',      # 深蓝色 - 可行驶区域边界
        'vehicle': '#FF3333',        # 红色 - 车辆
        'pedestrian': '#FF00FF',     # 品红 - 行人
        'sidewalk': '#00CC00',       # 绿色 - 人行道
        'barrier': '#666666',        # 灰色 - 障碍物
    }
    
    def __init__(self, class_names: List[str], output_dir: str = "visualization_results"):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_boundaries(self, mask: np.ndarray, class_idx: int) -> List[np.ndarray]:
        """提取区域边界轮廓"""
        binary = (mask == class_idx).astype(np.uint8) * 255
        
        # 提取外部轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boundaries = []
        for cnt in contours:
            if len(cnt) >= 3:
                # 简化轮廓
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) >= 3:
                    boundaries.append(approx.reshape(-1, 2))
        
        return boundaries
    
    def extract_boxes(self, mask: np.ndarray, class_idx: int) -> List[Tuple[int, int, int, int]]:
        """提取边界框"""
        binary = (mask == class_idx).astype(np.uint8)
        
        # 查找连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        boxes = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area >= 10:
                boxes.append((x, y, w, h))
        
        return boxes
    
    def visualize_hdmap(self,
                       segmentation_mask: np.ndarray,
                       save_path: Optional[str] = None,
                       title: str = "HDMap Style Visualization",
                       figsize: Tuple[int, int] = (10, 10),
                       dpi: int = 300) -> plt.Figure:
        """
        创建HDMap风格的BEV可视化
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        H, W = segmentation_mask.shape
        legend_handles = []
        
        # 绘制每个类别
        for class_idx in range(min(self.num_classes, len(self.class_names))):
            class_name = self.class_names[class_idx].lower()
            
            if class_name in ['background', 'ignore']:
                continue
            
            # 绘制可行驶区域
            if 'driveable' in class_name:
                boundaries = self.extract_boundaries(segmentation_mask, class_idx)
                color = self.COLORS['driveable']
                
                for boundary in boundaries:
                    if len(boundary) >= 3:
                        boundary_closed = np.vstack([boundary, boundary[0]])
                        ax.plot(boundary_closed[:, 0], boundary_closed[:, 1], 
                               color=color, linewidth=2.5, alpha=0.9)
                
                if boundaries:
                    legend_handles.append(
                        plt.Line2D([0], [0], color=color, linewidth=2.5, label='Road')
                    )
            
            # 绘制车辆
            elif class_name in ['vehicle', 'car']:
                boxes = self.extract_boxes(segmentation_mask, class_idx)
                color = self.COLORS['vehicle']
                
                for x, y, w, h in boxes:
                    rect = mpatches.Rectangle((x, y), w, h, 
                                            linewidth=2.5, 
                                            edgecolor=color, 
                                            facecolor='none',
                                            alpha=0.9)
                    ax.add_patch(rect)
                
                if boxes:
                    legend_handles.append(
                        mpatches.Rectangle((0, 0), 1, 1, linewidth=2.5, 
                                       edgecolor=color, facecolor='none', label='Vehicle')
                    )
            
            # 绘制人行道
            elif class_name in ['sidewalk']:
                boundaries = self.extract_boundaries(segmentation_mask, class_idx)
                color = self.COLORS['sidewalk']
                
                for boundary in boundaries:
                    if len(boundary) >= 3:
                        boundary_closed = np.vstack([boundary, boundary[0]])
                        ax.plot(boundary_closed[:, 0], boundary_closed[:, 1], 
                               color=color, linewidth=2.0, alpha=0.9)
                
                if boundaries:
                    legend_handles.append(
                        plt.Line2D([0], [0], color=color, linewidth=2.0, label='Sidewalk')
                    )
            
            # 绘制其他类别
            else:
                boundaries = self.extract_boundaries(segmentation_mask, class_idx)
                color = '#{:06x}'.format(0xFF0000 + (class_idx * 0x111111) % 0xFFFFFF)
                for boundary in boundaries:
                    if len(boundary) >= 3:
                        boundary_closed = np.vstack([boundary, boundary[0]])
                        ax.plot(boundary_closed[:, 0], boundary_closed[:, 1], 
                               color=color, linewidth=1.5, alpha=0.8)
        
        # 设置坐标轴
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect('equal')
        ax.axis('off')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        
        # 添加图例
        if legend_handles:
            ax.legend(handles=legend_handles, 
                     loc='upper right',
                     bbox_to_anchor=(1.2, 1.0),
                     fontsize=9,
                     frameon=True)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"保存至: {save_path}")
        
        return fig
