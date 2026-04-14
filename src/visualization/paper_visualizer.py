#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEV-TextCLIP Paper风格完整对比可视化

生成类似您图片中的完整对比图：
左边: 多视角相机图像 (6 views)
右边: 线条风格分割结果 (GT/M2/M3/M6)

日期: 2026-03-16
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple, Optional
import cv2
import os


class PaperVisualizer:
    """Paper风格完整可视化工具"""
    
    # 颜色配置
    COLORS = {
        'driveable': '#0000FF',   # 蓝色 - 道路
        'vehicle': '#FF0000',      # 红色 - 车辆
        'sidewalk': '#00FF00',     # 绿色 - 人行道
        'lane': '#0000FF',         # 蓝色 - 车道线
    }
    
    def __init__(self, output_dir: str = "visualization_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_paper_figure(self,
                          camera_images: List[np.ndarray],
                          segmentation_masks: List[np.ndarray],
                          mask_labels: List[str],
                          save_path: str = "paper_visualization.png",
                          figsize: Tuple[int, int] = (16, 8),
                          dpi: int = 300) -> plt.Figure:
        """
        创建完整的Paper对比图
        
        布局: [Images (2x3)] [GT] [M2] [M3] [M6]
        
        Args:
            camera_images: 6个相机图像列表 [6, H, W, 3]
            segmentation_masks: 4个分割掩码列表 [4, H, W]
            mask_labels: 掩码标签 ['GT', 'M2', 'M3', 'M6']
            save_path: 保存路径
            figsize: 图像大小
            dpi: 分辨率
        """
        n_cameras = len(camera_images)
        n_masks = len(segmentation_masks)
        n_cols = 1 + n_masks  # 1列Images + n_masks列分割结果
        
        # 创建大图
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor('white')
        
        # 创建网格
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, n_cols, figure=fig, width_ratios=[3] + [1]*n_masks, 
                     wspace=0.1, hspace=0.1)
        
        # ========== 左边: 相机图像 (2行3列) ==========
        ax_images = []
        for i in range(n_cameras):
            row = i // 3
            col = i % 3
            ax = fig.add_subplot(gs[row, 0])
            ax_images.append(ax)
            
            # 显示相机图像
            img = camera_images[i]
            ax.imshow(img)
            ax.axis('off')
            
            # 添加视角标签
            if i < 3:
                ax.set_title(f'Cam {i+1}', fontsize=10, pad=5)
        
        # 添加 Images 标签
        fig.text(0.15, 0.02, 'Images', ha='center', fontsize=14, fontweight='bold')
        
        # ========== 右边: 分割结果 ==========
        for mask_idx, (mask, label) in enumerate(zip(segmentation_masks, mask_labels)):
            col = mask_idx + 1  # 从第2列开始
            
            # 创建子图跨2行
            ax = fig.add_subplot(gs[:, col])
            ax.set_facecolor('white')
            
            # 绘制线条风格分割
            self._draw_hdmap_lines(ax, mask)
            
            # 设置标题
            ax.set_title(label, fontsize=14, fontweight='bold', pad=10)
            ax.set_xlim(0, mask.shape[1])
            ax.set_ylim(mask.shape[0], 0)
            ax.set_aspect('equal')
            ax.axis('off')
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], color=self.COLORS['driveable'], linewidth=2.5, label='Road'),
            mpatches.Rectangle((0, 0), 1, 1, linewidth=2.5, edgecolor=self.COLORS['vehicle'], 
                             facecolor='none', label='Vehicle'),
            plt.Line2D([0], [0], color=self.COLORS['sidewalk'], linewidth=2.0, label='Sidewalk'),
        ]
        fig.legend(handles=legend_elements, loc='upper center', 
                  bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=10, frameon=True)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Paper风格可视化已保存至: {save_path}")
        
        return fig
    
    def _draw_hdmap_lines(self, ax, mask: np.ndarray):
        """绘制HDMap风格的线条"""
        H, W = mask.shape
        
        # 绘制道路边界 (蓝色)
        road_mask = (mask == 1).astype(np.uint8)
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) >= 3:
                pts = cnt.reshape(-1, 2)
                pts_closed = np.vstack([pts, pts[0]])
                ax.plot(pts_closed[:, 0], pts_closed[:, 1], 
                       color=self.COLORS['driveable'], linewidth=2.0, alpha=0.9)
        
        # 绘制车辆 (红色矩形)
        vehicle_mask = (mask == 2).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(vehicle_mask, connectivity=8)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area >= 10:
                rect = mpatches.Rectangle((x, y), w, h, 
                                        linewidth=2.5, 
                                        edgecolor=self.COLORS['vehicle'], 
                                        facecolor='none',
                                        alpha=0.9)
                ax.add_patch(rect)
        
        # 绘制人行道 (绿色)
        sidewalk_mask = (mask == 3).astype(np.uint8)
        contours, _ = cv2.findContours(sidewalk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) >= 3:
                pts = cnt.reshape(-1, 2)
                pts_closed = np.vstack([pts, pts[0]])
                ax.plot(pts_closed[:, 0], pts_closed[:, 1], 
                       color=self.COLORS['sidewalk'], linewidth=1.8, alpha=0.9)
    
    def create_simple_paper_figure(self,
                                   save_path: str = "paper_demo.png",
                                   figsize: Tuple[int, int] = (14, 7),
                                   dpi: int = 200):
        """
        创建简化版Paper对比图（使用模拟数据）
        """
        print("=" * 60)
        print("创建 Paper 风格对比图")
        print("=" * 60)
        
        # 创建模拟相机图像 (6个视角)
        print("[1/3] 生成模拟相机图像...")
        camera_images = []
        for i in range(6):
            # 模拟相机图像 (224x224)
            img = np.ones((224, 224, 3), dtype=np.uint8) * 220
            
            # 添加道路
            cv2.rectangle(img, (50, 100), (174, 140), (180, 180, 180), -1)
            
            # 添加车辆
            if i % 2 == 0:
                cv2.rectangle(img, (90, 110), (110, 130), (100, 100, 255), -1)
            
            # 添加树木背景
            for _ in range(5):
                x, y = np.random.randint(0, 224, 2)
                cv2.circle(img, (x, y), 10, (150, 200, 150), -1)
            
            camera_images.append(img)
        
        # 创建模拟分割掩码
        print("[2/3] 生成分割掩码...")
        H, W = 200, 200
        
        # GT掩码
        gt_mask = np.zeros((H, W), dtype=np.uint8)
        gt_mask[90:110, 20:180] = 1  # 道路
        gt_mask[85:95, 95:105] = 2   # 车辆
        gt_mask[110:130, 20:50] = 3  # 人行道
        gt_mask[110:130, 150:180] = 3
        
        # M2掩码 (噪声更多)
        m2_mask = gt_mask.copy()
        noise_m2 = np.random.rand(H, W) < 0.1
        m2_mask[noise_m2] = 0
        
        # M3掩码 (中等)
        m3_mask = gt_mask.copy()
        noise_m3 = np.random.rand(H, W) < 0.05
        m3_mask[noise_m3] = 0
        
        # M6掩码 (接近GT)
        m6_mask = gt_mask.copy()
        noise_m6 = np.random.rand(H, W) < 0.02
        m6_mask[noise_m6] = 0
        
        segmentation_masks = [gt_mask, m2_mask, m3_mask, m6_mask]
        mask_labels = ['GT', 'M2', 'M3', 'M6']
        
        # 生成完整图
        print("[3/3] 生成对比图...")
        fig = self.create_paper_figure(
            camera_images=camera_images,
            segmentation_masks=segmentation_masks,
            mask_labels=mask_labels,
            save_path=save_path,
            figsize=figsize,
            dpi=dpi
        )
        
        print()
        print("=" * 60)
        print("完成！")
        print(f"输出: {save_path}")
        print("=" * 60)
        
        return fig


# ========== 使用示例 ==========

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    
    print()
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 8 + "BEV-TextCLIP Paper风格对比图生成器" + " " * 12 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # 创建可视化器
    visualizer = PaperVisualizer(output_dir=r'G:\YMSJ\gaibandianzhen\BEV-TextCLIP\visualization_results\paper_style')
    
    # 生成Paper对比图
    fig = visualizer.create_simple_paper_figure(
        save_path=r'G:\YMSJ\gaibandianzhen\BEV-TextCLIP\visualization_results\paper_style\paper_comparison.png',
        figsize=(16, 7),
        dpi=200
    )
    
    print()
    print("使用说明:")
    print("  1. 准备您的6个相机图像 (列表格式)")
    print("  2. 准备4个分割掩码: GT, M2, M3, M6")
    print("  3. 调用 visualizer.create_paper_figure() 生成对比图")
    print()
    print("示例代码:")
    print("  visualizer = PaperVisualizer()")
    print("  visualizer.create_paper_figure(")
    print("      camera_images=[img1, img2, ...],  # 6个相机图像")
    print("      segmentation_masks=[gt, m2, m3, m6],  # 4个分割结果")
    print("      mask_labels=['GT', 'M2', 'M3', 'M6'],")
    print("      save_path='output.png'")
    print("  )")
    print()
    print("BY ChangXiu.Huang")
