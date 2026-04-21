#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEV-TextCLIP Paper风格可视化 (专业版)

参考: HDMapNet, BEVFormer等顶会论文的可视化风格
生成高质量对比图：
- 左边: 6个相机视角 (2x3网格)
- 右边: GT/M2/M3/M6对比 (4列)
- 专业线条风格，带图例

日期: 2026-03-16
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Optional
import cv2
import os


class PaperVisualizer:
    """Paper风格专业可视化"""
    
    # HDMap专业配色
    COLORS = {
        'road': '#1E90FF',       # 道奇蓝 - 道路边界
        'lane': '#1E90FF',       # 道奇蓝 - 车道线
        'vehicle': '#DC143C',     # 猩红 - 车辆
        'sidewalk': '#32CD32',    # 酸橙绿 - 人行道
        'crosswalk': '#FFD700',   # 金色 - 斑马线
    }
    
    def __init__(self, output_dir: str = "visualization_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_paper_figure_v2(self,
                               camera_images: List[np.ndarray],
                               segmentation_masks: List[np.ndarray],
                               mask_labels: List[str],
                               save_path: str,
                               figsize: Tuple[int, int] = (16, 7),
                               dpi: int = 300) -> plt.Figure:
        """
        创建专业Paper风格对比图
        
        布局:
        [Images 2x3] [GT] [M2] [M3] [M6]
        """
        n_masks = len(segmentation_masks)
        
        # 创建图形
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor('white')
        
        # 定义网格: 2行, 5列 (1列Images + 4列分割结果)
        # Images占2行，分割结果各占2行
        gs = GridSpec(2, 5, figure=fig, 
                     width_ratios=[2.5, 1, 1, 1, 1],
                     height_ratios=[1, 1],
                     wspace=0.05, hspace=0.05,
                     left=0.02, right=0.98, top=0.92, bottom=0.08)
        
        # ========== 左边: 6个相机图像 ==========
        camera_titles = ['Front', 'Front Right', 'Front Left',
                        'Back', 'Back Right', 'Back Left']
        
        for idx, (img, title) in enumerate(zip(camera_images, camera_titles)):
            row = idx // 3
            col = idx % 3
            
            # 在Images列中创建子图
            ax = fig.add_subplot(gs[row, 0])
            ax.imshow(img)
            ax.axis('off')
            
            # 添加相机视角标题
            if idx == 0:
                ax.set_title('Cam 1', fontsize=9, pad=2)
            elif idx == 1:
                ax.set_title('Cam 2', fontsize=9, pad=2)
            elif idx == 2:
                ax.set_title('Cam 3', fontsize=9, pad=2)
            elif idx == 3:
                ax.set_title('Cam 4', fontsize=9, pad=2)
            elif idx == 4:
                ax.set_title('Cam 5', fontsize=9, pad=2)
            elif idx == 5:
                ax.set_title('Cam 6', fontsize=9, pad=2)
        
        # ========== 右边: 4个分割结果 ==========
        for idx, (mask, label) in enumerate(zip(segmentation_masks, mask_labels)):
            col = idx + 1  # 从第2列开始
            
            # 创建跨2行的子图
            ax = fig.add_subplot(gs[:, col])
            ax.set_facecolor('white')
            
            # 绘制HDMap风格分割
            self._draw_hdmap_v2(ax, mask)
            
            # 设置标题 (GT, M2, M3, M6)
            ax.set_title(label, fontsize=13, fontweight='bold', 
                        pad=8, color='black')
            
            # 设置坐标轴
            ax.set_xlim(0, mask.shape[1])
            ax.set_ylim(mask.shape[0], 0)
            ax.set_aspect('equal')
            ax.axis('off')
        
        # 添加Images标签
        fig.text(0.135, 0.02, 'Images', ha='center', 
                fontsize=14, fontweight='bold', color='black')
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], color=self.COLORS['road'], 
                      linewidth=3, label='Road Boundary'),
            mpatches.Rectangle((0, 0), 1, 1, linewidth=2.5, 
                             edgecolor=self.COLORS['vehicle'], 
                             facecolor='none', label='Vehicle'),
            plt.Line2D([0], [0], color=self.COLORS['sidewalk'], 
                      linewidth=2.5, label='Sidewalk'),
        ]
        
        fig.legend(handles=legend_elements, 
                  loc='lower center',
                  bbox_to_anchor=(0.5, -0.02),
                  ncol=3,
                  fontsize=11,
                  frameon=True,
                  fancybox=True,
                  shadow=False)
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none',
                       pad_inches=0.1)
            print(f"保存至: {save_path}")
        
        return fig
    
    def _draw_hdmap_v2(self, ax, mask: np.ndarray):
        """绘制HDMap风格线条 (改进版)"""
        H, W = mask.shape
        
        # 绘制道路边界 (蓝色粗线)
        road_mask = (mask == 1).astype(np.uint8) * 255
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if len(cnt) >= 3:
                # 简化轮廓
                epsilon = 0.002 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) >= 3:
                    pts = approx.reshape(-1, 2)
                    # 闭合轮廓
                    pts_closed = np.vstack([pts, pts[0]])
                    ax.plot(pts_closed[:, 0], pts_closed[:, 1], 
                           color=self.COLORS['road'], 
                           linewidth=3.5, 
                           alpha=0.95,
                           solid_capstyle='round',
                           solid_joinstyle='round')
        
        # 绘制车辆 (红色矩形)
        vehicle_mask = (mask == 2).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(vehicle_mask, connectivity=8)
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area >= 8:
                rect = mpatches.Rectangle((x, y), w, h, 
                                        linewidth=2.5, 
                                        edgecolor=self.COLORS['vehicle'], 
                                        facecolor='none',
                                        alpha=0.95,
                                        joinstyle='round')
                ax.add_patch(rect)
        
        # 绘制人行道 (绿色)
        sidewalk_mask = (mask == 3).astype(np.uint8) * 255
        contours, _ = cv2.findContours(sidewalk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if len(cnt) >= 3:
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) >= 3:
                    pts = approx.reshape(-1, 2)
                    pts_closed = np.vstack([pts, pts[0]])
                    ax.plot(pts_closed[:, 0], pts_closed[:, 1], 
                           color=self.COLORS['sidewalk'], 
                           linewidth=2.5, 
                           alpha=0.9,
                           solid_capstyle='round')
    
    def create_demo(self, save_path: str = "paper_v2.png"):
        """生成演示图"""
        print("=" * 60)
        print("生成 Paper 风格对比图 (v2 - 专业版)")
        print("=" * 60)
        
        # 生成模拟相机图像
        print("[1/2] 生成相机图像...")
        camera_images = []
        np.random.seed(42)
        
        for i in range(6):
            # 创建更真实的道路场景
            img = np.ones((224, 224, 3), dtype=np.uint8) * 240
            
            # 绘制天空
            img[:80, :] = [200, 210, 230]
            
            # 绘制道路 (透视效果)
            if i < 3:  # 前向相机
                # 左车道线
                pts_left = np.array([[80, 150], [110, 100], [100, 100], [60, 150]], np.int32)
                cv2.fillPoly(img, [pts_left], [180, 180, 180])
                # 右车道线
                pts_right = np.array([[164, 150], [134, 100], [124, 100], [144, 150]], np.int32)
                cv2.fillPoly(img, [pts_right], [180, 180, 180])
                
                # 添加车辆
                if i == 0:
                    cv2.rectangle(img, (105, 110), (115, 125), [100, 100, 200], -1)
                
            else:  # 后向相机
                cv2.rectangle(img, (50, 120), (174, 160), [180, 180, 180], -1)
            
            # 添加树木
            for _ in range(8):
                x = np.random.randint(0, 224)
                y = np.random.randint(20, 100)
                cv2.circle(img, (x, y), 15, [100, 160, 100], -1)
            
            camera_images.append(img)
        
        # 生成分割掩码
        print("[2/2] 生成分割结果...")
        H, W = 200, 200
        
        # GT
        gt_mask = np.zeros((H, W), dtype=np.uint8)
        # 道路
        gt_mask[90:110, 20:180] = 1
        gt_mask[60:140, 90:110] = 1
        # 车辆
        gt_mask[85:95, 95:105] = 2
        gt_mask[125:135, 95:105] = 2
        gt_mask[85:95, 30:40] = 2
        # 人行道
        gt_mask[60:80, 85:115] = 3
        gt_mask[120:140, 85:115] = 3
        
        # M2 (噪声多)
        m2_mask = gt_mask.copy()
        noise = np.random.rand(H, W) < 0.15
        m2_mask[noise] = 0
        
        # M3 (中等)
        m3_mask = gt_mask.copy()
        noise = np.random.rand(H, W) < 0.08
        m3_mask[noise] = 0
        
        # M6 (接近GT)
        m6_mask = gt_mask.copy()
        noise = np.random.rand(H, W) < 0.03
        m6_mask[noise] = 0
        
        segmentation_masks = [gt_mask, m2_mask, m3_mask, m6_mask]
        mask_labels = ['GT', 'M2', 'M3', 'M6']
        
        # 生成完整图
        fig = self.create_paper_figure_v2(
            camera_images=camera_images,
            segmentation_masks=segmentation_masks,
            mask_labels=mask_labels,
            save_path=save_path,
            figsize=(16, 6.5),
            dpi=300
        )
        
        print(f"\n完成！输出: {save_path}")
        return fig


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    
    # 创建可视化器
    visualizer = PaperVisualizer(output_dir=r'G:\YMSJ\gaibandianzhen\BEV-TextCLIP\visualization_results\paper_style')
    
    # 生成高质量对比图
    fig = visualizer.create_demo(
        save_path=r'G:\YMSJ\gaibandianzhen\BEV-TextCLIP\visualization_results\paper_style\paper_v2_professional.png'
    )
    
    print("\n" + "=" * 60)
    print("使用方法:")
    print("  visualizer = PaperVisualizer()")
    print("  visualizer.create_paper_figure_v2(")
    print("      camera_images=[img1, img2, ...],  # 6个相机图像")
    print("      segmentation_masks=[gt, m2, m3, m6],")
    print("      mask_labels=['GT', 'M2', 'M3', 'M6']")
    print("  )")
    print("=" * 60)
