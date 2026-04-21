#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEV-TextCLIP Paper可视化 (HDMapNet风格)

基于您提供的参考图实现：
- Images: 6个相机视角 (2行3列)
- GT/M2/M3/M6: 线条风格分割结果
- 参考: HDMapNet论文可视化风格

日期: 2026-03-16
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Optional
import cv2
import os


class HDMapNetVisualizer:
    """HDMapNet风格可视化"""
    
    # 专业配色
    COLORS = {
        'road': '#0066CC',      # 深蓝色 - 道路边界
        'lane_divider': '#0066CC',  # 蓝色 - 车道分隔线
        'vehicle': '#CC0000',    # 深红色 - 车辆
        'sidewalk': '#00AA00',   # 深绿色 - 人行道
    }
    
    def __init__(self, output_dir: str = "visualization_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_hdmapnet_figure(self,
                              camera_images: List[np.ndarray],
                              bev_masks: List[np.ndarray],
                              labels: List[str],
                              save_path: str,
                              figsize: Tuple[int, int] = (16, 7),
                              dpi: int = 300) -> plt.Figure:
        """
        创建HDMapNet风格的对比图
        
        布局: [Images 2x3] | [GT] [M2] [M3] [M6]
        """
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor('white')
        
        n_masks = len(bev_masks)
        
        # 定义网格布局
        # Images占左侧大块，分割结果占右侧4列
        gs = GridSpec(2, 5, figure=fig,
                     width_ratios=[2.2, 1, 1, 1, 1],
                     height_ratios=[1, 1],
                     wspace=0.08, hspace=0.05,
                     left=0.01, right=0.99, top=0.9, bottom=0.06)
        
        # ========== 左侧: 6个相机图像 ==========
        cam_positions = [
            (0, 0, 'Cam 1'),  # 左上
            (0, 1, 'Cam 2'),  # 中上
            (0, 2, 'Cam 3'),  # 右上
            (1, 0, 'Cam 4'),  # 左下
            (1, 1, 'Cam 5'),  # 中下
            (1, 2, 'Cam 6'),  # 右下
        ]
        
        for idx, (img, (row, col_span, title)) in enumerate(zip(camera_images, cam_positions)):
            # 在Images区域创建子图
            ax = fig.add_subplot(gs[row, 0])
            ax.imshow(img)
            ax.axis('off')
            
            # 添加相机标签
            if idx == 0:
                ax.set_title('Cam 1', fontsize=10, pad=3)
            elif idx == 1:
                ax.set_title('Cam 2', fontsize=10, pad=3)
            elif idx == 2:
                ax.set_title('Cam 3', fontsize=10, pad=3)
            elif idx == 3:
                ax.set_title('Cam 4', fontsize=10, pad=3)
            elif idx == 4:
                ax.set_title('Cam 5', fontsize=10, pad=3)
            elif idx == 5:
                ax.set_title('Cam 6', fontsize=10, pad=3)
        
        # ========== 右侧: 分割结果 ==========
        for idx, (mask, label) in enumerate(zip(bev_masks, labels)):
            col = idx + 1  # 从第2列开始
            
            # 跨2行的子图
            ax = fig.add_subplot(gs[:, col])
            ax.set_facecolor('white')
            
            # 绘制HDMapNet风格分割
            self._draw_hdmapnet_style(ax, mask)
            
            # 标题
            ax.set_title(label, fontsize=13, fontweight='bold', pad=10)
            ax.set_xlim(0, mask.shape[1])
            ax.set_ylim(mask.shape[0], 0)
            ax.set_aspect('equal')
            ax.axis('off')
        
        # Images标签
        fig.text(0.12, 0.01, 'Images', ha='center', 
                fontsize=14, fontweight='bold')
        
        # 图例
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
                  bbox_to_anchor=(0.5, -0.01),
                  ncol=3,
                  fontsize=11,
                  frameon=True,
                  fancybox=False,
                  edgecolor='black',
                  facecolor='white')
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none',
                       pad_inches=0.05)
            print(f"已保存: {save_path}")
        
        return fig
    
    def _draw_hdmapnet_style(self, ax, mask: np.ndarray):
        """绘制HDMapNet风格"""
        H, W = mask.shape
        
        # 道路边界
        road = (mask == 1).astype(np.uint8) * 255
        contours, _ = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) >= 3:
                epsilon = 0.003 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) >= 3:
                    pts = approx.reshape(-1, 2)
                    pts = np.vstack([pts, pts[0]])  # 闭合
                    ax.plot(pts[:, 0], pts[:, 1], 
                           color=self.COLORS['road'], 
                           linewidth=3.5, 
                           alpha=0.95)
        
        # 车道分隔线 (虚线效果)
        road_skel = self._get_skeleton(road)
        lines = self._extract_centerlines(road_skel)
        for line in lines:
            if len(line) >= 2:
                ax.plot(line[:, 0], line[:, 1], 
                       color=self.COLORS['lane_divider'], 
                       linewidth=2, 
                       alpha=0.9,
                       linestyle='-')
        
        # 车辆
        vehicle = (mask == 2).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(vehicle, 8)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area >= 5:
                rect = mpatches.Rectangle((x, y), w, h, 
                                        linewidth=2.5, 
                                        edgecolor=self.COLORS['vehicle'], 
                                        facecolor='none',
                                        alpha=0.95)
                ax.add_patch(rect)
        
        # 人行道
        sidewalk = (mask == 3).astype(np.uint8) * 255
        contours, _ = cv2.findContours(sidewalk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) >= 3:
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) >= 3:
                    pts = approx.reshape(-1, 2)
                    pts = np.vstack([pts, pts[0]])
                    ax.plot(pts[:, 0], pts[:, 1], 
                           color=self.COLORS['sidewalk'], 
                           linewidth=2.5, 
                           alpha=0.9)
    
    def _get_skeleton(self, mask: np.ndarray) -> np.ndarray:
        """获取骨架"""
        from skimage.morphology import skeletonize
        return skeletonize(mask > 0).astype(np.uint8)
    
    def _extract_centerlines(self, skeleton: np.ndarray) -> List[np.ndarray]:
        """提取中心线"""
        lines = []
        contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) >= 10:
                pts = cnt.reshape(-1, 2)
                # 拟合直线
                if len(pts) >= 2:
                    lines.append(pts)
        return lines
    
    def create_demo(self, save_path: str = "hdmapnet_viz.png"):
        """生成演示图"""
        print("=" * 60)
        print("生成 HDMapNet 风格可视化")
        print("=" * 60)
        
        # 生成相机图像
        print("[1/2] 生成相机图像...")
        camera_images = []
        np.random.seed(123)
        
        for cam_id in range(6):
            # 创建更真实的道路场景
            img = np.ones((180, 320, 3), dtype=np.uint8) * 240
            
            # 天空
            img[:50, :] = [180, 200, 230]
            
            # 道路 (透视效果)
            if cam_id < 3:  # 前向
                # 左车道
                pts = np.array([[60, 150], [140, 80], [180, 80], [100, 150]], np.int32)
                cv2.fillPoly(img, [pts], [160, 160, 160])
                # 右车道
                pts = np.array([[260, 150], [180, 80], [220, 80], [280, 150]], np.int32)
                cv2.fillPoly(img, [pts], [160, 160, 160])
                # 中心车道线
                cv2.line(img, (160, 150), (200, 80), [255, 255, 200], 2)
                
                # 远处车辆
                if cam_id == 1:
                    cv2.rectangle(img, (190, 90), (210, 110), [100, 100, 200], -1)
            else:  # 后向
                cv2.rectangle(img, (40, 120), (280, 160), [160, 160, 160], -1)
            
            # 树木
            for _ in range(6):
                x = np.random.randint(0, 320)
                y = np.random.randint(20, 80)
                cv2.circle(img, (x, y), 12, [80, 140, 80], -1)
            
            camera_images.append(img)
        
        # 生成分割掩码
        print("[2/2] 生成分割掩码...")
        H, W = 200, 200
        
        # GT
        gt_mask = np.zeros((H, W), dtype=np.uint8)
        # 道路
        cv2.rectangle(gt_mask, (20, 90), (180, 110), 1, -1)
        cv2.rectangle(gt_mask, (90, 60), (110, 140), 1, -1)
        # 车辆
        cv2.rectangle(gt_mask, (95, 85), (105, 95), 2, -1)
        # 人行道
        cv2.rectangle(gt_mask, (20, 110), (180, 130), 3, -1)
        cv2.rectangle(gt_mask, (20, 70), (180, 90), 3, -1)
        
        # 不同质量的预测
        m2_mask = gt_mask.copy()
        noise2 = np.random.rand(H, W) < 0.12
        m2_mask[noise2] = 0
        
        m3_mask = gt_mask.copy()
        noise3 = np.random.rand(H, W) < 0.06
        m3_mask[noise3] = 0
        
        m6_mask = gt_mask.copy()
        noise6 = np.random.rand(H, W) < 0.02
        m6_mask[noise6] = 0
        
        bev_masks = [gt_mask, m2_mask, m3_mask, m6_mask]
        labels = ['GT', 'M2', 'M3', 'M6']
        
        # 生成
        fig = self.create_hdmapnet_figure(
            camera_images=camera_images,
            bev_masks=bev_masks,
            labels=labels,
            save_path=save_path,
            figsize=(16, 6.5),
            dpi=300
        )
        
        print(f"\n输出: {save_path}")
        return fig


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    
    visualizer = HDMapNetVisualizer(
        output_dir=r'G:\YMSJ\gaibandianzhen\BEV-TextCLIP\visualization_results\paper_style'
    )
    
    fig = visualizer.create_demo(
        save_path=r'G:\YMSJ\gaibandianzhen\BEV-TextCLIP\visualization_results\paper_style\hdmapnet_style.png'
    )
    
    print("\n" + "=" * 60)
    print("使用方法:")
    print("  visualizer.create_hdmapnet_figure(")
    print("      camera_images=[img1,...],  # 6个相机图像")
    print("      bev_masks=[gt, m2, m3, m6],")
    print("      labels=['GT','M2','M3','M6'],")
    print("      save_path='output.png'")
    print("  )")
    print("=" * 60)
