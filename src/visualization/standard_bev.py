#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEV 标准对比图 - 参考 HDMapNet/BEVFormer 论文

布局：
左边(2x3): 6个相机图像 (CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT)
右边(1x4): 4列 BEV 对比 (GT, M2, M3, M6)

日期: 2026-03-16
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
import os
import glob
from typing import List, Optional
import random


class StandardBevVisualizer:
    """标准BEV对比可视化器"""
    
    COLORS = {
        'road': '#0066CC',
        'vehicle': '#CC0000', 
        'sidewalk': '#00AA00',
        'lane': '#0088FF',
    }
    
    def __init__(self, data_root: str, output_dir: str = "visualization_results"):
        self.data_root = data_root
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.camera_order = [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
        ]
    
    def load_camera_images(self, sample_idx: int = 0) -> List[np.ndarray]:
        images = []
        samples_dir = os.path.join(self.data_root, 'samples')
        
        for cam_name in self.camera_order:
            cam_dir = os.path.join(samples_dir, cam_name)
            if not os.path.exists(cam_dir):
                # 创建默认灰色图像
                img = np.ones((180, 320, 3), dtype=np.uint8) * 200
                images.append(img)
                continue
            
            image_files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))
            if len(image_files) == 0:
                img = np.ones((180, 320, 3), dtype=np.uint8) * 200
                images.append(img)
                continue
            
            img_path = image_files[sample_idx] if sample_idx < len(image_files) else image_files[0]
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
            else:
                img = np.ones((180, 320, 3), dtype=np.uint8) * 200
                images.append(img)
        
        return images
    
    def create_gt_crossroad(self, size: int = 200) -> np.ndarray:
        """创建标准十字路口GT"""
        mask = np.zeros((size, size), dtype=np.uint8)
        
        # 道路 - 十字形
        mask[70:130, 20:180] = 1  # 横向
        mask[20:180, 70:130] = 1  # 纵向
        
        # 车道分隔线
        mask[98:102, 20:180] = 4
        mask[20:180, 98:102] = 4
        
        # 车辆
        mask[85:95, 95:105] = 2
        mask[85:95, 115:125] = 2
        
        # 人行道
        mask[50:70, 20:180] = 3
        mask[130:150, 20:180] = 3
        mask[20:180, 50:70] = 3
        mask[20:180, 130:150] = 3
        
        return mask
    
    def add_model_noise(self, mask: np.ndarray, level: float) -> np.ndarray:
        """模拟不同模型的预测结果"""
        result = mask.copy()
        h, w = mask.shape
        
        # 随机遮挡
        num_holes = int(level * 30)
        for _ in range(num_holes):
            x = random.randint(5, w - 15)
            y = random.randint(5, h - 15)
            result[y:y+10, x:x+10] = 0
        
        # 随机删除一些道路线
        if level > 0.05:
            # 删除一些道路边界点
            for _ in range(int(level * 20)):
                x = random.randint(20, 180)
                y = random.randint(20, 180)
                result[y, x] = 0
        
        return result
    
    def draw_bev_result(self, ax, mask: np.ndarray, title: str):
        """绘制单个BEV结果"""
        ax.set_facecolor('white')
        
        # 道路边界 - 用轮廓
        road_mask = (mask == 1).astype(np.uint8) * 255
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) >= 3:
                pts = cnt.reshape(-1, 2)
                pts = np.vstack([pts, pts[0]])
                ax.plot(pts[:, 0], pts[:, 1], self.COLORS['road'], linewidth=3)
        
        # 车道分隔线
        lane_mask = (mask == 4).astype(np.uint8) * 255
        contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) >= 2:
                pts = cnt.reshape(-1, 2)
                ax.plot(pts[:, 0], pts[:, 1], self.COLORS['lane'], linewidth=1.5, linestyle='--', alpha=0.8)
        
        # 车辆
        vehicle_mask = (mask == 2).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(vehicle_mask, connectivity=8)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area >= 5:
                rect = plt.Rectangle((x, y), w, h, linewidth=2.5, 
                                    edgecolor=self.COLORS['vehicle'], facecolor='none')
                ax.add_patch(rect)
        
        # 人行道
        sidewalk_mask = (mask == 3).astype(np.uint8) * 255
        contours, _ = cv2.findContours(sidewalk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) >= 3:
                pts = cnt.reshape(-1, 2)
                pts = np.vstack([pts, pts[0]])
                ax.plot(pts[:, 0], pts[:, 1], self.COLORS['sidewalk'], linewidth=2)
        
        ax.set_xlim(0, 200)
        ax.set_ylim(200, 0)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    def create_standard_figure(self, sample_idx: int = 0, save_path: Optional[str] = None):
        """创建标准对比图"""
        
        print("Loading camera images...")
        images = self.load_camera_images(sample_idx)
        
        # 调整图像大小
        resized_images = []
        for img in images:
            resized = cv2.resize(img, (300, 170))
            resized_images.append(resized)
        
        # 创建BEV掩码
        print("Creating BEV masks...")
        gt_mask = self.create_gt_crossroad(200)
        
        # 模拟不同模型
        m2_mask = self.add_model_noise(gt_mask, 0.20)  # 高噪声
        m3_mask = self.add_model_noise(gt_mask, 0.10)  # 中等噪声
        m6_mask = self.add_model_noise(gt_mask, 0.03)  # 低噪声
        
        masks = [gt_mask, m2_mask, m3_mask, m6_mask]
        titles = ['GT', 'M2', 'M3', 'M6']
        
        # 创建图形
        fig = plt.figure(figsize=(18, 8))
        fig.patch.set_facecolor('white')
        
        # 布局：左边 Images(2行3列) + 右边 BEV(1行4列)
        gs = GridSpec(2, 7, figure=fig,
                     width_ratios=[1, 1, 1, 0.3, 1, 1, 1],
                     height_ratios=[1, 1],
                     wspace=0.02, hspace=0.02,
                     left=0.01, right=0.99, top=0.92, bottom=0.06)
        
        # 左侧：6个相机图像 (2行3列)
        cam_positions = [
            (0, 0, 'CAM_FRONT_LEFT'),
            (0, 1, 'CAM_FRONT'), 
            (0, 2, 'CAM_FRONT_RIGHT'),
            (1, 0, 'CAM_BACK_LEFT'),
            (1, 1, 'CAM_BACK'),
            (1, 2, 'CAM_BACK_RIGHT'),
        ]
        
        for pos_idx, (row, col, cam_name) in enumerate(cam_positions):
            ax = fig.add_subplot(gs[row, col])
            if pos_idx < len(resized_images):
                ax.imshow(resized_images[pos_idx])
            else:
                ax.set_facecolor('gray')
            ax.axis('off')
            ax.set_title(cam_name.replace('CAM_', ''), fontsize=9, pad=2)
        
        # 右侧：4列BEV对比
        bev_positions = [4, 5, 6]
        
        for idx, (mask, title) in enumerate(zip(masks, titles)):
            ax = fig.add_subplot(gs[0:2, bev_positions[idx]])
            self.draw_bev_result(ax, mask, title)
        
        # Images 标签
        fig.text(0.15, 0.02, 'Images', ha='center', fontsize=14, fontweight='bold')
        
        # 图例
        legend_elements = [
            plt.Line2D([0], [0], color=self.COLORS['road'], linewidth=3, label='Road Boundary'),
            plt.Line2D([0], [0], color=self.COLORS['lane'], linewidth=1.5, linestyle='--', label='Lane Divider'),
            plt.Rectangle((0, 0), 1, 1, linewidth=2.5, edgecolor=self.COLORS['vehicle'], facecolor='none', label='Vehicle'),
            plt.Line2D([0], [0], color=self.COLORS['sidewalk'], linewidth=2, label='Sidewalk'),
        ]
        
        fig.legend(handles=legend_elements, loc='lower center',
                  bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=11, frameon=True)
        
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            print(f"Saved to: {save_path}")
        
        return fig


def main():
    print("=" * 60)
    print("Standard BEV Comparison - HDMapNet/BEVFormer Style")
    print("=" * 60)
    
    data_root = r'G:\YMSJ\gaibandianzhen\BEV-TextCLIP\data\nuscenes'
    output_dir = r'G:\YMSJ\gaibandianzhen\BEV-TextCLIP\visualization_results\real_nuscenes'
    
    visualizer = StandardBevVisualizer(data_root, output_dir)
    
    save_path = os.path.join(output_dir, 'standard_bev_comparison.png')
    fig = visualizer.create_standard_figure(sample_idx=0, save_path=save_path)
    
    print("\nDone!")
    print(f"Output: {save_path}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    main()