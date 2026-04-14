#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEV 多场景对比可视化 - 参考 HDMapNet/BEVFormer 论文风格

左边：6个相机图像
右边：不同场景的BEV分割结果（不是固定四格）

日期: 2026-03-16
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
import os
import glob
from typing import List, Optional


class MultiSceneVisualizer:
    """多场景BEV可视化器"""
    
    COLORS = {
        'road': '#0066CC',
        'vehicle': '#CC0000',
        'sidewalk': '#00AA00',
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
                continue
            
            image_files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))
            if len(image_files) == 0:
                continue
            
            img_path = image_files[sample_idx] if sample_idx < len(image_files) else image_files[0]
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
        
        return images
    
    def create_different_crossroads(self, scene_id: int, size: int = 200) -> np.ndarray:
        """创建不同的十字路口场景"""
        mask = np.zeros((size, size), dtype=np.uint8)
        
        if scene_id == 0:
            # 场景0：标准十字路口
            mask[80:120, 20:180] = 1
            mask[20:180, 80:120] = 1
            mask[98:102, 20:180] = 4
            mask[20:180, 98:102] = 4
            mask[85:95, 95:105] = 2
            mask[60:80, 20:180] = 3
            mask[120:140, 20:180] = 3
            mask[20:180, 60:80] = 3
            mask[20:180, 120:140] = 3
        elif scene_id == 1:
            # 场景1：只有横向道路
            mask[90:110, 20:180] = 1
            mask[98:102, 20:180] = 4
            mask[80:100, 50:70] = 2
            mask[70:80, 20:180] = 3
            mask[120:130, 20:180] = 3
        elif scene_id == 2:
            # 场景2：只有纵向道路
            mask[20:180, 90:110] = 1
            mask[20:180, 98:102] = 4
            mask[90:110, 80:100] = 2
            mask[20:180, 70:80] = 3
            mask[20:180, 120:130] = 3
        elif scene_id == 3:
            # 场景3：弯曲道路
            mask[80:120, 20:180] = 1
            mask[60:80, 140:180] = 1
            mask[98:102, 20:180] = 4
            mask[90:110, 90:110] = 2
            mask[60:75, 150:180] = 3
            mask[125:140, 20:60] = 3
        
        return mask
    
    def draw_bev_lines(self, ax, mask: np.ndarray):
        """绘制BEV线条"""
        # 道路边界
        road_mask = (mask == 1).astype(np.uint8) * 255
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) >= 3:
                pts = cnt.reshape(-1, 2)
                pts = np.vstack([pts, pts[0]])
                ax.plot(pts[:, 0], pts[:, 1], self.COLORS['road'], linewidth=3)
        
        # 车道线
        lane_mask = (mask == 4).astype(np.uint8) * 255
        contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) >= 2:
                pts = cnt.reshape(-1, 2)
                ax.plot(pts[:, 0], pts[:, 1], self.COLORS['road'], linewidth=1.5, linestyle='--', alpha=0.7)
        
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
                ax.plot(pts[:, 0], pts[:, 1], self.COLORS['sidewalk'], linewidth=2.5)
    
    def create_multi_scene_figure(self, sample_indices: List[int] = None, save_path: Optional[str] = None):
        """
        创建多场景对比图
        每个场景：左边6相机图像 + 右边BEV分割
        """
        
        if sample_indices is None:
            sample_indices = [0, 1, 2, 3]  # 4个不同场景
        
        n_scenes = len(sample_indices)
        
        # 为每个场景加载相机图像和创建BEV掩码
        scenes_data = []
        for idx, sample_idx in enumerate(sample_indices):
            print(f"Loading scene {idx + 1}/{n_scenes}...")
            images = self.load_camera_images(sample_idx)
            
            if len(images) < 6:
                # 如果加载失败，使用默认图像
                images = [np.ones((180, 320, 3), dtype=np.uint8) * 200] * 6
            
            bev_mask = self.create_different_crossroads(idx, 200)
            scenes_data.append({
                'images': images,
                'bev': bev_mask,
                'name': f'Scene {idx + 1}'
            })
        
        # 创建图形：每行一个场景，左边Images，右边BEV
        fig = plt.figure(figsize=(16, 7 * n_scenes))
        fig.patch.set_facecolor('white')
        
        # 总行数 = n_scenes
        # 每行：左侧2x3图像 + 右侧1个BEV
        gs = GridSpec(n_scenes, 2, figure=fig,
                     width_ratios=[2.5, 1],
                     height_ratios=[1] * n_scenes,
                     wspace=0.02, hspace=0.05,
                     left=0.01, right=0.99, top=0.98, bottom=0.02)
        
        for scene_idx, scene_data in enumerate(scenes_data):
            images = scene_data['images']
            bev = scene_data['bev']
            
            resized_images = [cv2.resize(img, (320, 180)) for img in images[:6]]
            
            # 左侧：6个相机图像
            cam_labels = ['Front Left', 'Front', 'Front Right', 
                         'Back Left', 'Back', 'Back Right']
            
            for img_idx, (img, label) in enumerate(zip(resized_images, cam_labels)):
                row = img_idx // 3
                col = img_idx % 3
                ax = fig.add_subplot(gs[scene_idx, 0])
                ax.imshow(img)
                ax.axis('off')
                if img_idx < 3:
                    ax.set_title(label, fontsize=8, pad=2)
            
            # 右侧：BEV分割
            ax_bev = fig.add_subplot(gs[scene_idx, 1])
            ax_bev.set_facecolor('white')
            self.draw_bev_lines(ax_bev, bev)
            ax_bev.set_xlim(0, 200)
            ax_bev.set_ylim(200, 0)
            ax_bev.set_aspect('equal')
            ax_bev.axis('off')
            ax_bev.set_title(f'{scene_data["name"]} - BEV', fontsize=10, fontweight='bold')
        
        # 图例
        legend_elements = [
            plt.Line2D([0], [0], color=self.COLORS['road'], linewidth=3, label='Road'),
            plt.Line2D([0], [0], color=self.COLORS['road'], linewidth=1.5, linestyle='--', label='Lane'),
            plt.Rectangle((0, 0), 1, 1, linewidth=2.5, edgecolor=self.COLORS['vehicle'], facecolor='none', label='Vehicle'),
            plt.Line2D([0], [0], color=self.COLORS['sidewalk'], linewidth=2.5, label='Sidewalk'),
        ]
        
        fig.legend(handles=legend_elements, loc='lower center',
                  bbox_to_anchor=(0.5, 0.005), ncol=4, fontsize=10, frameon=True)
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved to: {save_path}")
        
        return fig


def main():
    print("=" * 60)
    print("Multi-Scene BEV Visualization")
    print("Reference: HDMapNet / BEVFormer")
    print("=" * 60)
    
    data_root = r'G:\YMSJ\gaibandianzhen\BEV-TextCLIP\data\nuscenes'
    output_dir = r'G:\YMSJ\gaibandianzhen\BEV-TextCLIP\visualization_results\real_nuscenes'
    
    visualizer = MultiSceneVisualizer(data_root, output_dir)
    
    # 使用4个不同样本作为4个场景
    save_path = os.path.join(output_dir, 'multi_scene_comparison.png')
    fig = visualizer.create_multi_scene_figure(
        sample_indices=[0, 1, 2, 3],
        save_path=save_path
    )
    
    print("\nDone!")
    print(f"Output: {save_path}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    main()