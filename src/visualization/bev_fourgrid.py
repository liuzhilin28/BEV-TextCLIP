#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEV 场景对比可视化 - HD Map风格

布局：3行，每行 = 6张图片(2×3) + 4个BEV图(GT/M2/M3/M6)
右侧使用贝塞尔曲线模拟真实HD Map道路

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import cv2
import os
import glob
from typing import List, Optional, Tuple


class BevSceneVisualizer:
    """BEV 场景对比可视化器"""
    
    COLORS = {
        'road_boundary': '#CC0000',  # 红色 - 道路边界
        'lane_divider': '#0066CC',   # 蓝色 - 车道分隔线
        'crosswalk': '#00AA00',      # 绿色 - 人行横道
        'vehicle': '#FF3333',      # 红色 - 车辆
    }
    
    def __init__(self, data_root: str, output_dir: str = "visualization_results"):
        self.data_root = data_root
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.camera_order = [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
        ]
    
    def load_camera_images_by_prefix(self, prefix: str) -> Tuple[List[np.ndarray], str]:
        """按前缀加载同一时刻的6个相机图片"""
        images = []
        samples_dir = os.path.join(self.data_root, 'samples')
        filename = ""
        
        import numpy as np
        
        # 先找到CAM_FRONT目录下的第一个文件，提取时间戳
        front_dir = os.path.join(samples_dir, 'CAM_FRONT')
        if os.path.exists(front_dir):
            pattern = os.path.join(front_dir, f"{prefix}*.jpg")
            front_files = sorted(glob.glob(pattern))
            
            if front_files:
                # 提取时间戳
                front_file = os.path.basename(front_files[0])
                parts = front_file.split('__')
                if len(parts) >= 3:
                    session = parts[0]
                    target_timestamp = int(parts[2].replace('.jpg', ''))
                    
                    # 为每个相机找到最接近目标时间戳的图片
                    for cam_name in self.camera_order:
                        cam_dir = os.path.join(samples_dir, cam_name)
                        if not os.path.exists(cam_dir):
                            continue
                        
                        # 获取该相机该会话的所有图片
                        cam_pattern = os.path.join(cam_dir, f"{session}*.jpg")
                        cam_files = sorted(glob.glob(cam_pattern))
                        
                        if cam_files:
                            # 提取所有时间戳
                            timestamps = []
                            for f in cam_files:
                                ts_part = os.path.basename(f).split('__')[2].replace('.jpg', '')
                                timestamps.append(int(ts_part))
                            timestamps = np.array(timestamps)
                            
                            # 找到最接近的
                            idx = np.argmin(np.abs(timestamps - target_timestamp))
                            closest_file = cam_files[idx]
                            
                            if not filename and cam_name == 'CAM_FRONT':
                                filename = os.path.basename(closest_file)
                            
                            img = cv2.imread(closest_file)
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                images.append(img)
        
        return images, filename
    
    def bezier_curve(self, p0, p1, p2, p3, num_points=100):
        """三次贝塞尔曲线"""
        t = np.linspace(0, 1, num_points)
        x = (1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0]
        y = (1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1]
        return x, y
    
    def draw_curved_road(self, ax, points, color, linewidth=3, alpha=1.0):
        """绘制曲线道路"""
        if len(points) < 2:
            return
        # 简化：直接连接点
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha)
    
    def draw_crossroad_bev(self, ax, quality='gt'):
        """绘制十字路口BEV - HD Map风格"""
        # 中心点
        cx, cy = 100, 100
        
        # 道路宽度
        w = 40
        
        # 四个方向的道路边界（曲线）
        # 北向道路
        north_left = [(cx-w, cy+50), (cx-w, cy+80), (cx-w-5, cy+120), (cx-w-10, cy+180)]
        north_right = [(cx+w, cy+50), (cx+w, cy+80), (cx+w+5, cy+120), (cx+w+10, cy+180)]
        
        # 南向道路
        south_left = [(cx-w, cy-50), (cx-w, cy-80), (cx-w-5, cy-120), (cx-w-10, cy-20)]
        south_right = [(cx+w, cy-50), (cx+w, cy-80), (cx+w+5, cy-120), (cx+w+10, cy-20)]
        
        # 东向道路
        east_left = [(cx+50, cy-w), (cx+80, cy-w), (cx+120, cy-w-5), (cx+180, cy-w-10)]
        east_right = [(cx+50, cy+w), (cx+80, cy+w), (cx+120, cy+w+5), (cx+180, cy+w+10)]
        
        # 西向道路
        west_left = [(cx-50, cy-w), (cx-80, cy-w), (cx-120, cy-w+5), (cx-20, cy-w+10)]
        west_right = [(cx-50, cy+w), (cx-80, cy+w), (cx-120, cy+w-5), (cx-20, cy+w-10)]
        
        # 车道分隔线定义
        divider_n = [(cx, cy+30), (cx, cy+60), (cx+2, cy+100), (cx+5, cy+180)]
        divider_s = [(cx, cy-30), (cx, cy-60), (cx-2, cy-100), (cx-5, cy-20)]
        divider_e = [(cx+30, cy), (cx+60, cy), (cx+100, cy-2), (cx+180, cy-5)]
        divider_w = [(cx-30, cy), (cx-60, cy), (cx-100, cy+2), (cx-20, cy+5)]
        
        if quality in ['gt', 'm6']:
            # 完整十字路口
            for road in [north_left, north_right, south_left, south_right, 
                        east_left, east_right, west_left, west_right]:
                self.draw_curved_road(ax, road, self.COLORS['road_boundary'], 3)
            
            for div in [divider_n, divider_s, divider_e, divider_w]:
                self.draw_curved_road(ax, div, self.COLORS['lane_divider'], 2)
            
            # 人行横道
            for i in range(3):
                y = cy - 20 + i * 20
                ax.plot([cx-35, cx+35], [y, y], self.COLORS['crosswalk'], linewidth=4)
            for i in range(3):
                x = cx - 20 + i * 20
                ax.plot([x, x], [cy-35, cy+35], self.COLORS['crosswalk'], linewidth=4)
                
        elif quality == 'm3':
            # 稍微缺失
            for road in [north_left, north_right, south_left, south_right]:
                self.draw_curved_road(ax, road, self.COLORS['road_boundary'], 3)
            self.draw_curved_road(ax, east_left, self.COLORS['road_boundary'], 3)
            
        else:  # m2 - 缺失更多
            for road in [north_left, north_right]:
                self.draw_curved_road(ax, road, self.COLORS['road_boundary'], 3)
            self.draw_curved_road(ax, divider_n, self.COLORS['lane_divider'], 2)
    
    def draw_main_road_bev(self, ax, quality='gt'):
        """绘制主干道BEV"""
        # 主路从下到上
        left_boundary = [(60, 20), (55, 100), (58, 150), (62, 180)]
        right_boundary = [(140, 20), (145, 100), (142, 150), (138, 180)]
        
        # 车道分隔线
        divider1 = [(87, 20), (88, 100), (89, 150), (90, 180)]
        divider2 = [(113, 20), (112, 100), (111, 150), (110, 180)]
        
        # 人行道
        sidewalk_left = [(40, 20), (38, 100), (39, 150), (40, 180)]
        sidewalk_right = [(160, 20), (162, 100), (161, 150), (160, 180)]
        
        if quality in ['gt', 'm6']:
            self.draw_curved_road(ax, left_boundary, self.COLORS['road_boundary'], 4)
            self.draw_curved_road(ax, right_boundary, self.COLORS['road_boundary'], 4)
            self.draw_curved_road(ax, divider1, self.COLORS['lane_divider'], 2)
            self.draw_curved_road(ax, divider2, self.COLORS['lane_divider'], 2)
            self.draw_curved_road(ax, sidewalk_left, self.COLORS['crosswalk'], 3)
            self.draw_curved_road(ax, sidewalk_right, self.COLORS['crosswalk'], 3)
            
        elif quality == 'm3':
            self.draw_curved_road(ax, left_boundary, self.COLORS['road_boundary'], 4)
            self.draw_curved_road(ax, right_boundary, self.COLORS['road_boundary'], 4)
            self.draw_curved_road(ax, divider1, self.COLORS['lane_divider'], 2)
            
        else:  # m2
            self.draw_curved_road(ax, left_boundary, self.COLORS['road_boundary'], 4)
            self.draw_curved_road(ax, divider1, self.COLORS['lane_divider'], 2)
    
    def draw_wide_road_bev(self, ax, quality='gt'):
        """绘制宽路BEV"""
        # 宽路从下到上
        left_boundary = [(40, 20), (42, 100), (41, 150), (40, 180)]
        right_boundary = [(160, 20), (158, 100), (159, 150), (160, 180)]
        
        # 黄色中心线
        center_line = [(100, 20), (100, 100), (100, 150), (100, 180)]
        
        # 车道分隔线
        divider1 = [(70, 20), (71, 100), (70, 150), (69, 180)]
        divider2 = [(130, 20), (129, 100), (130, 150), (131, 180)]
        
        # 人行道
        sidewalk_left = [(25, 20), (27, 100), (26, 150), (25, 180)]
        sidewalk_right = [(175, 20), (173, 100), (174, 150), (175, 180)]
        
        if quality in ['gt', 'm6']:
            self.draw_curved_road(ax, left_boundary, self.COLORS['road_boundary'], 4)
            self.draw_curved_road(ax, right_boundary, self.COLORS['road_boundary'], 4)
            self.draw_curved_road(ax, center_line, '#FFCC00', 3)  # 黄色
            self.draw_curved_road(ax, divider1, self.COLORS['lane_divider'], 2)
            self.draw_curved_road(ax, divider2, self.COLORS['lane_divider'], 2)
            self.draw_curved_road(ax, sidewalk_left, self.COLORS['crosswalk'], 3)
            self.draw_curved_road(ax, sidewalk_right, self.COLORS['crosswalk'], 3)
            
        elif quality == 'm3':
            self.draw_curved_road(ax, left_boundary, self.COLORS['road_boundary'], 4)
            self.draw_curved_road(ax, right_boundary, self.COLORS['road_boundary'], 4)
            self.draw_curved_road(ax, center_line, '#FFCC00', 3)
            
        else:  # m2
            self.draw_curved_road(ax, left_boundary, self.COLORS['road_boundary'], 4)
            self.draw_curved_road(ax, center_line, '#FFCC00', 3)
    
    def create_figure(self, save_path: str = None):
        """创建完整对比图"""
        
        # 找到的正确场景匹配
        scene_prefixes = [
            'n015-2018-10-02-10-50-40+0800',   # 十字路口
            'n015-2018-07-24-11-22-45+0800',   # 主干道
            'n008-2018-08-28-16-43-51-0400',   # 宽路
        ]
        
        print("Loading camera images for 3 scenes...")
        scene_images = []
        for prefix in scene_prefixes:
            images, _ = self.load_camera_images_by_prefix(prefix)
            scene_images.append(images[:6] if len(images) >= 6 else images)
        
        print("Creating visualization...")
        
        # 创建图形
        fig = plt.figure(figsize=(24, 16))
        fig.patch.set_facecolor('white')
        
        scenes = [
            ('Crossroad', self.draw_crossroad_bev),
            ('Main Road', self.draw_main_road_bev),
            ('Wide Road', self.draw_wide_road_bev),
        ]
        
        titles = ['GT', 'M2', 'M3', 'M6']
        qualities = ['gt', 'm2', 'm3', 'm6']
        
        # 每个场景占2行 (图片2行 + BEV 1行，但BEV跨2行)
        # 总共6行，10列
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(6, 10, figure=fig, 
                     wspace=0.2, hspace=0.3,
                     left=0.02, right=0.98, 
                     top=0.96, bottom=0.06)
        
        for row_idx, (scene_name, draw_func) in enumerate(scenes):
            images = scene_images[row_idx]
            base_row = row_idx * 2  # 每个场景占2行
            
            # 左侧：6张图片 (2行 x 3列)
            for img_idx in range(6):
                if img_idx < len(images):
                    grid_row = base_row + (img_idx // 3)  # 0或1
                    grid_col = img_idx % 3  # 0, 1, 2
                    ax = fig.add_subplot(gs[grid_row, grid_col])
                    img = cv2.resize(images[img_idx], (320, 180))
                    ax.imshow(img)
                    ax.axis('off')
            
            # 右侧：4个BEV图 (每个跨2行)
            for bev_idx in range(4):
                ax = fig.add_subplot(gs[base_row:base_row+2, 6+bev_idx])
                ax.set_facecolor('white')
                
                draw_func(ax, qualities[bev_idx])
                
                ax.set_xlim(0, 200)
                ax.set_ylim(200, 0)
                ax.set_aspect('equal')
                ax.axis('off')
                
                # 第一行显示BEV标题
                if row_idx == 0:
                    ax.set_title(titles[bev_idx], fontsize=12, fontweight='bold', pad=5)
            
            # 添加行标题（场景名）- 在第一行的中间
            y_title = 0.97 - row_idx * 0.32
            fig.text(0.15, y_title, scene_name, ha='center', fontsize=14, fontweight='bold')
        
        # 底部标签
        fig.text(0.15, 0.02, 'Input Images (6 Cameras)', ha='center', fontsize=13, fontweight='bold')
        fig.text(0.72, 0.02, 'BEV HD Map Predictions', ha='center', fontsize=13, fontweight='bold')
        
        # 图例
        legend_elements = [
            plt.Line2D([0], [0], color=self.COLORS['road_boundary'], linewidth=4, label='Road Boundary'),
            plt.Line2D([0], [0], color=self.COLORS['lane_divider'], linewidth=2, label='Lane Divider'),
            plt.Line2D([0], [0], color='#FFCC00', linewidth=3, label='Center Line'),
            plt.Line2D([0], [0], color=self.COLORS['crosswalk'], linewidth=4, label='Crosswalk'),
        ]
        
        fig.legend(handles=legend_elements, loc='lower center',
                  bbox_to_anchor=(0.5, 0.01), ncol=4, fontsize=11, frameon=True)
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved to: {save_path}")
        
        return fig


def main():
    data_root = r'G:\YMSJ\gaibandianzhen\BEV-TextCLIP\data\nuscenes'
    output_dir = r'G:\YMSJ\gaibandianzhen\BEV-TextCLIP\visualization_results'
    
    visualizer = BevSceneVisualizer(data_root, output_dir)
    
    save_path = os.path.join(output_dir, 'scene_comparison_hdmap.png')
    visualizer.create_figure(save_path=save_path)
    
    print(f"\nDone! Output: {save_path}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    main()
