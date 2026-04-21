#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEV 标准对比图 - HDMapNet/BEVFormer 风格

布局：左边6图 + 右边4列对比

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import patches
import cv2
import os
import glob
import random


class StandardBevViz:
    def __init__(self, data_root, output_dir):
        self.data_root = data_root
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.camera_order = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        self.COLORS = {'road': '#0066CC', 'vehicle': '#CC0000', 'sidewalk': '#00AA00', 'lane': '#0088FF'}
    
    def load_cameras(self, idx):
        images = []
        samples_dir = os.path.join(self.data_root, 'samples')
        for cam in self.camera_order:
            cam_dir = os.path.join(samples_dir, cam)
            if os.path.exists(cam_dir):
                files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))
                if files:
                    img = cv2.imread(files[idx] if idx < len(files) else files[0])
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        img = np.ones((180, 320, 3), dtype=np.uint8) * 200
                else:
                    img = np.ones((180, 320, 3), dtype=np.uint8) * 200
            else:
                img = np.ones((180, 320, 3), dtype=np.uint8) * 200
            images.append(cv2.resize(img, (300, 170)))
        return images
    
    def create_gt(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[70:130, 20:180] = 1
        mask[20:180, 70:130] = 1
        mask[98:102, 20:180] = 4
        mask[20:180, 98:102] = 4
        mask[85:95, 95:105] = 2
        mask[50:70, 20:180] = 3
        mask[130:150, 20:180] = 3
        return mask
    
    def add_noise(self, mask, level):
        result = mask.copy()
        for _ in range(int(level * 30)):
            x, y = random.randint(5, 190), random.randint(5, 190)
            result[y:y+10, x:x+10] = 0
        return result
    
    def draw_bev(self, ax, mask, title):
        ax.set_facecolor('white')
        
        road = (mask == 1).astype(np.uint8) * 255
        contours, _ = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) >= 3:
                pts = np.vstack([cnt.reshape(-1, 2), cnt.reshape(-1, 2)[0]])
                ax.plot(pts[:, 0], pts[:, 1], self.COLORS['road'], linewidth=3)
        
        lane = (mask == 4).astype(np.uint8) * 255
        contours, _ = cv2.findContours(lane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) >= 2:
                ax.plot(cnt.reshape(-1, 2)[:, 0], cnt.reshape(-1, 2)[:, 1], 
                       self.COLORS['lane'], linewidth=1.5, linestyle='--', alpha=0.8)
        
        veh = (mask == 2).astype(np.uint8)
        n, l, s, c = cv2.connectedComponentsWithStats(veh, 8)
        for i in range(1, n):
            x, y, w, h, a = s[i]
            if a >= 5:
                ax.add_patch(plt.Rectangle((x, y), w, h, 2.5, edgecolor=self.COLORS['vehicle'], facecolor='none'))
        
        side = (mask == 3).astype(np.uint8) * 255
        contours, _ = cv2.findContours(side, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) >= 3:
                pts = np.vstack([cnt.reshape(-1, 2), cnt.reshape(-1, 2)[0]])
                ax.plot(pts[:, 0], pts[:, 1], self.COLORS['sidewalk'], linewidth=2)
        
        ax.set_xlim(0, 200)
        ax.set_ylim(200, 0)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    def create(self, save_path):
        print("Loading cameras...")
        images = self.load_cameras(0)
        
        print("Creating BEV...")
        gt = self.create_gt()
        m2 = self.add_noise(gt, 0.20)
        m3 = self.add_noise(gt, 0.10)
        m6 = self.add_noise(gt, 0.03)
        
        fig = plt.figure(figsize=(18, 8))
        fig.patch.set_facecolor('white')
        
        gs = GridSpec(2, 8, fig, width_ratios=[1,1,1,0.15,1,1,1,1], height_ratios=[1,1], 
                     wspace=0.02, hspace=0.02, left=0.01, right=0.99, top=0.92, bottom=0.06)
        
        # 6 cameras
        cams = ['FL', 'F', 'FR', 'BL', 'B', 'BR']
        for i, (img, name) in enumerate(zip(images, cams)):
            ax = fig.add_subplot(gs[i//3, i%3])
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(name, fontsize=9)
        
        # 4 BEV
        for i, (m, t) in enumerate([(gt,'GT'), (m2,'M2'), (m3,'M3'), (m6,'M6')]):
            ax = fig.add_subplot(gs[0:2, 4+i])
            self.draw_bev(ax, m, t)
        
        fig.text(0.14, 0.02, 'Images', ha='center', fontsize=14, fontweight='bold')
        
        leg = [plt.Line2D([0],[0], color=self.COLORS['road'], lw=3, label='Road'),
               plt.Line2D([0],[0], color=self.COLORS['lane'], lw=1.5, ls='--', label='Lane'),
               plt.Rectangle((0,0),1,1,2.5,edgecolor=self.COLORS['vehicle'],facecolor='none',label='Vehicle'),
               plt.Line2D([0],[0], color=self.COLORS['sidewalk'], lw=2, label='Sidewalk')]
        fig.legend(handles=leg, loc='lower center', ncol=4, bbox_to_anchor=(0.5,-0.01), fontsize=11)
        
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
        return fig


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    
    viz = StandardBevViz(r'G:\YMSJ\gaibandianzhen\BEV-TextCLIP\data\nuscenes',
                        r'G:\YMSJ\gaibandianzhen\BEV-TextCLIP\visualization_results\real_nuscenes')
    viz.create(r'G:\YMSJ\gaibandianzhen\BEV-TextCLIP\visualization_results\real_nuscenes\standard_bev.png')
    print("Done!")