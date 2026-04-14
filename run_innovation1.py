#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创新点1: BEV空间表征学习与多模态融合测试
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.image_encoder import ImageEncoder
from src.models.point_encoder import PointEncoder
from src.models.fusion_module import BevFusion

def main():
    print("=" * 60)
    print("创新点1: BEV空间表征学习与多模态融合")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    batch_size = 2
    bev_channels = 256
    num_classes = 16
    
    # 1. 测试图像编码器
    print("\n[1/3] 测试图像编码器...")
    image_encoder = ImageEncoder(
        image_encoder_type="resnet50",
        out_channels=bev_channels,
        freeze=False,
    ).to(device)
    
    images = torch.randn(batch_size, 6, 3, 224, 224).to(device)
    intrinsics = torch.eye(3).unsqueeze(0).expand(batch_size, 6, -1, -1).to(device)
    extrinsics = torch.eye(4).unsqueeze(0).expand(batch_size, 6, -1, -1).to(device)
    
    image_bev = image_encoder(images, intrinsics, extrinsics)
    print(f"  图像BEV shape: {image_bev.shape}")
    
    # 2. 测试点云编码器
    print("\n[2/3] 测试点云编码器...")
    point_encoder = PointEncoder(
        in_channels=4,
        out_channels=bev_channels,
        encoder_type="pointpillar",
    ).to(device)
    
    point_cloud = torch.randn(batch_size, 1000, 4).to(device)
    point_bev = point_encoder(point_cloud)
    print(f"  点云BEV shape: {point_bev.shape}")
    
    # 3. 测试融合模块
    print("\n[3/3] 测试多模态融合...")
    fusion = BevFusion(
        in_channels=bev_channels,
        out_channels=bev_channels,
        fusion_type="gated_attention",
    ).to(device)
    
    fused_bev = fusion(image_bev, point_bev)
    print(f"  融合BEV shape: {fused_bev.shape}")
    
    print("\n" + "=" * 60)
    print("[OK] 创新点1测试通过!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
