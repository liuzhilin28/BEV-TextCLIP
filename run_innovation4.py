#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创新点4: 多层级对比学习损失函数测试
"""

import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.losses import MultiModalContrastiveLoss

def main():
    print("=" * 60)
    print("创新点4: 多层级对比学习损失函数")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    batch_size = 4
    bev_channels = 256
    num_classes = 11
    
    # 创建损失函数
    contrastive_loss = MultiModalContrastiveLoss(
        temperature=0.07,
        global_weight=1.0,
        local_weight=0.5,
        cross_modal_weight=0.5,
    ).to(device)
    
    # 准备测试数据
    image_global = F.normalize(torch.randn(batch_size, bev_channels), dim=-1).to(device)
    point_global = F.normalize(torch.randn(batch_size, bev_channels), dim=-1).to(device)
    # text_global 用于对比损失，每个batch使用相同的类别嵌入
    text_global = F.normalize(torch.randn(num_classes, bev_channels), dim=-1).to(device)
    
    image_features = torch.randn(batch_size, bev_channels, 50, 50).to(device)
    point_features = torch.randn(batch_size, bev_channels, 50, 50).to(device)
    bev_features = torch.randn(batch_size, bev_channels, 50, 50).to(device)
    
    # 1. 测试全局对比损失
    print("\n[1/3] 测试全局对比损失...")
    loss_global = contrastive_loss.global_contrast(image_global, point_global, text_global)
    print(f"  全局对比损失: {loss_global.item():.4f}")
    
    # 2. 测试局部对比损失
    print("\n[2/3] 测试局部对比损失...")
    loss_local = contrastive_loss.local_contrast(image_features, point_features, bev_features)
    print(f"  局部对比损失: {loss_local.item():.4f}")
    
    # 3. 测试跨模态对比损失
    print("\n[3/3] 测试跨模态对比损失...")
    loss_cross = contrastive_loss.cross_modal_contrast(image_global, point_global, text_global)
    print(f"  跨模态对比损失: {loss_cross.item():.4f}")
    
    # 4. 测试完整损失
    print("\n[4/4] 测试完整多层级对比损失...")
    loss_dict = contrastive_loss(
        image_global=image_global,
        point_global=point_global,
        text_global=text_global,
        image_features=image_features,
        point_features=point_features,
        bev_features=bev_features,
    )
    
    print("  损失详情:")
    for k, v in loss_dict.items():
        print(f"    {k}: {v.item():.4f}")
    
    print("\n" + "=" * 60)
    print("[OK] 创新点4测试通过!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
