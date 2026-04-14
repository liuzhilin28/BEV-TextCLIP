#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创新点3: 跨模态注意力机制测试
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.cross_attention import BEVTextCrossAttention, BevTextInteraction

def main():
    print("=" * 60)
    print("创新点3: 跨模态注意力机制")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    batch_size = 2
    bev_channels = 256
    text_channels = 256
    num_classes = 11
    
    # 1. 测试BEV-文本跨模态注意力
    print("\n[1/2] 测试BEV-文本跨模态注意力...")
    cross_attn = BEVTextCrossAttention(
        bev_channels=bev_channels,
        text_channels=text_channels,
        num_heads=8,
        dropout=0.1,
        use_bidirectional=True,
    ).to(device)
    
    bev_features = torch.randn(batch_size, 200 * 200, bev_channels).to(device)
    text_embeddings = torch.randn(num_classes, text_channels).to(device)
    
    enhanced_bev, attn_weights = cross_attn(bev_features, text_embeddings)
    print(f"  增强BEV shape: {enhanced_bev.shape}")
    print(f"  注意力权重 shape: {attn_weights.shape}")
    
    # 2. 测试BEV-文本交互模块
    print("\n[2/2] 测试BEV-文本交互模块...")
    interaction = BevTextInteraction(
        bev_channels=bev_channels,
        text_channels=text_channels,
        num_heads=8,
        dropout=0.1,
        use_bidirectional=True,
    ).to(device)
    
    bev_2d = torch.randn(batch_size, bev_channels, 200, 200).to(device)
    
    interaction_output = interaction(bev_2d, text_embeddings)
    print(f"  增强BEV shape: {interaction_output['enhanced_bev'].shape}")
    print(f"  注意力权重 shape: {interaction_output['attention_weights'].shape}")
    
    print("\n" + "=" * 60)
    print("[OK] 创新点3测试通过!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
