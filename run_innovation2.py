#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创新点2: 动态文本编码与语义引导测试
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.text_encoder import TextEncoder, CategoryEmbedder

def main():
    print("=" * 60)
    print("创新点2: 动态文本编码与语义引导")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    class_names = [
        "vehicle", "pedestrian", "motorcycle", "bicycle",
        "traffic cone", "barrier", "driveable surface",
        "sidewalk", "other flat", "vegetation", "manmade"
    ]
    
    # 1. 创建本地随机CLIP编码器
    print("\n[1/2] 测试本地随机CLIP文本编码器...")
    text_encoder = TextEncoder(
        encoder_type="local_clip",
        model_path="./models/clip_random",
        output_dim=256,
        freeze=False,
        device=device,
    )
    
    texts = ["a car on the road", "a pedestrian crossing"]
    tokens = text_encoder.tokenize(texts, device=device)
    features = text_encoder(tokens)
    print(f"  Token shape: {tokens.shape}")
    print(f"  文本特征 shape: {features['text_features'].shape}")
    
    # 2. 生成类别嵌入
    print("\n[2/2] 测试类别嵌入生成...")
    category_embedder = CategoryEmbedder(
        class_names=class_names,
        text_encoder=text_encoder,
        template="a {} in a driving scenario",
        device=device,
    )
    
    class_embeddings = category_embedder()
    print(f"  类别嵌入 shape: {class_embeddings.shape}")
    
    print("\n" + "=" * 60)
    print("[OK] 创新点2测试通过!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
