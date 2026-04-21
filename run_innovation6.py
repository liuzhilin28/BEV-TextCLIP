#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创新点6: 分割结果可视化系统测试
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.configs.bev_textclip_config import get_config
from src.models.bev_textclip import BEVTextCLIP
from src.visualization.visualizer import SegmentationVisualizer

def main():
    print("=" * 60)
    print("创新点6: 分割结果可视化系统")
    print("=" * 60)
    
    config = get_config("nuscenes")
    class_names = config.class_names
    print(f"类别数量: {len(class_names)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 创建可视化工具
    print("\n[1/1] 创建可视化报告...")
    visualizer = SegmentationVisualizer(class_names)
    
    # 生成虚拟分割结果
    batch_size = 1
    predictions = np.random.randint(0, len(class_names), (200, 200))
    probabilities = np.random.randn(len(class_names), 200, 200)
    probabilities = probabilities / probabilities.sum(axis=0)
    text_embeddings = np.random.randn(len(class_names), 256)
    
    # 生成分割结果可视化
    os.makedirs("visualization_results", exist_ok=True)
    
    print("  生成分割结果图...")
    fig = visualizer.visualize_segmentation(
        predictions,
        save_path="visualization_results/demo_segmentation.png",
        title="BEV Segmentation Result"
    )
    import matplotlib.pyplot as plt
    plt.close(fig)
    
    print("  生成概率图...")
    for i in range(min(3, len(class_names))):
        fig = visualizer.visualize_probability_map(
            probabilities, i,
            save_path=f"visualization_results/demo_probability_{class_names[i]}.png"
        )
        plt.close(fig)
    
    print("  生成文本嵌入可视化...")
    fig = visualizer.visualize_text_embeddings(
        text_embeddings,
        save_path="visualization_results/demo_text_embeddings.png"
    )
    plt.close(fig)
    
    print("  生成对比图...")
    ground_truth = np.random.randint(0, len(class_names), predictions.shape)
    fig = visualizer.visualize_comparison(
        predictions, ground_truth,
        save_path="visualization_results/demo_comparison.png"
    )
    plt.close(fig)
    
    print("\n可视化结果已保存至: visualization_results/")
    
    print("\n" + "=" * 60)
    print("[OK] 创新点6测试通过!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    success = main()
    sys.exit(0 if success else 1)
