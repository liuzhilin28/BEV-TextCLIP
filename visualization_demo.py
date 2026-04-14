#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: BEV-TextCLIP 可视化演示脚本，演示可视化系统的各项功能
日期: 2026年1月25日
"""

import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.configs.bev_textclip_config import get_config
from src.models.bev_textclip import BEVTextCLIP


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='BEV-TextCLIP 可视化演示')

    parser.add_argument('--config', type=str, default='nuscenes',
                        choices=['nuscenes', 'scannet', 'kitti'],
                        help='数据集配置')

    parser.add_argument('--generate_report', action='store_true',
                        help='生成完整可视化报告')

    parser.add_argument('--output_dir', type=str, default='visualization_results',
                        help='输出目录')

    parser.add_argument('--save_images', action='store_true', default=True,
                        help='保存图像')

    return parser.parse_args()


def create_demo_data(config):
    """创建演示数据"""
    batch_size = 1

    images = torch.randn(batch_size, 6, 3, 224, 224)
    intrinsics = torch.eye(3).unsqueeze(0).expand(batch_size, 6, -1, -1)
    extrinsics = torch.eye(4).unsqueeze(0).expand(batch_size, 6, -1, -1)
    point_cloud = torch.randn(batch_size, 10000, 4)

    predictions = np.random.randint(0, config.num_classes, (200, 200))
    probabilities = np.random.rand(config.num_classes, 200, 200)
    probabilities = probabilities / probabilities.sum(axis=0)

    bev_features = np.random.randn(256, 200, 200)
    text_embeddings = np.random.randn(len(config.class_names), 256)
    attention_weights = np.random.rand(len(config.class_names), 200, 200)

    ground_truth = np.random.randint(0, config.num_classes, (200, 200))

    return {
        'images': images,
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'point_cloud': point_cloud,
        'predictions': predictions,
        'probabilities': probabilities,
        'bev_features': bev_features,
        'text_embeddings': text_embeddings,
        'attention_weights': attention_weights,
        'ground_truth': ground_truth
    }


def visualize_with_model(args):
    """使用模型进行推理并可视化"""
    print("=" * 60)
    print("BEV-TextCLIP 可视化演示")
    print("=" * 60)

    config = get_config(args.config)
    class_names = config.class_names
    print(f"数据集: {args.config}")
    print(f"类别数量: {len(class_names)}")
    print(f"类别列表: {class_names}")
    print()

    from src.visualization.visualizer import SegmentationVisualizer

    visualizer = SegmentationVisualizer(class_names, output_dir=args.output_dir)

    demo_data = create_demo_data(config)

    model = BEVTextCLIP(
        num_classes=config.num_classes,
        class_names=config.class_names,
        image_encoder_type="resnet50",
        point_encoder_type="pointpillar",
        pretrained=False,
    )
    model.eval()

    print("1. 生成分割结果可视化...")
    fig = visualizer.visualize_segmentation(
        demo_data['predictions'],
        save_path=os.path.join(args.output_dir, "demo_segmentation.png"),
        title=f"BEV Segmentation Result ({args.config})"
    )
    plt.close(fig)
    print("   ✓ 分割结果可视化完成")

    print("\n2. 生成类别概率图...")
    for i in range(min(3, len(class_names))):
        fig = visualizer.visualize_probability_map(
            demo_data['probabilities'], i,
            save_path=os.path.join(args.output_dir, f"demo_probability_{class_names[i]}.png")
        )
        plt.close(fig)
    print("   ✓ 概率图可视化完成")

    print("\n3. 生成BEV特征可视化...")
    fig = visualizer.visualize_bev_features(
        demo_data['bev_features'],
        save_path=os.path.join(args.output_dir, "demo_bev_features.png")
    )
    plt.close(fig)
    print("   ✓ BEV特征可视化完成")

    print("\n4. 生成注意力权重可视化...")
    fig = visualizer.visualize_attention_weights(
        demo_data['attention_weights'],
        save_path=os.path.join(args.output_dir, "demo_attention_weights.png")
    )
    plt.close(fig)
    print("   ✓ 注意力权重可视化完成")

    print("\n5. 生成文本嵌入可视化...")
    fig = visualizer.visualize_text_embeddings(
        demo_data['text_embeddings'],
        save_path=os.path.join(args.output_dir, "demo_text_embeddings.png")
    )
    plt.close(fig)
    print("   ✓ 文本嵌入可视化完成")

    print("\n6. 生成预测与真值对比图...")
    fig = visualizer.visualize_comparison(
        demo_data['predictions'],
        demo_data['ground_truth'],
        save_path=os.path.join(args.output_dir, "demo_comparison.png")
    )
    plt.close(fig)
    print("   ✓ 对比可视化完成")

    print("\n7. 生成多样本对比可视化...")
    predictions_list = [
        np.random.randint(0, config.num_classes, (200, 200)),
        np.random.randint(0, config.num_classes, (200, 200)),
        np.random.randint(0, config.num_classes, (200, 200))
    ]
    sample_names = ["Sample 1", "Sample 2", "Sample 3"]
    fig = visualizer.visualize_multi_sample(
        predictions_list, sample_names,
        save_path=os.path.join(args.output_dir, "demo_multi_sample.png")
    )
    plt.close(fig)
    print("   ✓ 多样本对比可视化完成")

    if args.generate_report:
        print("\n8. 生成完整可视化报告...")
        saved_files = visualizer.create_visualization_report(
            predictions=demo_data['predictions'],
            ground_truth=demo_data['ground_truth'],
            probabilities=demo_data['probabilities'],
            bev_features=demo_data['bev_features'],
            text_embeddings=demo_data['text_embeddings'],
            attention_weights=demo_data['attention_weights'],
            save_prefix="full_report"
        )
        print("   ✓ 完整报告生成完成")

    print("\n" + "=" * 60)
    print("可视化结果已保存至:", args.output_dir)
    print("=" * 60)

    print("\n生成的文件:")
    for filename in os.listdir(args.output_dir):
        if filename.endswith('.png'):
            filepath = os.path.join(args.output_dir, filename)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  - {filename} ({size_kb:.1f} KB)")


def demo_api_usage():
    """演示API使用方式"""
    print("\n" + "=" * 60)
    print("可视化 API 使用示例")
    print("=" * 60)

    from src.visualization.visualizer import SegmentationVisualizer

    class_names = ["car", "truck", "bus", "pedestrian", "cyclist"]

    visualizer = SegmentationVisualizer(class_names)

    predictions = np.random.randint(0, 5, (200, 200))
    probabilities = np.random.rand(5, 200, 200)
    probabilities = probabilities / probabilities.sum(axis=0)

    print("\n示例 1: 分割结果可视化")
    print("""
    from src.visualization.visualizer import SegmentationVisualizer
    import numpy as np

    visualizer = SegmentationVisualizer(class_names=['car', 'truck', 'pedestrian'])
    predictions = np.random.randint(0, 3, (200, 200))

    fig = visualizer.visualize_segmentation(
        predictions,
        save_path='segmentation_result.png',
        title='My Segmentation Result'
    )
    """)

    print("\n示例 2: 概率图可视化")
    print("""
    probabilities = np.random.rand(3, 200, 200)

    fig = visualizer.visualize_probability_map(
        probabilities,
        class_idx=0,
        save_path='probability_car.png',
        title='Car Probability Map'
    )
    """)

    print("\n示例 3: 完整报告生成")
    print("""
    saved_files = visualizer.create_visualization_report(
        predictions=predictions,
        ground_truth=ground_truth,
        probabilities=probabilities,
        bev_features=bev_features,
        text_embeddings=text_embeddings,
        attention_weights=attention_weights,
        save_prefix='my_report'
    )

    for name, path in saved_files.items():
        print(f'{name}: {path}')
    """)


def main():
    """主函数"""
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    visualize_with_model(args)

    demo_api_usage()

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
