#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: 推理演示脚本，展示如何使用 BEV-TextCLIP 模型进行推理
日期: 2026年1月23日
"""

import sys
import os
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.configs.bev_textclip_config import BEVTextCLIPConfig, get_config
from src.models.bev_textclip import create_bev_textclip_model


def load_model(
    checkpoint_path: str,
    config: BEVTextCLIPConfig,
    device: torch.device,
):
    """
    加载模型和检查点

    Args:
        checkpoint_path: 检查点路径
        config: 配置对象
        device: 计算设备

    Returns:
        模型实例
    """
    model = create_bev_textclip_model(config)
    model = model.to(device)
    model.eval()

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded successfully")

    return model


def preprocess_dummy_data(
    batch_size: int = 1,
    device: torch.device = None,
):
    """
    创建虚拟输入数据

    Args:
        batch_size: 批次大小
        device: 计算设备

    Returns:
        包含输入数据的字典
    """
    B, N = batch_size, 6
    C, H, W = 3, 370, 1224
    num_points = 10000

    images = torch.randn(B, N, C, H, W, dtype=torch.float32)
    intrinsics = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1).unsqueeze(1).expand(-1, N, -1, -1)
    extrinsics = torch.eye(4, dtype=torch.float32).unsqueeze(0).unsqueeze(1).expand(-1, N, -1, -1)
    point_cloud = torch.randn(num_points, 4, dtype=torch.float32)
    point_cloud_lengths = torch.tensor([num_points], dtype=torch.long)

    if device:
        images = images.to(device)
        intrinsics = intrinsics.to(device)
        extrinsics = extrinsics.to(device)
        point_cloud = point_cloud.to(device)
        point_cloud_lengths = point_cloud_lengths.to(device)

    return {
        'images': images,
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'point_cloud': point_cloud,
        'point_cloud_lengths': point_cloud_lengths,
    }


@torch.no_grad()
def inference(
    model: torch.nn.Module,
    data: dict,
    config: BEVTextCLIPConfig,
):
    """
    执行推理

    Args:
        model: 模型实例
        data: 输入数据字典
        config: 配置对象

    Returns:
        包含推理结果的字典
    """
    output = model(
        images=data['images'],
        intrinsics=data['intrinsics'],
        extrinsics=data['extrinsics'],
        point_cloud=data['point_cloud'],
        point_cloud_lengths=data.get('point_cloud_lengths'),
    )

    predictions = output['segmentation_logits'].argmax(dim=1)
    probabilities = torch.softmax(output['segmentation_logits'], dim=1)

    result = {
        'predictions': predictions,
        'probabilities': probabilities,
        'bev_features': output.get('bev_features'),
        'image_features': output.get('image_features'),
        'text_features': output.get('text_features'),
    }

    return result


def postprocess_predictions(
    predictions: torch.Tensor,
    class_names: list,
    colormap: np.ndarray = None,
) -> np.ndarray:
    """
    后处理预测结果

    Args:
        predictions: 预测结果 [B, H, W]
        class_names: 类别名称列表
        colormap: 颜色映射表

    Returns:
        可视化图像 [H, W, 3]
    """
    if colormap is None:
        colormap = np.random.randint(0, 255, size=(len(class_names), 3))

    pred_np = predictions.cpu().numpy()
    if pred_np.ndim == 3:
        pred_np = pred_np[0]

    vis = np.zeros((*pred_np.shape, 3), dtype=np.uint8)
    for class_id in range(len(class_names)):
        mask = pred_np == class_id
        if mask.any():
            vis[mask] = colormap[class_id]

    return vis


def save_predictions(
    predictions: torch.Tensor,
    save_path: str,
    format: str = 'npy',
):
    """
    保存预测结果

    Args:
        predictions: 预测结果
        save_path: 保存路径
        format: 保存格式 ('npy', 'png', 'pt')
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if format == 'npy':
        np.save(save_path, predictions.cpu().numpy())
    elif format == 'pt':
        torch.save(predictions, save_path)
    else:
        from PIL import Image
        pred_np = predictions.cpu().numpy().astype(np.uint8)
        if pred_np.ndim == 3:
            pred_np = pred_np[0]
        Image.fromarray(pred_np).save(save_path)

    print(f"Predictions saved to {save_path}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='BEV-TextCLIP Inference Demo')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--config', type=str, default='dummy', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='outputs/inference', help='Output directory')
    parser.add_argument('--save_results', action='store_true', help='Save results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\n" + "=" * 60)
    print("BEV-TextCLIP Inference Demo")
    print("=" * 60)

    config = get_config(args.config)

    print("\n[1] Loading model...")
    model = load_model(args.checkpoint, config, device)
    print(f"   Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    print("\n[2] Preparing input data...")
    data = preprocess_dummy_data(args.batch_size, device)
    print(f"   Images: {data['images'].shape}")
    print(f"   Point cloud: {data['point_cloud'].shape}")

    print("\n[3] Running inference...")
    result = inference(model, data, config)

    print("\n[4] Results:")
    print(f"   Predictions shape: {result['predictions'].shape}")
    print(f"   Probabilities shape: {result['probabilities'].shape}")

    print("\n   Class distribution:")
    for i, name in enumerate(config.class_names[:5]):
        count = (result['predictions'] == i).sum().item()
        print(f"      {name}: {count}")

    if args.save_results:
        print(f"\n[5] Saving results to {args.output_dir}")
        save_predictions(result['predictions'], f"{args.output_dir}/predictions.npy")
        save_predictions(result['probabilities'], f"{args.output_dir}/probabilities.pt")

    print("\n" + "=" * 60)
    print("Inference completed successfully!")
    print("=" * 60)

    return result


if __name__ == '__main__':
    main()
