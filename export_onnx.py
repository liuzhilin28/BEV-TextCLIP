#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: ONNX 模型导出脚本，将 BEV-TextCLIP 模型导出为 ONNX 格式
日期: 2026年1月23日
"""

import sys
import os
import argparse
import torch
import onnx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.configs.bev_textclip_config import get_config
from src.models.bev_textclip import create_bev_textclip_model


def load_model(
    checkpoint_path: str,
    config,
    device: torch.device,
):
    """
    加载模型检查点

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
    else:
        print("Warning: No checkpoint provided, using random weights")

    return model


def create_dummy_inputs(
    config,
    batch_size: int = 1,
    device: torch.device = None,
):
    """
    创建虚拟输入用于导出

    Args:
        config: 配置对象
        batch_size: 批次大小
        device: 计算设备

    Returns:
        包含虚拟输入的字典
    """
    B = batch_size
    N = 6
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

    return (images, intrinsics, extrinsics, point_cloud, point_cloud_lengths)


def export_onnx(
    model: torch.nn.Module,
    dummy_inputs: tuple,
    output_path: str,
    opset_version: int = 11,
    dynamic_axes: dict = None,
):
    """
    导出模型为 ONNX 格式

    Args:
        model: 模型实例
        dummy_inputs: 虚拟输入
        output_path: 输出路径
        opset_version: ONNX opset 版本
        dynamic_axes: 动态轴定义
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\nExporting model to ONNX...")
    print(f"  Output: {output_path}")
    print(f"  Opset: {opset_version}")

    torch.onnx.export(
        model,
        dummy_inputs,
        output_path,
        input_names=[
            'images',
            'intrinsics',
            'extrinsics',
            'point_cloud',
            'point_cloud_lengths',
        ],
        output_names=[
            'segmentation_logits',
            'bev_features',
        ],
        dynamic_axes=dynamic_axes or {
            'images': {0: 'batch_size', 2: 'height', 3: 'width'},
            'intrinsics': {0: 'batch_size'},
            'extrinsics': {0: 'batch_size'},
            'point_cloud': {0: 'num_points'},
            'segmentation_logits': {0: 'batch_size'},
            'bev_features': {0: 'batch_size'},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        keep_initializers_as_inputs=True,
    )

    print(f"ONNX export completed!")

    onnx_model = onnx.load(output_path)
    print(f"\nONNX Model Info:")
    print(f"  Inputs: {len(onnx_model.graph.input)}")
    print(f"  Outputs: {len(onnx_model.graph.output)}")
    print(f"  Nodes: {len(onnx_model.graph.node)}")

    model_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Size: {model_size:.2f} MB")

    return output_path


def verify_onnx_model(onnx_path: str, dummy_inputs: tuple):
    """
    验证 ONNX 模型

    Args:
        onnx_path: ONNX 模型路径
        dummy_inputs: 虚拟输入
    """
    print(f"\nVerifying ONNX model...")

    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path)
    input_names = [inp.name for inp in session.get_inputs()]

    inputs = {
        'images': dummy_inputs[0].numpy(),
        'intrinsics': dummy_inputs[1].numpy(),
        'extrinsics': dummy_inputs[2].numpy(),
        'point_cloud': dummy_inputs[3].numpy(),
        'point_cloud_lengths': dummy_inputs[4].numpy(),
    }

    outputs = session.run(None, inputs)
    print(f"  Outputs shape: {[out.shape for out in outputs]}")

    print("ONNX model verification passed!")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='Export BEV-TextCLIP to ONNX')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--config', type=str, default='dummy', help='Dataset name')
    parser.add_argument('--output', type=str, default='exports/bev_textclip.onnx', help='Output path')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--opset_version', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--verify', action='store_true', help='Verify exported model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\n" + "=" * 60)
    print("BEV-TextCLIP ONNX Export")
    print("=" * 60)

    config = get_config(args.config)

    print("\n[1] Loading model...")
    model = load_model(args.checkpoint, config, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params: {total_params:,}")

    print("\n[2] Creating dummy inputs...")
    dummy_inputs = create_dummy_inputs(config, args.batch_size, device)
    print(f"   Images: {dummy_inputs[0].shape}")
    print(f"   Intrinsics: {dummy_inputs[1].shape}")
    print(f"   Extrinsics: {dummy_inputs[2].shape}")
    print(f"   Point cloud: {dummy_inputs[3].shape}")

    print("\n[3] Exporting to ONNX...")
    export_onnx(
        model,
        dummy_inputs,
        args.output,
        opset_version=args.opset_version,
    )

    if args.verify:
        print("\n[4] Verifying ONNX model...")
        try:
            verify_onnx_model(args.output, dummy_inputs)
        except ImportError:
            print("   Warning: onnxruntime not installed, skipping verification")

    print("\n" + "=" * 60)
    print("Export completed successfully!")
    print(f"Output: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
