#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: TorchScript 模型导出脚本，将 BEV-TextCLIP 模型导出为 TorchScript 格式
日期: 2026年1月23日
"""

import sys
import os
import argparse
import torch
import torch.nn as nn

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
    device: torch.device = None,
):
    """
    创建虚拟输入用于导出

    Args:
        config: 配置对象
        device: 计算设备

    Returns:
        包含虚拟输入的字典
    """
    B, N = 1, 6
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


def export_torchscript(
    model: nn.Module,
    dummy_inputs: dict,
    output_path: str,
    method: str = 'trace',
):
    """
    导出模型为 TorchScript 格式

    Args:
        model: 模型实例
        dummy_inputs: 虚拟输入
        output_path: 输出路径
        method: 导出方法 ('trace' 或 'script')
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\nExporting model to TorchScript...")
    print(f"  Method: {method}")
    print(f"  Output: {output_path}")

    model.eval()

    if method == 'trace':
        traced_model = torch.jit.trace(model, example_inputs=tuple(dummy_inputs.values()))
        traced_model.save(output_path)
        print("TorchScript (traced) export completed!")
    else:
        scripted_model = torch.jit.script(model)
        scripted_model.save(output_path)
        print("TorchScript (scripted) export completed!")

    model_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Size: {model_size:.2f} MB")

    return output_path


def verify_torchscript(
    model_path: str,
    dummy_inputs: dict,
):
    """
    验证 TorchScript 模型

    Args:
        model_path: TorchScript 模型路径
        dummy_inputs: 虚拟输入
    """
    print(f"\nVerifying TorchScript model...")

    loaded_model = torch.jit.load(model_path)
    loaded_model.eval()

    with torch.no_grad():
        output = loaded_model(**dummy_inputs)

    print(f"  Outputs shape: {[v.shape for v in output.values()]}")
    print("TorchScript model verification passed!")


def compare_models(
    original_model: nn.Module,
    traced_model: nn.Module,
    dummy_inputs: dict,
):
    """
    比较原始模型和 TorchScript 模型的输出

    Args:
        original_model: 原始模型
        traced_model: TorchScript 模型
        dummy_inputs: 虚拟输入
    """
    print(f"\nComparing original and TorchScript models...")

    original_model.eval()
    traced_model.eval()

    with torch.no_grad():
        original_output = original_model(**dummy_inputs)
        traced_output = traced_model(**dummy_inputs)

    for key in original_output.keys():
        orig = original_output[key]
        traced = traced_output[key]
        max_diff = (orig - traced).abs().max().item()
        print(f"  {key}: max_diff = {max_diff:.6f}")

    print("Model comparison completed!")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='Export BEV-TextCLIP to TorchScript')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--config', type=str, default='dummy', help='Dataset name')
    parser.add_argument('--output', type=str, default='exports/bev_textclip_traced.pt', help='Output path')
    parser.add_argument('--method', type=str, default='trace', choices=['trace', 'script'], help='Export method')
    parser.add_argument('--verify', action='store_true', help='Verify exported model')
    parser.add_argument('--compare', action='store_true', help='Compare with original model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\n" + "=" * 60)
    print("BEV-TextCLIP TorchScript Export")
    print("=" * 60)

    config = get_config(args.config)

    print("\n[1] Loading model...")
    model = load_model(args.checkpoint, config, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params: {total_params:,}")

    print("\n[2] Creating dummy inputs...")
    dummy_inputs = create_dummy_inputs(config, device)
    print(f"   Images: {dummy_inputs['images'].shape}")
    print(f"   Point cloud: {dummy_inputs['point_cloud'].shape}")

    if args.compare:
        print("\n[3] Exporting TorchScript...")
        export_torchscript(model, dummy_inputs, args.output, method=args.method)

        print("\n[4] Comparing models...")
        traced_model = torch.jit.load(args.output)
        compare_models(model, traced_model, dummy_inputs)
    else:
        print("\n[3] Exporting to TorchScript...")
        export_torchscript(model, dummy_inputs, args.output, method=args.method)

        if args.verify:
            print("\n[4] Verifying TorchScript model...")
            verify_torchscript(args.output, dummy_inputs)

    print("\n" + "=" * 60)
    print("Export completed successfully!")
    print(f"Output: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
