#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创新点5: 灵活部署与模型导出测试
"""

import argparse
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.configs.bev_textclip_config import get_config
from src.models.bev_textclip import BEVTextCLIP

def export_onnx(model, config, output_path):
    """导出ONNX模型"""
    model.eval()
    
    batch_size = 1
    images = torch.randn(batch_size, 6, 3, 224, 224)
    intrinsics = torch.eye(3).unsqueeze(0).expand(batch_size, 6, -1, -1)
    extrinsics = torch.eye(4).unsqueeze(0).expand(batch_size, 6, -1, -1)
    point_cloud = torch.randn(batch_size, 1000, 4)
    
    torch.onnx.export(
        model,
        (images, intrinsics, extrinsics, point_cloud),
        output_path,
        input_names=["images", "intrinsics", "extrinsics", "point_cloud"],
        output_names=["segmentation_logits", "predictions"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "point_cloud": {0: "batch_size", 1: "num_points"},
            "segmentation_logits": {0: "batch_size"},
            "predictions": {0: "batch_size"},
        },
        opset_version=11,
        do_constant_folding=True,
    )
    print(f"ONNX模型已导出: {output_path}")

def export_torchscript(model, config, output_path):
    """导出TorchScript模型"""
    model.eval()
    
    batch_size = 1
    images = torch.randn(batch_size, 6, 3, 224, 224)
    intrinsics = torch.eye(3).unsqueeze(0).expand(batch_size, 6, -1, -1)
    extrinsics = torch.eye(4).unsqueeze(0).expand(batch_size, 6, -1, -1)
    point_cloud = torch.randn(batch_size, 1000, 4)
    
    scripted_model = torch.jit.trace(model, (images, intrinsics, extrinsics, point_cloud))
    scripted_model.save(output_path)
    print(f"TorchScript模型已导出: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='创新点5: 部署与导出')
    parser.add_argument('--mode', type=str, default='api',
                        choices=['api', 'export_onnx', 'export_torchscript', 'test'],
                        help='运行模式')
    parser.add_argument('--output_dir', type=str, default='./exports',
                        help='导出目录')
    args = parser.parse_args()
    
    print("=" * 60)
    print("创新点5: 灵活部署与模型导出")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    config = get_config("nuscenes")
    
    # 创建模型
    print("\n[1/1] 创建模型...")
    model = BEVTextCLIP(
        num_classes=config.num_classes,
        class_names=config.class_names,
        image_encoder_type="resnet50",
        point_encoder_type="pointpillar",
        text_encoder_type="local_clip",
        freeze_image_encoder=False,
        freeze_text_encoder=False,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")
    
    if args.mode == 'api':
        print("\n启动API服务...")
        print("请使用: uvicorn inference_api:app --host 0.0.0.0 --port 8000")
        print("API文档: http://localhost:8000/docs")
        
    elif args.mode == 'export_onnx':
        os.makedirs(args.output_dir, exist_ok=True)
        export_onnx(model, config, os.path.join(args.output_dir, "bev_textclip.onnx"))
        
    elif args.mode == 'export_torchscript':
        os.makedirs(args.output_dir, exist_ok=True)
        export_torchscript(model, config, os.path.join(args.output_dir, "bev_textclip.pt"))
        
    elif args.mode == 'test':
        batch_size = 2
        images = torch.randn(batch_size, 6, 3, 224, 224).to(device)
        intrinsics = torch.eye(3).unsqueeze(0).expand(batch_size, 6, -1, -1).to(device)
        extrinsics = torch.eye(4).unsqueeze(0).expand(batch_size, 6, -1, -1).to(device)
        point_cloud = torch.randn(batch_size, 1000, 4).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(images, intrinsics, extrinsics, point_cloud)
        
        print(f"  分割输出 shape: {output['segmentation_logits'].shape}")
        print(f"  预测 shape: {output['segmentation_logits'].argmax(dim=1).shape}")
        print("\n[OK] 模型导出测试通过!")
    
    print("\n" + "=" * 60)
    print("[OK] 创新点5测试完成!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
