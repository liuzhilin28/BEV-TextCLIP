#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: BEV-TextCLIP 数据集准备脚本
日期: 2026年1月23日
"""

import os
import argparse


def prepare_nuscenes(data_root: str):
    """准备 nuScenes 数据集"""
    print("\n" + "="*60)
    print("nuScenes 数据集准备")
    print("="*60)
    
    required_dirs = [
        'samples',           # 图像和点云
        'maps',              # 地图
        'v1.0-trainval',    # 元数据
    ]
    
    print(f"\n数据根目录: {data_root}")
    print("\n检查必需目录:")
    
    for dir_name in required_dirs:
        dir_path = os.path.join(data_root, dir_name)
        exists = os.path.exists(dir_path)
        status = "[OK]" if exists else "[NO]"
        print(f"  {status} {dir_name}: {'存在' if exists else '缺失'}")
    
    if not all(os.path.exists(os.path.join(data_root, d)) for d in required_dirs):
        print("\n[!!] 数据集目录不完整!")
        print("\n下载数据集:")
        print("  1. 访问 https://www.nuscenes.org/download")
        print("  2. 下载 nuScenes v1.0-trainval")
        print("  3. 解压到 data_root 目录")
        return False
    
    print("\n[OK] nuScenes 数据集准备完成")
    return True


def prepare_scannet(data_root: str):
    """准备 ScanNet 数据集"""
    print("\n" + "="*60)
    print("ScanNet 数据集准备")
    print("="*60)
    
    required_dirs = [
        'scans',            # 场景数据
    ]
    
    print(f"\n数据根目录: {data_root}")
    print("\n检查必需目录:")
    
    for dir_name in required_dirs:
        dir_path = os.path.join(data_root, dir_name)
        exists = os.path.exists(dir_path)
        status = "[OK]" if exists else "[NO]"
        print(f"  {status} {dir_name}: {'存在' if exists else '缺失'}")
    
    split_files = ['scannet_train.txt', 'scannet_val.txt', 'scannet_test.txt']
    print("\n检查划分文件:")
    for split_file in split_files:
        file_path = os.path.join(data_root, split_file)
        exists = os.path.exists(file_path)
        status = "[OK]" if exists else "[NO]"
        print(f"  {status} {split_file}")
    
    if not os.path.exists(os.path.join(data_root, 'scans')):
        print("\n[!!] 数据集目录不完整!")
        print("\n下载数据集:")
        print("  1. 访问 https://scannet.cs.stanford.edu")
        print("  2. 下载 ScanNet 数据")
        print("  3. 解压到 data_root 目录")
        return False
    
    print("\n[OK] ScanNet 数据集准备完成")
    return True


def prepare_kitti(data_root: str):
    """准备 KITTI 数据集"""
    print("\n" + "="*60)
    print("KITTI 数据集准备")
    print("="*60)
    
    required_dirs = [
        'velodyne_points',  # 点云
        'image_02',         # 左视角图像
        'image_03',         # 右视角图像
        'calib',            # 标定文件
        'label_2',          # 标签文件
    ]
    
    print(f"\n数据根目录: {data_root}")
    print("\n检查必需目录:")
    
    for dir_name in required_dirs:
        dir_path = os.path.join(data_root, dir_name)
        exists = os.path.exists(dir_path)
        status = "[OK]" if exists else "[NO]"
        print(f"  {status} {dir_name}: {'存在' if exists else '缺失'}")
    
    split_files = ['kitti_train.txt', 'kitti_val.txt', 'kitti_test.txt']
    print("\n检查划分文件:")
    for split_file in split_files:
        file_path = os.path.join(data_root, split_file)
        exists = os.path.exists(file_path)
        status = "[OK]" if exists else "[NO]"
        print(f"  {status} {split_file}")
    
    if not all(os.path.exists(os.path.join(data_root, d)) for d in required_dirs[:4]):
        print("\n[!!] 数据集目录不完整!")
        print("\n下载数据集:")
        print("  1. 访问 https://www.cvlibs.net/datasets/kitti/")
        print("  2. 下载 Object Detection Dataset")
        print("  3. 解压到 data_root 目录")
        return False
    
    print("\n[OK] KITTI 数据集准备完成")
    return True


def create_split_file(data_root: str, dataset_name: str, split: str, samples: list):
    """创建划分文件"""
    split_file = os.path.join(data_root, f"{dataset_name}_{split}.txt")
    with open(split_file, 'w') as f:
        for sample in samples:
            f.write(f"{sample}\n")
    print(f"  创建划分文件: {split_file}")


def generate_dummy_data(data_root: str, dataset_name: str):
    """生成虚拟数据用于测试"""
    print(f"\n生成 {dataset_name} 虚拟数据...")
    
    if dataset_name == 'nuscenes':
        create_split_file(data_root, 'nuscenes', 'train', [f'{i:06d}' for i in range(100)])
        create_split_file(data_root, 'nuscenes', 'val', [f'{i:06d}' for i in range(10)])
    elif dataset_name == 'scannet':
        create_split_file(data_root, 'scannet', 'train', [f'scene0000_{i:04d}' for i in range(100)])
        create_split_file(data_root, 'scannet', 'val', [f'scene0000_{i:04d}' for i in range(10)])
    elif dataset_name == 'kitti':
        create_split_file(data_root, 'kitti', 'train', [f'{i:06d}' for i in range(100)])
        create_split_file(data_root, 'kitti', 'val', [f'{i:06d}' for i in range(10)])
    
    print(f"  [OK] 虚拟数据划分文件已生成")


def main():
    parser = argparse.ArgumentParser(description="BEV-TextCLIP 数据集准备工具")
    parser.add_argument('--dataset', type=str, default='nuscenes',
                        choices=['nuscenes', 'scannet', 'kitti', 'all'],
                        help='数据集类型')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='数据根目录')
    parser.add_argument('--dummy', action='store_true',
                        help='生成虚拟数据（用于测试）')
    
    args = parser.parse_args()
    
    print("""
+============================================================+
|           BEV-TextCLIP 数据集准备工具                       |
+============================================================+
    """)
    
    os.makedirs(args.data_root, exist_ok=True)
    
    if args.dataset == 'all':
        for ds in ['nuscenes', 'scannet', 'kitti']:
            if args.dummy:
                generate_dummy_data(args.data_root, ds)
            else:
                if ds == 'nuscenes':
                    prepare_nuscenes(args.data_root)
                elif ds == 'scannet':
                    prepare_scannet(args.data_root)
                elif ds == 'kitti':
                    prepare_kitti(args.data_root)
    else:
        if args.dummy:
            generate_dummy_data(args.data_root, args.dataset)
        elif args.dataset == 'nuscenes':
            prepare_nuscenes(args.data_root)
        elif args.dataset == 'scannet':
            prepare_scannet(args.data_root)
        elif args.dataset == 'kitti':
            prepare_kitti(args.data_root)
    
    print("\n" + "="*60)
    print("使用说明")
    print("="*60)
    print("""
1. 准备真实数据:
   python prepare_data.py --dataset nuscenes --data_root /path/to/nuscenes
   python prepare_data.py --dataset scannet --data_root /path/to/scannet
   python prepare_data.py --dataset kitti --data_root /path/to/kitti

2. 生成虚拟数据（用于测试）:
   python prepare_data.py --dataset nuscenes --dummy
   python prepare_data.py --dataset scannet --dummy
   python prepare_data.py --dataset kitti --dummy

3. 使用配置训练:
   python train.py --config configs/nuscenes.yaml
   python train.py --config configs/scannet.yaml
   python train.py --config configs/kitti.yaml
    """)


if __name__ == "__main__":
    main()
