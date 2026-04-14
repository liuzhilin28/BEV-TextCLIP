#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: 推理性能基准测试，测试 BEV-TextCLIP 模型的推理性能
日期: 2026年1月23日
"""

import sys
import os
import time
import argparse
import statistics
from typing import Dict, List, Any

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.configs.bev_textclip_config import get_config
from src.models.bev_textclip import create_bev_textclip_model


class BenchmarkResult:
    """
    基准测试结果
    """

    def __init__(self):
        self.inference_times: List[float] = []
        self.throughputs: List[float] = []
        self.memory_usages: List[float] = []
        self.warmup_times: List[float] = []

    def add_inference_time(self, time_ms: float):
        self.inference_times.append(time_ms)

    def add_warmup_time(self, time_ms: float):
        self.warmup_times.append(time_ms)

    def add_throughput(self, samples_per_sec: float):
        self.throughputs.append(samples_per_sec)

    def add_memory_usage(self, memory_mb: float):
        self.memory_usages.append(memory_mb)

    def summary(self) -> Dict[str, Any]:
        """
        获取结果摘要

        Returns:
            包含统计信息的字典
        """
        if not self.inference_times:
            return {}

        return {
            'inference_time': {
                'mean_ms': statistics.mean(self.inference_times),
                'std_ms': statistics.stdev(self.inference_times) if len(self.inference_times) > 1 else 0,
                'min_ms': min(self.inference_times),
                'max_ms': max(self.inference_times),
                'median_ms': statistics.median(self.inference_times),
            },
            'warmup_time': {
                'mean_ms': statistics.mean(self.warmup_times) if self.warmup_times else 0,
            },
            'throughput': {
                'mean_sps': statistics.mean(self.throughputs) if self.throughputs else 0,
            },
            'memory': {
                'mean_mb': statistics.mean(self.memory_usages) if self.memory_usages else 0,
            },
            'samples': len(self.inference_times),
        }


def create_dummy_data(
    config,
    batch_size: int = 1,
    device: torch.device = None,
) -> Dict[str, torch.Tensor]:
    """
    创建虚拟输入数据

    Args:
        config: 配置对象
        batch_size: 批次大小
        device: 计算设备

    Returns:
        包含虚拟输入的字典
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


def get_memory_usage(device: torch.device) -> float:
    """
    获取 GPU 内存使用

    Args:
        device: 计算设备

    Returns:
        内存使用量 (MB)
    """
    if device.type == 'cuda':
        return torch.cuda.memory_allocated(device) / (1024 * 1024)
    return 0.0


def benchmark_model(
    model: torch.nn.Module,
    dummy_inputs: Dict[str, torch.Tensor],
    iterations: int = 100,
    warmup_iterations: int = 10,
    device: torch.device = None,
) -> BenchmarkResult:
    """
    基准测试模型

    Args:
        model: 模型实例
        dummy_inputs: 虚拟输入
        iterations: 测试迭代次数
        warmup_iterations: 预热迭代次数
        device: 计算设备

    Returns:
        基准测试结果
    """
    model.eval()
    result = BenchmarkResult()

    print(f"\nRunning benchmark...")
    print(f"  Warmup: {warmup_iterations} iterations")
    print(f"  Benchmark: {iterations} iterations")

    with torch.no_grad():
        for i in range(warmup_iterations):
            start_time = time.time()
            _ = model(**dummy_inputs)
            warmup_time = (time.time() - start_time) * 1000
            result.add_warmup_time(warmup_time)

        print(f"\n  Warmup completed. Avg warmup time: {statistics.mean(result.warmup_times):.2f} ms")

        for i in range(iterations):
            start_time = time.time()
            _ = model(**dummy_inputs)
            inference_time = (time.time() - start_time) * 1000
            result.add_inference_time(inference_time)

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{iterations}")

    throughput = iterations / (sum(result.inference_times) / 1000)
    result.add_throughput(throughput)
    result.add_memory_usage(get_memory_usage(device))

    return result


def print_results(result: BenchmarkResult):
    """
    打印结果

    Args:
        result: 基准测试结果
    """
    summary = result.summary()

    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)

    print(f"\nSamples: {summary.get('samples', 0)}")

    print("\n--- Inference Time ---")
    inf_time = summary.get('inference_time', {})
    print(f"  Mean:   {inf_time.get('mean_ms', 0):.2f} ms")
    print(f"  Std:    {inf_time.get('std_ms', 0):.2f} ms")
    print(f"  Min:    {inf_time.get('min_ms', 0):.2f} ms")
    print(f"  Max:    {inf_time.get('max_ms', 0):.2f} ms")
    print(f"  Median: {inf_time.get('median_ms', 0):.2f} ms")

    print("\n--- Warmup Time ---")
    warmup = summary.get('warmup_time', {})
    print(f"  Mean: {warmup.get('mean_ms', 0):.2f} ms")

    print("\n--- Throughput ---")
    throughput = summary.get('throughput', {})
    print(f"  Mean: {throughput.get('mean_sps', 0):.2f} samples/sec")

    print("\n--- Memory Usage ---")
    memory = summary.get('memory', {})
    print(f"  Mean: {memory.get('mean_mb', 0):.2f} MB")

    print("\n" + "=" * 60)

    return summary


def export_results(summary: Dict[str, Any], output_path: str):
    """
    导出结果到 JSON

    Args:
        summary: 结果摘要
        output_path: 输出路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    import json

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults exported to {output_path}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='BEV-TextCLIP Benchmark')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--config', type=str, default='dummy', help='Dataset name')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--output', type=str, default='benchmark_results.json', help='Output file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\n" + "=" * 60)
    print("BEV-TextCLIP Inference Benchmark")
    print("=" * 60)

    config = get_config(args.config)

    print("\n[1] Loading model...")
    model = create_bev_textclip_model(config)
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params: {total_params:,}")

    print("\n[2] Creating dummy inputs...")
    dummy_inputs = create_dummy_data(config, args.batch_size, device)
    print(f"   Batch size: {args.batch_size}")
    print(f"   Images: {dummy_inputs['images'].shape}")
    print(f"   Point cloud: {dummy_inputs['point_cloud'].shape}")

    print("\n[3] Running benchmark...")
    result = benchmark_model(
        model,
        dummy_inputs,
        iterations=args.iterations,
        warmup_iterations=args.warmup,
        device=device,
    )

    summary = print_results(result)
    export_results(summary, args.output)

    print("\n" + "=" * 60)
    print("Benchmark completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
