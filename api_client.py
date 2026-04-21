#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: API 客户端，BEV-TextCLIP 推理 API 客户端
日期: 2026年1月23日
"""

import sys
import os
import time
import argparse
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import requests
import numpy as np


@dataclass
class InferenceResult:
    """
    推理结果
    """
    predictions: np.ndarray
    probabilities: np.ndarray
    inference_time_ms: float
    model_info: Dict[str, Any]


class BEVTextCLIPClient:
    """
    BEV-TextCLIP 推理 API 客户端
    """

    def __init__(
        self,
        base_url: str = 'http://localhost:8000',
        timeout: int = 300
    ):
        """
        初始化客户端

        Args:
            base_url: API 服务器地址
            timeout: 请求超时时间 (秒)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

    def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            健康状态信息
        """
        response = self.session.get(
            f'{self.base_url}/health',
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            模型信息
        """
        response = self.session.get(
            f'{self.base_url}/info',
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_demo_request(self) -> Dict[str, Any]:
        """
        获取演示请求数据

        Returns:
            演示请求数据
        """
        response = self.session.get(
            f'{self.base_url}/demo',
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def inference(
        self,
        images: List[List[List[List[int]]]],
        intrinsics: List[List[List[float]]],
        extrinsics: List[List[List[float]]],
        point_cloud: List[List[float]],
    ) -> InferenceResult:
        """
        执行推理

        Args:
            images: 多视角图像
            intrinsics: 相机内参
            extrinsics: 相机外参
            point_cloud: 点云

        Returns:
            推理结果
        """
        request_data = {
            'images': images,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'point_cloud': point_cloud,
        }

        start_time = time.time()

        response = self.session.post(
            f'{self.base_url}/inference',
            json=request_data,
            timeout=self.timeout,
        )
        response.raise_for_status()

        result = response.json()
        inference_time = (time.time() - start_time) * 1000

        return InferenceResult(
            predictions=np.array(result['predictions']),
            probabilities=np.array(result['probabilities']),
            inference_time_ms=inference_time,
            model_info=result['model_info'],
        )

    def batch_inference(
        self,
        batch: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        批量推理

        Args:
            batch: 批量请求数据列表

        Returns:
            批量推理结果
        """
        request_data = {'batch': batch}

        start_time = time.time()

        response = self.session.post(
            f'{self.base_url}/inference/batch',
            json=request_data,
            timeout=self.timeout,
        )
        response.raise_for_status()

        total_time = (time.time() - start_time) * 1000
        result = response.json()
        result['total_time_ms'] = total_time

        return result

    def create_demo_request(self) -> Dict[str, Any]:
        """
        创建演示请求

        Returns:
            演示请求数据
        """
        N, H, W = 6, 370, 1224
        num_points = 10000

        images = np.random.randint(0, 255, (N, H, W, 3), dtype=np.uint8).tolist()
        intrinsics = np.eye(3, dtype=np.float32).tolist()
        intrinsics = [intrinsics] * N
        extrinsics = np.eye(4, dtype=np.float32).tolist()
        extrinsics = [extrinsics] * N
        point_cloud = (np.random.randn(num_points, 4) * 10).tolist()

        return {
            'images': images,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'point_cloud': point_cloud,
        }


def demo_single_inference(base_url: str = 'http://localhost:8000'):
    """
    演示单样本推理

    Args:
        base_url: API 服务器地址
    """
    print("=" * 60)
    print("Single Inference Demo")
    print("=" * 60)

    client = BEVTextCLIPClient(base_url)

    print("\n[1] Health check...")
    health = client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Device: {health['device']}")
    print(f"   Model loaded: {health['model_loaded']}")

    print("\n[2] Getting model info...")
    info = client.get_info()
    print(f"   Model: {info['model_name']}")
    print(f"   Classes: {info['num_classes']}")

    print("\n[3] Creating demo request...")
    request = client.create_demo_request()
    print(f"   Images shape: {len(request['images'])}, {len(request['images'][0])}x{len(request['images'][0][0])}")
    print(f"   Point cloud: {len(request['point_cloud'])} points")

    print("\n[4] Running inference...")
    result = client.inference(
        request['images'],
        request['intrinsics'],
        request['extrinsics'],
        request['point_cloud'],
    )

    print(f"\n[5] Results:")
    print(f"   Predictions shape: {result.predictions.shape}")
    print(f"   Probabilities shape: {result.probabilities.shape}")
    print(f"   API inference time: {result.inference_time_ms:.2f} ms")

    class_counts = {}
    for i in range(result.predictions.shape[0]):
        for j in range(result.predictions.shape[1]):
            pred = result.predictions[i, j]
            class_counts[pred] = class_counts.get(pred, 0) + 1

    print(f"\n   Class distribution (top 5):")
    for class_id, count in sorted(class_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"      Class {class_id}: {count} pixels")


def demo_batch_inference(base_url: str = 'http://localhost:8000', batch_size: int = 4):
    """
    演示批量推理

    Args:
        base_url: API 服务器地址
        batch_size: 批量大小
    """
    print("\n" + "=" * 60)
    print(f"Batch Inference Demo (batch_size={batch_size})")
    print("=" * 60)

    client = BEVTextCLIPClient(base_url)

    print("\n[1] Creating batch requests...")
    batch = [client.create_demo_request() for _ in range(batch_size)]
    print(f"   Batch size: {len(batch)}")

    print("\n[2] Running batch inference...")
    result = client.batch_inference(batch)

    print(f"\n[3] Results:")
    print(f"   Total time: {result['total_time_ms']:.2f} ms")
    print(f"   Avg time per sample: {result['avg_time_ms']:.2f} ms")
    print(f"   Throughput: {batch_size / (result['total_time_ms'] / 1000):.2f} samples/sec")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='BEV-TextCLIP API Client')
    parser.add_argument('--url', type=str, default='http://localhost:8000', help='API server URL')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for demo')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("BEV-TextCLIP API Client Demo")
    print("=" * 60)

    try:
        demo_single_inference(args.url)
        demo_batch_inference(args.url, args.batch_size)

        print("\n" + "=" * 60)
        print("All demos completed!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to API server")
        print(f"Please start the server first: uvicorn inference_api:app --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == '__main__':
    main()
