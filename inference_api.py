#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: 推理 API 服务，基于 FastAPI 的 RESTful 推理服务
日期: 2026年1月23日
"""

import sys
import os
import time
import argparse
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.configs.bev_textclip_config import get_config
from src.models.bev_textclip import create_bev_textclip_model


class InferenceRequest(BaseModel):
    """
    推理请求模型
    """
    images: List[List[List[List[int]]]] = Field(..., description='多视角图像 [N, H, W, C]')
    intrinsics: List[List[List[float]]] = Field(..., description='相机内参 [N, 3, 3]')
    extrinsics: List[List[List[float]]] = Field(..., description='相机外参 [N, 4, 4]')
    point_cloud: List[List[float]] = Field(..., description='点云 [M, 4]')


class BatchInferenceRequest(BaseModel):
    """
    批量推理请求模型
    """
    batch: List[InferenceRequest]


class InferenceResponse(BaseModel):
    """
    推理响应模型
    """
    predictions: List[List[int]] = Field(..., description='分割预测 [H, W]')
    probabilities: List[List[List[float]]] = Field(..., description='类别概率 [H, W, C]')
    inference_time_ms: float = Field(..., description='推理耗时 (毫秒)')
    model_info: Dict[str, Any] = Field(..., description='模型信息')


class HealthResponse(BaseModel):
    """
    健康检查响应
    """
    status: str
    device: str
    model_loaded: bool


class InfoResponse(BaseModel):
    """
    模型信息响应
    """
    model_name: str
    num_classes: int
    input_shape: Dict[str, List[int]]
    output_shape: Dict[str, List[int]]


_global_model = None
_global_config = None
_device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    """
    global _global_model, _global_config, _device

    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing inference service on {_device}")

    config = get_config('dummy')
    _global_config = config

    _global_model = create_bev_textclip_model(config)
    _global_model = _global_model.to(_device)
    _global_model.eval()

    print(f"Model loaded: {sum(p.numel() for p in _global_model.parameters()):,} parameters")

    yield

    print("Shutting down inference service")


app = FastAPI(
    title='BEV-TextCLIP Inference API',
    description='BEV-TextCLIP 多模态语义分割推理服务',
    version='1.0.0',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


def preprocess_request(request: InferenceRequest) -> Dict[str, torch.Tensor]:
    """
    预处理推理请求

    Args:
        request: 推理请求

    Returns:
        模型输入字典
    """
    images = np.array(request.images, dtype=np.float32)
    images = torch.from_numpy(images).permute(0, 3, 1, 2).unsqueeze(0)

    intrinsics = np.array(request.intrinsics, dtype=np.float32)
    intrinsics = torch.from_numpy(intrinsics).unsqueeze(0)

    extrinsics = np.array(request.extrinsics, dtype=np.float32)
    extrinsics = torch.from_numpy(extrinsics).unsqueeze(0)

    point_cloud = np.array(request.point_cloud, dtype=np.float32)
    point_cloud = torch.from_numpy(point_cloud)

    point_cloud_lengths = torch.tensor([len(point_cloud)], dtype=torch.long)

    images = images.to(_device)
    intrinsics = intrinsics.to(_device)
    extrinsics = extrinsics.to(_device)
    point_cloud = point_cloud.to(_device)
    point_cloud_lengths = point_cloud_lengths.to(_device)

    return {
        'images': images,
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'point_cloud': point_cloud,
        'point_cloud_lengths': point_cloud_lengths,
    }


def postprocess_output(output: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    后处理推理输出

    Args:
        output: 模型输出

    Returns:
        响应字典
    """
    predictions = output['segmentation_logits'].argmax(dim=1).squeeze(0).cpu().numpy().tolist()
    probabilities = torch.softmax(output['segmentation_logits'], dim=1).squeeze(0).cpu().numpy().tolist()

    return {
        'predictions': predictions,
        'probabilities': probabilities,
    }


@app.get('/health', response_model=HealthResponse)
async def health_check():
    """
    健康检查接口
    """
    return HealthResponse(
        status='healthy',
        device=str(_device),
        model_loaded=_global_model is not None,
    )


@app.get('/info', response_model=InfoResponse)
async def get_info():
    """
    获取模型信息
    """
    if _global_model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    return InfoResponse(
        model_name='BEV-TextCLIP',
        num_classes=_global_config.num_classes,
        input_shape={
            'images': [1, 6, 3, 370, 1224],
            'intrinsics': [1, 6, 3, 3],
            'extrinsics': [1, 6, 4, 4],
            'point_cloud': [10000, 4],
        },
        output_shape={
            'segmentation_logits': [1, 16, 200, 200],
            'bev_features': [1, 256, 200, 200],
        },
    )


@app.post('/inference', response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """
    单样本推理接口
    """
    if _global_model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    start_time = time.time()

    try:
        inputs = preprocess_request(request)

        with torch.no_grad():
            output = _global_model(**inputs)

        results = postprocess_output(output)

        inference_time = (time.time() - start_time) * 1000

        return InferenceResponse(
            predictions=results['predictions'],
            probabilities=results['probabilities'],
            inference_time_ms=inference_time,
            model_info={
                'model_name': 'BEV-TextCLIP',
                'num_classes': _global_config.num_classes,
                'device': str(_device),
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/inference/batch')
async def batch_inference(request: BatchInferenceRequest):
    """
    批量推理接口
    """
    if _global_model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    start_time = time.time()
    results = []

    for i, item in enumerate(request.batch):
        try:
            inputs = preprocess_request(item)

            with torch.no_grad():
                output = _global_model(**inputs)

            postprocessed = postprocess_output(output)
            results.append({
                'index': i,
                'predictions': postprocessed['predictions'],
                'probabilities': postprocessed['probabilities'],
            })

        except Exception as e:
            results.append({
                'index': i,
                'error': str(e),
            })

    total_time = (time.time() - start_time) * 1000

    return JSONResponse({
        'results': results,
        'total_time_ms': total_time,
        'avg_time_ms': total_time / len(request.batch),
    })


def create_demo_data() -> InferenceRequest:
    """
    创建演示数据

    Returns:
        演示推理请求
    """
    B, N = 1, 6
    H, W = 370, 1224
    num_points = 10000

    images = np.random.randint(0, 255, (N, H, W, 3), dtype=np.uint8).tolist()
    intrinsics = np.eye(3, dtype=np.float32).tolist()
    intrinsics = [intrinsics] * N
    extrinsics = np.eye(4, dtype=np.float32).tolist()
    extrinsics = [extrinsics] * N
    point_cloud = (np.random.randn(num_points, 4) * 10).tolist()

    return InferenceRequest(
        images=images,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        point_cloud=point_cloud,
    )


@app.get('/demo')
async def get_demo():
    """
    获取演示请求数据
    """
    demo_request = create_demo_data()
    return demo_request.model_dump()


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='BEV-TextCLIP Inference API')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("BEV-TextCLIP Inference API")
    print("=" * 60)
    print(f"\nStarting server at http://{args.host}:{args.port}")
    print("\nEndpoints:")
    print("  GET  /health       - Health check")
    print("  GET  /info         - Model information")
    print("  GET  /demo         - Demo request data")
    print("  POST /inference    - Single inference")
    print("  POST /inference/batch - Batch inference")
    print("=" * 60)

    uvicorn.run(
        'inference_api:app',
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == '__main__':
    main()
