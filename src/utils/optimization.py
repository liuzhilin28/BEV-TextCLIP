#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: BEV-TextCLIP 性能优化工具模块，包含梯度检查点、混合精度、推理优化等功能
日期: 2026年1月22日
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import gc


class GradientCheckpointing:
    """
    梯度检查点管理器

    用于节省显存，适合大模型训练
    """

    def __init__(self, enabled: bool = True):
        """
        初始化梯度检查点管理器

        Args:
            enabled: 是否启用梯度检查点
        """
        self.enabled = enabled

    def checkpoint(self, module: nn.Module, *args, **kwargs) -> torch.Tensor:
        """
        对模块应用梯度检查点

        Args:
            module: 要检查点的模块
            *args: 输入参数

        Returns:
            输出张量
        """
        if self.enabled:
            return torch.utils.checkpoint.checkpoint(module, *args, **kwargs)
        return module(*args, **kwargs)


class MixedPrecisionManager:
    """
    混合精度训练管理器

    支持 FP16/BF16 混合精度训练
    """

    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16,
        amp_backend: str = "native",
    ):
        """
        初始化混合精度管理器

        Args:
            enabled: 是否启用混合精度
            dtype: 精度类型 (float16/bfloat16)
            amp_backend: AMP 后端 ('native' 或 'apex')
        """
        self.enabled = enabled
        self.dtype = dtype
        self.amp_backend = amp_backend
        self.scaler = None

        if enabled and amp_backend == "native":
            self.scaler = torch.amp.GradScaler(device="cuda" if torch.cuda.is_available() else "cpu")

    def autocast(self, device: str = "cuda"):
        """
        创建自动精度上下文

        Args:
            device: 设备类型

        Returns:
            上下文管理器
        """
        if self.enabled:
            return torch.amp.autocast(device, dtype=self.dtype)
        return torch.cuda.amp.autocast(enabled=False) if device == "cuda" else F.contextlib.nullcontext()

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        缩放损失值

        Args:
            loss: 原始损失

        Returns:
            缩放后的损失
        """
        if self.enabled and self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer):
        """
        执行优化器步骤

        Args:
            optimizer: 优化器
        """
        if self.enabled and self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def update(self):
        """更新梯度缩放器"""
        if self.enabled and self.scaler is not None:
            self.scaler.update()


class MemoryOptimizer:
    """
    内存优化工具

    提供显存清理、动态分配等功能
    """

    def __init__(self, enabled: bool = True, max_memory_fraction: float = 0.9):
        """
        初始化内存优化器

        Args:
            enabled: 是否启用内存优化
            max_memory_fraction: 最大显存使用比例
        """
        self.enabled = enabled
        self.max_memory_fraction = max_memory_fraction

    def set_memory_limit(self):
        """设置显存限制"""
        if self.enabled and torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(self.max_memory_fraction)

    def clear_cache(self):
        """清理显存缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def get_memory_info(self) -> Dict[str, float]:
        """
        获取当前显存信息

        Returns:
            显存信息字典
        """
        info = {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
        }

        if torch.cuda.is_available():
            info["allocated_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
            info["reserved_gb"] = torch.cuda.memory_reserved() / (1024 ** 3)
            info["max_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)

        return info


class InferenceOptimizer:
    """
    推理优化工具

    提供模型推理优化功能
    """

    def __init__(self):
        self.optimizations_applied = []

    def optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """
        优化模型以进行推理

        Args:
            model: 待优化的模型

        Returns:
            优化后的模型
        """
        model.eval()

        model = torch.jit.script(model)
        self.optimizations_applied.append("torch.jit.script")

        return model

    def enable_cudnn_benchmark(self, enabled: bool = True):
        """
        启用/禁用 cuDNN benchmark

        Args:
            enabled: 是否启用
        """
        if enabled:
            torch.backends.cudnn.benchmark = True
            self.optimizations_applied.append("cudnn.benchmark")
        else:
            torch.backends.cudnn.benchmark = False

    def set_determinism(self, enabled: bool = True):
        """
        设置确定性计算

        Args:
            enabled: 是否启用
        """
        if enabled:
            torch.use_deterministic_algorithms(True)
            torch.set_deterministic_debug_mode(1)
            self.optimizations_applied.append("deterministic_algorithms")
        else:
            torch.use_deterministic_algorithms(False)
            torch.set_deterministic_debug_mode(0)


def get_optimizer_config(
    model: nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    momentum: float = 0.9,
) -> Dict[str, Any]:
    """
    获取优化器配置

    Args:
        model: 模型
        optimizer_type: 优化器类型 ('adamw', 'adam', 'sgd')
        learning_rate: 学习率
        weight_decay: 权重衰减
        beta1: Adam beta1
        beta2: Adam beta2
        momentum: SGD 动量

    Returns:
        优化器配置字典
    """
    config = {
        "optimizer_type": optimizer_type,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
    }

    if optimizer_type == "adamw":
        config["optimizer"] = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
        )
    elif optimizer_type == "adam":
        config["optimizer"] = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
        )
    elif optimizer_type == "sgd":
        config["optimizer"] = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return config


def get_scheduler_config(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    num_epochs: int = 100,
    min_lr: float = 1e-6,
    warmup_epochs: int = 5,
) -> Dict[str, Any]:
    """
    获取学习率调度器配置

    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型 ('cosine', 'step', 'linear', 'warmup')
        num_epochs: 总轮数
        min_lr: 最小学习率
        warmup_epochs: 预热轮数

    Returns:
        调度器配置字典
    """
    config = {"scheduler_type": scheduler_type}

    if scheduler_type == "cosine":
        config["scheduler"] = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=min_lr,
        )
    elif scheduler_type == "step":
        config["scheduler"] = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=num_epochs // 3,
            gamma=0.1,
        )
    elif scheduler_type == "linear":
        config["scheduler"] = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr / optimizer.param_groups[0]["lr"],
            total_iters=num_epochs,
        )
    elif scheduler_type == "warmup":
        config["scheduler"] = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=num_epochs - warmup_epochs,
                    eta_min=min_lr,
                ),
            ],
            milestones=[warmup_epochs],
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return config
