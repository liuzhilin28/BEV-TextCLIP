#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: BEV-TextCLIP 融合模块，多模态 BEV 特征融合，支持多种融合策略
日期: 2026年1月22日
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from .cross_attention import BEVTextCrossAttention


class ChannelAttention(nn.Module):
    """

    通道注意力模块

    对特征图进行通道维度的注意力加权

    """

    def __init__(self, channels: int, reduction: int = 16):
        """

        初始化通道注意力

        Args:
            channels: 输入通道数
            reduction: 降维比例

        """
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        前向传播

        Args:
            x: 输入特征 [B, C, H, W]

        Returns:
            out: 加权后的特征 [B, C, H, W]

        """
        B, C, H, W = x.shape

        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))

        out = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)

        return x * out


class SpatialAttention(nn.Module):
    """

    空间注意力模块

    对特征图进行空间维度的注意力加权

    """

    def __init__(self, kernel_size: int = 7):
        """

        初始化空间注意力

        Args:
            kernel_size: 卷积核大小

        """
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        前向传播

        Args:
            x: 输入特征 [B, C, H, W]

        Returns:
            out: 加权后的特征 [B, C, H, W]

        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))

        return x * out


class GatedFeatureFusion(nn.Module):
    """

    门控特征融合模块

    通过门控机制自适应融合图像 BEV 和点云 BEV 特征

    基于 BEVFusion 实现

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduction: int = 16,
    ):
        """

        初始化门控融合

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            reduction: 注意力降维比例

        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.image_channel_attention = ChannelAttention(in_channels, reduction)
        self.point_channel_attention = ChannelAttention(in_channels, reduction)

        self.image_spatial_attention = SpatialAttention()
        self.point_spatial_attention = SpatialAttention()

        self.gate_net = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 2, 1),
            nn.Softmax(dim=1),
        )

        self.fusion_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(
        self,
        image_bev: torch.Tensor,
        point_bev: torch.Tensor,
    ) -> torch.Tensor:
        """

        前向传播

        Args:
            image_bev: 图像 BEV 特征 [B, C, X, Y]
            point_bev: 点云 BEV 特征 [B, C, X, Y]

        Returns:
            fused_bev: 融合后的 BEV 特征 [B, C', X, Y]

        """
        if image_bev.shape != point_bev.shape:
            point_bev = F.interpolate(
                point_bev,
                size=image_bev.shape[2:],
                mode='bilinear',
                align_corners=True,
            )

        image_attended = self.image_channel_attention(image_bev)
        image_attended = self.image_spatial_attention(image_attended)

        point_attended = self.point_channel_attention(point_bev)
        point_attended = self.point_spatial_attention(point_attended)

        concat_features = torch.cat([image_attended, point_attended], dim=1)

        gate = self.gate_net(concat_features)
        gate_image = gate[:, 0:1, :, :]
        gate_point = gate[:, 1:2, :, :]

        gated_fusion = gate_image * image_attended + gate_point * point_attended

        fused_bev = self.fusion_conv(gated_fusion)

        return fused_bev


class GatedCrossAttentionFusion(nn.Module):
    """

    门控交叉注意力融合

    在 BEV 空间中进行门控交叉注意力融合

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """

        初始化门控交叉注意力融合

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            num_heads: 注意力头数
            dropout: Dropout 比率

        """
        super().__init__()

        self.image_proj = nn.Conv2d(in_channels, out_channels, 1)
        self.point_proj = nn.Conv2d(in_channels, out_channels, 1)

        self.cross_attention = BEVTextCrossAttention(
            bev_channels=out_channels,
            text_channels=out_channels,
            num_heads=num_heads,
            dropout=dropout,
            use_bidirectional=True,
        )

        self.gate_net = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 2, 1),
            nn.Sigmoid(),
        )

        self.fusion_conv = nn.Conv2d(out_channels * 2, out_channels, 1)

    def forward(
        self,
        image_bev: torch.Tensor,
        point_bev: torch.Tensor,
    ) -> torch.Tensor:
        """

        前向传播

        Args:
            image_bev: 图像 BEV 特征 [B, C, X, Y]
            point_bev: 点云 BEV 特征 [B, C, X, Y]

        Returns:
            fused_bev: 融合后的 BEV 特征 [B, C, X, Y]

        """
        B, C, H, W = image_bev.shape

        image_proj = self.image_proj(image_bev)
        point_proj = self.point_proj(point_bev)

        image_flat = image_proj.flatten(2).transpose(1, 2)
        point_flat = point_proj.flatten(2).transpose(1, 2)

        image_global = image_proj.mean(dim=[2, 3])
        point_global = point_proj.mean(dim=[2, 3])

        gate_value = self.gate_net(
            torch.cat([image_global, point_global], dim=-1)
        ).view(B, 1, 1, 1)

        cross_image = self.cross_attention(image_flat, point_flat)
        cross_point = self.cross_attention(point_flat, image_flat)

        cross_image = cross_image.transpose(1, 2).view(B, C, H, W)
        cross_point = cross_point.transpose(1, 2).view(B, C, H, W)

        fused = gate_value * cross_image + (1 - gate_value) * cross_point

        output = self.fusion_conv(fused)

        return output


class ConcatenationFusion(nn.Module):
    """

    拼接融合模块

    将图像 BEV 和点云 BEV 特征拼接后通过卷积融合

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        """

        初始化拼接融合

        Args:
            in_channels: 输入通道数 (图像 + 点云)
            out_channels: 输出通道数

        """
        super().__init__()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        image_bev: torch.Tensor,
        point_bev: torch.Tensor,
    ) -> torch.Tensor:
        """

        前向传播

        Args:
            image_bev: 图像 BEV 特征 [B, C, X, Y]
            point_bev: 点云 BEV 特征 [B, C, X, Y]

        Returns:
            fused_bev: 融合后的 BEV 特征 [B, C', X, Y]

        """
        if image_bev.shape != point_bev.shape:
            point_bev = F.interpolate(
                point_bev,
                size=image_bev.shape[2:],
                mode='bilinear',
                align_corners=True,
            )

        concat_bev = torch.cat([image_bev, point_bev], dim=1)
        fused_bev = self.fusion_conv(concat_bev)

        return fused_bev


class AdditionFusion(nn.Module):
    """

    相加融合模块

    将图像 BEV 和点云 BEV 特征相加

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        """

        初始化相加融合

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数

        """
        super().__init__()

        self.image_proj = nn.Conv2d(in_channels, out_channels, 1)
        self.point_proj = nn.Conv2d(in_channels, out_channels, 1)

    def forward(
        self,
        image_bev: torch.Tensor,
        point_bev: torch.Tensor,
    ) -> torch.Tensor:
        """

        前向传播

        Args:
            image_bev: 图像 BEV 特征 [B, C, X, Y]
            point_bev: 点云 BEV 特征 [B, C, X, Y]

        Returns:
            fused_bev: 融合后的 BEV 特征 [B, C', X, Y]

        """
        if image_bev.shape != point_bev.shape:
            point_bev = F.interpolate(
                point_bev,
                size=image_bev.shape[2:],
                mode='bilinear',
                align_corners=True,
            )

        image_proj = self.image_proj(image_bev)
        point_proj = self.point_proj(point_bev)

        fused_bev = image_proj + point_proj

        return fused_bev


class BevFusion(nn.Module):
    """

    BEV 融合模块主类

    根据配置选择不同的融合策略

    支持的融合策略:
    - 'gated_attention': 门控注意力融合
    - 'gated_cross_attention': 门控交叉注意力融合
    - 'concatenation': 拼接融合
    - 'addition': 相加融合

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fusion_type: str = "gated_attention",
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """

        初始化 BEV 融合模块

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            fusion_type: 融合类型
            num_heads: 注意力头数
            dropout: Dropout 比率

        """
        super().__init__()

        self.fusion_type = fusion_type

        if fusion_type == "gated_attention":
            self.fusion = GatedFeatureFusion(
                in_channels=in_channels,
                out_channels=out_channels,
            )
        elif fusion_type == "gated_cross_attention":
            self.fusion = GatedCrossAttentionFusion(
                in_channels=in_channels,
                out_channels=out_channels,
                num_heads=num_heads,
                dropout=dropout,
            )
        elif fusion_type == "concatenation":
            self.fusion = ConcatenationFusion(
                in_channels=in_channels,
                out_channels=out_channels,
            )
        elif fusion_type == "addition":
            self.fusion = AdditionFusion(
                in_channels=in_channels,
                out_channels=out_channels,
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(
        self,
        image_bev: torch.Tensor,
        point_bev: torch.Tensor,
    ) -> torch.Tensor:
        """

        前向传播

        Args:
            image_bev: 图像 BEV 特征 [B, C, X, Y]
            point_bev: 点云 BEV 特征 [B, C, X, Y]

        Returns:
            fused_bev: 融合后的 BEV 特征 [B, C', X, Y]

        """
        return self.fusion(image_bev, point_bev)
