#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: BEV-TextCLIP 跨模态注意力模块，实现 BEV 特征与文本特征之间的跨模态注意力机制
日期: 2026年1月22日
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import math


class MultiHeadAttention(nn.Module):
    """

    多头注意力模块

    标准的多头自注意力实现

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        """

        初始化多头注意力

        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout 比率
            bias: 是否使用偏置

        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        前向传播

        Args:
            query: 查询张量 [B, L, C]
            key: 键张量 [B, S, C]
            value: 值张量 [B, S, C]
            key_padding_mask: 键填充掩码 [B, S]
            attn_mask: 注意力掩码 [L, S]

        Returns:
            output: 输出张量 [B, L, C]
            attention_weights: 注意力权重 [B, H, L, S]

        """
        B, L, C = query.shape
        _, S, _ = key.shape
        H = self.num_heads

        q = self.q_proj(query).view(B, L, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, S, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, S, H, self.head_dim).transpose(1, 2)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))

        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2) == 0,
                float('-inf'),
            )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = (attn_weights @ v).transpose(1, 2).contiguous().view(B, L, C)

        output = self.out_proj(output)

        return output, attn_weights


class BEVTextCrossAttention(nn.Module):
    """

    BEV 特征与文本特征之间的跨模态注意力机制

    支持两种模式:
    1. 单向注意力: BEV 关注文本
    2. 双向注意力: BEV 关注文本 + 文本关注 BEV

    """

    def __init__(
        self,
        bev_channels: int = 256,
        text_channels: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_bidirectional: bool = True,
    ):
        """

        初始化跨模态注意力

        Args:
            bev_channels: BEV 特征通道数
            text_channels: 文本特征通道数
            num_heads: 注意力头数
            dropout: Dropout 比率
            use_bidirectional: 是否使用双向注意力

        """
        super().__init__()

        self.bev_channels = bev_channels
        self.text_channels = text_channels
        self.num_heads = num_heads
        self.use_bidirectional = use_bidirectional

        self.bev_proj = nn.Linear(bev_channels, bev_channels)
        self.text_proj = nn.Linear(text_channels, bev_channels)

        self.attention = MultiHeadAttention(
            embed_dim=bev_channels,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.gate_net = nn.Sequential(
            nn.Linear(bev_channels * 2, bev_channels),
            nn.Sigmoid(),
        )

        self.ffn = nn.Sequential(
            nn.Linear(bev_channels, bev_channels * 4),
            nn.GELU(),
            nn.Linear(bev_channels * 4, bev_channels),
        )

        self.norm1 = nn.LayerNorm(bev_channels)
        self.norm2 = nn.LayerNorm(bev_channels)
        self.norm3 = nn.LayerNorm(bev_channels)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        bev_features: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        前向传播

        Args:
            bev_features: BEV 特征 [B, L, C] (L = X*Y)
            text_embeddings: 文本嵌入 [K, C] 或 [B, K, C]

        Returns:
            enhanced_bev: 增强后的 BEV 特征 [B, L, C]
            attention_weights: 注意力权重 [B, H, L, K]

        """
        B, L, C = bev_features.shape

        bev_h = self.bev_proj(bev_features)

        if text_embeddings.dim() == 2:
            text_embeddings = text_embeddings.unsqueeze(0).expand(B, -1, -1)

        B, K, _ = text_embeddings.shape
        text_h = self.text_proj(text_embeddings)

        attn_output, attn_weights = self.attention(
            query=bev_h,
            key=text_h,
            value=text_h,
        )
        attn_output = self.dropout(attn_output)

        fused = torch.cat([bev_h, attn_output], dim=-1)
        gate = self.gate_net(fused)

        bev_enhanced = bev_h + gate * (attn_output - bev_h)
        bev_enhanced = self.norm1(bev_enhanced)

        bev_enhanced = bev_enhanced + self.ffn(bev_enhanced)
        bev_enhanced = self.norm2(bev_enhanced)

        if self.use_bidirectional:
            text_attn_output, text_attn_weights = self.attention(
                query=text_h,
                key=bev_h,
                value=bev_h,
            )
            text_attn_output = self.dropout(text_attn_output)

            text_enhanced = text_h + text_attn_output
            text_enhanced = self.norm3(text_enhanced)

            text_to_bev = self.text_proj(text_enhanced)
            bev_enhanced = bev_enhanced + text_to_bev.mean(dim=1, keepdim=True)

            return bev_enhanced, attn_weights

        return bev_enhanced, attn_weights


class MultiScaleBEVTextAttention(nn.Module):
    """

    多尺度 BEV-文本注意力

    在不同尺度的 BEV 特征上进行注意力操作，然后融合

    """

    def __init__(
        self,
        bev_channels: int = 256,
        text_channels: int = 512,
        num_heads: int = 8,
        scales: int = 3,
    ):
        """

        初始化多尺度注意力

        Args:
            bev_channels: BEV 特征通道数
            text_channels: 文本特征通道数
            num_heads: 注意力头数
            scales: 尺度数量

        """
        super().__init__()

        self.scales = scales

        self.attn_layers = nn.ModuleList()
        for i in range(scales):
            scale_channels = bev_channels // (2 ** i)
            self.attn_layers.append(
                BEVTextCrossAttention(
                    bev_channels=scale_channels,
                    text_channels=text_channels,
                    num_heads=num_heads,
                )
            )

        self.fusion_conv = nn.Conv2d(bev_channels * scales, bev_channels, 1)

    def forward(
        self,
        bev_features_multi_scale: List[torch.Tensor],
        text_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """

        前向传播

        Args:
            bev_features_multi_scale: 多尺度 BEV 特征列表
                [B, C, H, W], [B, C/2, 2H, 2W], ...
            text_embeddings: 文本嵌入 [K, C]

        Returns:
            enhanced_bev: 增强后的 BEV 特征 [B, C, H, W]
            attention_weights_list: 注意力权重列表

        """
        enhanced_features = []
        attention_weights_list = []

        for i, bev in enumerate(bev_features_multi_scale):
            B, C, H, W = bev.shape

            bev_flat = bev.flatten(2).transpose(1, 2)

            enhanced, attn_weights = self.attn_layers[i](bev_flat, text_embeddings)

            enhanced = enhanced.transpose(1, 2).view(B, -1, H, W)
            enhanced_features.append(enhanced)
            attention_weights_list.append(attn_weights)

        fused = torch.cat(enhanced_features, dim=1)
        enhanced_bev = self.fusion_conv(fused)

        return enhanced_bev, attention_weights_list


class CrossModalAlignment(nn.Module):
    """

    跨模态对齐模块

    用于对齐不同模态的特征空间

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
    ):
        """

        初始化跨模态对齐模块

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            num_layers: 对齐层数

        """
        super().__init__()

        self.align_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.align_layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, out_channels),
                    nn.LayerNorm(out_channels),
                    nn.GELU(),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        前向传播

        Args:
            x: 输入特征 [B, C]

        Returns:
            aligned: 对齐后的特征 [B, C]

        """
        for layer in self.align_layers:
            x = layer(x)
        return x


class BevTextInteraction(nn.Module):
    """

    BEV-文本交互模块

    整合 BEV-文本交叉注意力和跨模态对齐

    """

    def __init__(
        self,
        bev_channels: int = 256,
        text_channels: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_bidirectional: bool = True,
    ):
        """

        初始化交互模块

        Args:
            bev_channels: BEV 特征通道数
            text_channels: 文本特征通道数
            num_heads: 注意力头数
            dropout: Dropout 比率
            use_bidirectional: 是否使用双向注意力

        """
        super().__init__()

        self.cross_attention = BEVTextCrossAttention(
            bev_channels=bev_channels,
            text_channels=text_channels,
            num_heads=num_heads,
            dropout=dropout,
            use_bidirectional=use_bidirectional,
        )

        self.bev_align = CrossModalAlignment(bev_channels, bev_channels, num_layers=2)
        self.text_align = CrossModalAlignment(text_channels, text_channels, num_layers=2)

        self.output_proj = nn.Linear(bev_channels, bev_channels)

    def forward(
        self,
        bev_features: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """

        前向传播

        Args:
            bev_features: BEV 特征 [B, C, H, W]
            text_embeddings: 文本嵌入 [K, C]

        Returns:
            output_dict: {
                'enhanced_bev': 增强后的 BEV 特征,
                'attention_weights': 注意力权重,
                'text_embeddings': 对齐后的文本嵌入,
            }

        """
        B, C, H, W = bev_features.shape

        bev_flat = bev_features.flatten(2).transpose(1, 2)
        bev_aligned = self.bev_align(bev_flat.mean(dim=1))

        text_aligned = self.text_align(text_embeddings)

        enhanced_bev_flat, attn_weights = self.cross_attention(bev_flat, text_aligned)

        enhanced_bev = enhanced_bev_flat.transpose(1, 2).view(B, C, H, W)

        output_dict = {
            'enhanced_bev': enhanced_bev,
            'attention_weights': attn_weights,
            'text_embeddings': text_aligned,
            'bev_global': bev_aligned,
        }

        return output_dict
