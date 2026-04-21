#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: 模型包
日期: 2026年1月22日
"""

from .image_encoder import (
    ResNetEncoder,
    ViTEncoder,
    LiftSplatTransform,
    ImageEncoder,
)

from .point_encoder import (
    Voxelization,
    VoxelNet,
    PointPillarEncoder,
    PointEncoder,
)

from .fusion_module import (
    ChannelAttention,
    SpatialAttention,
    GatedFeatureFusion,
    GatedCrossAttentionFusion,
    ConcatenationFusion,
    AdditionFusion,
    BevFusion,
)

from .cross_attention import (
    MultiHeadAttention,
    BEVTextCrossAttention,
    MultiScaleBEVTextAttention,
    CrossModalAlignment,
    BevTextInteraction,
)

from .text_encoder import (
    CLIPTextEncoder,
    CustomTextEncoder,
    LocalCLIPTextEncoder,
    TextEncoder,
    TextPromptLearner,
    CategoryEmbedder,
)

from .losses import (
    InfoNCE,
    MultiModalContrastiveLoss,
    FocalLoss,
    DiceLoss,
    SegmentationLoss,
    BEVTextCLIPLoss,
)

from .bev_textclip import (
    BEVTextCLIP,
    BEVTextCLIPLightning,
    create_bev_textclip_model,
)

__all__ = [
    'ResNetEncoder',
    'ViTEncoder',
    'LiftSplatTransform',
    'ImageEncoder',
    'Voxelization',
    'VoxelNet',
    'PointPillarEncoder',
    'PointEncoder',
    'ChannelAttention',
    'SpatialAttention',
    'GatedFeatureFusion',
    'GatedCrossAttentionFusion',
    'ConcatenationFusion',
    'AdditionFusion',
    'BevFusion',
    'MultiHeadAttention',
    'BEVTextCrossAttention',
    'MultiScaleBEVTextAttention',
    'CrossModalAlignment',
    'BevTextInteraction',
    'CLIPTextEncoder',
    'CustomTextEncoder',
    'LocalCLIPTextEncoder',
    'TextEncoder',
    'TextPromptLearner',
    'CategoryEmbedder',
    'InfoNCE',
    'MultiModalContrastiveLoss',
    'FocalLoss',
    'DiceLoss',
    'SegmentationLoss',
    'BEVTextCLIPLoss',
    'BEVTextCLIP',
    'BEVTextCLIPLightning',
    'create_bev_textclip_model',
]
