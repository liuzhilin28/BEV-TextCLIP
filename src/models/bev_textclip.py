#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: BEV-TextCLIP 主模型、多模态语义分割主模型
日期: 2026年1月22日
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any, Union
from .image_encoder import ImageEncoder
from .point_encoder import PointEncoder
from .fusion_module import BevFusion
from .cross_attention import BevTextInteraction
from .text_encoder import TextEncoder, CategoryEmbedder
from .losses import BEVTextCLIPLoss


class BEVTextCLIP(nn.Module):
    """

    BEV-TextCLIP 主模型

    多模态语义分割模型:
    1. 图像 → LSS → Image BEV
    2. 点云 → PointPillar → Point BEV
    3. 类别文本 → CLIP → Text Embedding
    4. Image BEV + Point BEV → Fusion → Fused BEV
    5. Fused BEV + Text Embedding → Cross-Attention → Enhanced BEV
    6. Enhanced BEV → Segmentation Head → 分割结果

    """

    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        image_encoder_type: str = "resnet50",
        point_encoder_type: str = "pointpillar",
        text_encoder_type: str = "clip",
        fusion_type: str = "gated_attention",
        bev_resolution: Tuple[int, int] = (200, 200),
        bev_channels: int = 256,
        point_cloud_range: Tuple[float, float, float, float, float, float] = (
            -20.0, -20.0, -2.0, 20.0, 20.0, 6.0
        ),
        use_contrastive: bool = True,
        contrastive_weight: float = 0.5,
        pretrained: bool = True,
        freeze_image_encoder: bool = True,
        freeze_point_encoder: bool = True,
        freeze_text_encoder: bool = True,
    ):
        """

        初始化 BEV-TextCLIP

        Args:
            num_classes: 类别数
            class_names: 类别名称列表
            image_encoder_type: 图像编码器类型
            point_encoder_type: 点云编码器类型
            text_encoder_type: 文本编码器类型
            fusion_type: 融合类型
            bev_resolution: BEV 分辨率 [X, Y]
            bev_channels: BEV 通道数
            point_cloud_range: 点云范围
            use_contrastive: 是否使用对比学习
            contrastive_weight: 对比损失权重
            pretrained: 是否使用预训练权重
            freeze_image_encoder: 是否冻结图像编码器
            freeze_point_encoder: 是否冻结点云编码器
            freeze_text_encoder: 是否冻结文本编码器

        """
        super().__init__()

        self.num_classes = num_classes
        self.class_names = class_names
        self.bev_resolution = bev_resolution
        self.bev_channels = bev_channels
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.point_cloud_range = point_cloud_range

        self.image_encoder = ImageEncoder(
            in_channels=3,
            out_channels=bev_channels,
            image_encoder_type=image_encoder_type,
            bev_grid_size=bev_resolution,
            bev_depth_bins=4,
            point_cloud_range=point_cloud_range,
            pretrained=pretrained,
            freeze=freeze_image_encoder,
        )

        self.point_encoder = PointEncoder(
            in_channels=4,
            out_channels=bev_channels,
            encoder_type=point_encoder_type,
            voxel_size=(0.05, 0.05, 0.05),
            point_cloud_range=point_cloud_range,
            grid_size=bev_resolution,
            hidden_channels=64,
        )

        self.text_encoder = TextEncoder(
            encoder_type=text_encoder_type,
            output_dim=bev_channels,
            pretrained=pretrained,
            freeze=freeze_text_encoder,
            model_path="./models/clip_random",
        )

        self.category_embedder = CategoryEmbedder(
            class_names=class_names,
            text_encoder=self.text_encoder,
            template="a {} in a driving scenario",
        )

        self.bev_fusion = BevFusion(
            in_channels=bev_channels,
            out_channels=bev_channels,
            fusion_type=fusion_type,
            num_heads=8,
            dropout=0.1,
        )

        self.bev_text_interaction = BevTextInteraction(
            bev_channels=bev_channels,
            text_channels=bev_channels,
            num_heads=8,
            dropout=0.1,
            use_bidirectional=True,
        )

        self.seg_head = nn.Sequential(
            nn.Conv2d(bev_channels, bev_channels, 3, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_channels, bev_channels, 3, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_channels, num_classes, 1),
        )

        self.contrastive_loss = None
        if use_contrastive:
            self.contrastive_loss = BEVTextCLIPLoss(
                num_classes=num_classes,
                contrastive_weight=contrastive_weight,
            )

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        point_cloud: torch.Tensor,
        point_cloud_lengths: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """

        前向传播

        Args:
            images: 多视角图像 [B, N, 3, H, W]
            intrinsics: 相机内参 [B, N, 3, 3]
            extrinsics: 相机外参 [B, N, 4, 4]
            point_cloud: 点云 [B, N, 4] 或 [B, N]
            point_cloud_lengths: 点云长度 [B]
            labels: 分割标签 [B, H, W]

        Returns:
            output_dict: 输出字典

        """
        output_dict = {}

        image_bev = self.image_encoder(images, intrinsics, extrinsics)

        point_bev = self.point_encoder(point_cloud, point_cloud_lengths)

        fused_bev = self.bev_fusion(image_bev, point_bev)

        class_embeddings = self.category_embedder()
        class_embeddings = class_embeddings.detach()
        
        interaction_output = self.bev_text_interaction(fused_bev, class_embeddings)
        enhanced_bev = interaction_output['enhanced_bev']
        
        segmentation_logits = self.seg_head(enhanced_bev)
        
        output_dict['segmentation_logits'] = segmentation_logits
        output_dict['class_embeddings'] = class_embeddings
        output_dict['enhanced_bev'] = enhanced_bev
        
        if self.use_contrastive:
            image_global = image_bev.mean(dim=[2, 3])
            point_global = point_bev.mean(dim=[2, 3])
            bev_global = enhanced_bev.mean(dim=[2, 3])
            
            text_global = class_embeddings.unsqueeze(0).expand(image_global.size(0), -1, -1).mean(dim=1)
            
            output_dict['image_global'] = image_global
            output_dict['point_global'] = point_global
            output_dict['bev_global'] = bev_global
            output_dict['text_global'] = text_global
            
            output_dict['image_features'] = image_bev
            output_dict['point_features'] = point_bev
            output_dict['bev_features'] = enhanced_bev

        if labels is not None and self.contrastive_loss is not None:
            loss_dict = self.contrastive_loss(output_dict, {'labels': labels})
            output_dict['loss'] = loss_dict['total_loss']
            output_dict['loss_dict'] = loss_dict

        return output_dict

    def predict(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        point_cloud: torch.Tensor,
        point_cloud_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """

        预测模式

        Args:
            images: 多视角图像 [B, N, 3, H, W]
            intrinsics: 相机内参 [B, N, 3, 3]
            extrinsics: 相机外参 [B, N, 4, 4]
            point_cloud: 点云 [B, N, 4]
            point_cloud_lengths: 点云长度 [B]

        Returns:
            prediction_dict: 预测字典

        """
        self.eval()
        with torch.no_grad():
            output_dict = self.forward(
                images, intrinsics, extrinsics,
                point_cloud, point_cloud_lengths
            )

        segmentation_logits = output_dict['segmentation_logits']
        segmentation_probs = F.softmax(segmentation_logits, dim=1)
        predictions = segmentation_logits.argmax(dim=1)

        return {
            'predictions': predictions,
            'probabilities': segmentation_probs,
            'class_embeddings': output_dict.get('class_embeddings') if output_dict.get('class_embeddings') is not None else torch.tensor([]),
        }


class BEVTextCLIPLightning(nn.Module):
    """

    BEV-TextCLIP PyTorch Lightning 包装

    方便使用 PyTorch Lightning 训练

    """

    def __init__(self, model: BEVTextCLIP, config):
        """

        初始化 Lightning 包装

        Args:
            model: BEVTextCLIP 模型
            config: 配置对象

        """
        super().__init__()

        self.model = model
        self.config = config

        self.contrastive_weight = config.contrastive_weight

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """

        前向传播

        Args:
            batch: 批次数据

        Returns:
            output_dict: 输出字典

        """
        return self.model(
            images=batch['images'],
            intrinsics=batch['intrinsics'],
            extrinsics=batch['extrinsics'],
            point_cloud=batch['point_cloud'],
            point_cloud_lengths=batch.get('point_cloud_lengths'),
            labels=batch.get('labels'),
        )

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Dict[str, Any]:
        """

        训练步骤

        """
        output = self.forward(batch)

        if 'loss' in output:
            return {
                'loss': output['loss'],
                'log': output.get('loss_dict', {}),
            }

        return {'loss': output['segmentation_logits'].sum()}

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Dict[str, Any]:
        """

        验证步骤

        """
        output = self.forward(batch)

        predictions = output['segmentation_logits'].argmax(dim=1)
        labels = batch['labels']

        accuracy = (predictions == labels).float().mean()

        return {
            'val_loss': output.get('loss', 0),
            'val_accuracy': accuracy,
        }


def create_bev_textclip_model(config) -> BEVTextCLIP:
    """

    创建 BEV-TextCLIP 模型

    Args:
        config: BEVTextCLIPConfig 配置对象

    Returns:
        model: BEVTextCLIP 模型实例

    """
    model = BEVTextCLIP(
        num_classes=config.num_classes,
        class_names=config.class_names,
        image_encoder_type=config.image_encoder_type,
        point_encoder_type=config.point_encoder_type,
        text_encoder_type=config.text_encoder_type,
        fusion_type=config.fusion_type,
        bev_resolution=config.bev_resolution,
        bev_channels=config.bev_channels,
        point_cloud_range=config.point_cloud_range,
        use_contrastive=config.use_contrastive,
        contrastive_weight=config.contrastive_weight,
        pretrained=config.image_pretrained,
        freeze_image_encoder=config.image_freeze,
        freeze_point_encoder=True,
        freeze_text_encoder=config.text_freeze,
    )

    return model
