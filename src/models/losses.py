#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: BEV-TextCLIP 损失函数模块，包含 InfoNCE 对比损失、FocalLoss、DiceLoss 等
日期: 2026年1月22日
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import math


class InfoNCE(nn.Module):
    """

    InfoNCE 损失函数

    对比学习的标准损失函数

    """

    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = 'mean',
    ):
        """

        初始化 InfoNCE

        Args:
            temperature: 温度参数
            reduction: 归约方式 ('mean', 'sum', 'none')

        """
        super().__init__()

        self.temperature = temperature
        self.reduction = reduction

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))

    def forward(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """

        前向传播

        Args:
            query: 查询特征 [B, C]
            positive: 正样本特征 [B, C]
            negatives: 负样本特征 [B, N, C]

        Returns:
            loss: InfoNCE 损失

        """
        query = F.normalize(query, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        pos_sim = (query * positive).sum(dim=-1)

        neg_sim = query @ negatives.transpose(-2, -1)
        neg_sim = neg_sim.mean(dim=1)

        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)

        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)

        logits = logits * self.logit_scale.exp()

        loss = F.cross_entropy(logits, labels, reduction=self.reduction)

        return loss


class MultiModalContrastiveLoss(nn.Module):
    """

    多模态对比损失函数

    包含三种对比学习目标:
    1. 全局对比 (Global Contrast)
    2. 局部对比 (Local Contrast)
    3. 跨模态对比 (Cross-Modal Contrast)

    """

    def __init__(
        self,
        temperature: float = 0.07,
        global_weight: float = 1.0,
        local_weight: float = 0.5,
        cross_modal_weight: float = 0.5,
    ):
        """

        初始化多模态对比损失

        Args:
            temperature: 温度参数
            global_weight: 全局对比权重
            local_weight: 局部对比权重
            cross_modal_weight: 跨模态对比权重

        """
        super().__init__()

        self.temperature = temperature
        self.global_weight = global_weight
        self.local_weight = local_weight
        self.cross_modal_weight = cross_modal_weight

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))

    def global_contrast(
        self,
        image_global: torch.Tensor,
        point_global: torch.Tensor,
        text_global: torch.Tensor,
    ) -> torch.Tensor:
        """

        全局对比损失

        Args:
            image_global: 图像全局特征 [B, C]
            point_global: 点云全局特征 [B, C]
            text_global: 文本全局特征 [K, C]

        Returns:
            loss: 全局对比损失

        """
        B, C = image_global.shape
        
        image_global = F.normalize(image_global, dim=-1)
        point_global = F.normalize(point_global, dim=-1)
        
        # text_global 可能是 num_classes x C 或 B x C
        # 如果是类别嵌入 (num_classes x C)，需要扩展到 batch_size
        if text_global.shape[0] == B:
            # 已经匹配
            text_global = F.normalize(text_global, dim=-1)
            logits_img_text = image_global @ text_global.t()
            logits_img_text = logits_img_text * self.logit_scale.exp()
            
            labels_img = torch.arange(B, device=image_global.device)
            loss_img_text = F.cross_entropy(logits_img_text, labels_img)
            
            logits_text_img = logits_img_text.t()
            loss_text_img = F.cross_entropy(logits_text_img, labels_img)
        else:
            # text_global 是类别嵌入 (num_classes x C)
            # 只计算 image -> text 的对比，text -> image 不计算
            text_global = F.normalize(text_global, dim=-1)
            logits_img_text = image_global @ text_global.t()
            logits_img_text = logits_img_text * self.logit_scale.exp()
            
            # 为每个 image 样本分配一个目标文本类别 (使用对角线作为正样本)
            labels_img = torch.arange(B, device=image_global.device) % text_global.shape[0]
            loss_img_text = F.cross_entropy(logits_img_text, labels_img)
            loss_text_img = torch.tensor(0.0, device=image_global.device)
        
        logits_img_point = image_global @ point_global.t()
        logits_img_point = logits_img_point * self.logit_scale.exp()
        labels_img = torch.arange(B, device=image_global.device)
        loss_img_point = F.cross_entropy(logits_img_point, labels_img)

        logits_point_img = logits_img_point.t()
        loss_point_img = F.cross_entropy(logits_point_img, labels_img)

        total_loss = (loss_img_text + loss_text_img + loss_img_point + loss_point_img) / 4

        return total_loss

    def local_contrast(
        self,
        image_features: torch.Tensor,
        point_features: torch.Tensor,
        bev_features: torch.Tensor,
    ) -> torch.Tensor:
        """

        局部对比损失

        Args:
            image_features: 图像特征 [B, C, H, W]
            point_features: 点云特征 [B, C, H, W]
            bev_features: BEV 特征 [B, C, H, W]

        Returns:
            loss: 局部对比损失

        """
        image_flat = F.normalize(image_features, dim=1).flatten(2).transpose(1, 2)
        point_flat = F.normalize(point_features, dim=1).flatten(2).transpose(1, 2)
        bev_flat = F.normalize(bev_features, dim=1).flatten(2).transpose(1, 2)

        B, N, C = image_flat.shape

        pos_sims = []
        neg_sims = []

        for b in range(B):
            img = image_flat[b]
            pt = point_flat[b]
            bev = bev_flat[b]

            pos_img_bev = (img * bev).sum(dim=-1).mean()
            pos_pt_bev = (pt * bev).sum(dim=-1).mean()

            pos_sims.extend([pos_img_bev, pos_pt_bev])

            for i in range(N):
                neg_img = torch.cat([img[:i], img[i+1:]], dim=0)
                neg_pt = torch.cat([pt[:i], pt[i+1:]], dim=0)
                neg_bev = torch.cat([bev[:i], bev[i+1:]], dim=0)

                neg_img_bev = (img[i:i+1] @ neg_bev.transpose(-2, -1)).mean()
                neg_pt_bev = (pt[i:i+1] @ neg_bev.transpose(-2, -1)).mean()

                neg_sims.extend([neg_img_bev, neg_pt_bev])

        pos_loss = -torch.log(torch.stack(pos_sims).sigmoid() + 1e-8).mean()
        neg_loss = torch.stack(neg_sims).sigmoid().mean()

        return pos_loss + 0.1 * neg_loss

    def cross_modal_contrast(
        self,
        image_features: torch.Tensor,
        point_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """

        跨模态对比损失

        Args:
            image_features: 图像特征 [B, C]
            point_features: 点云特征 [B, C]
            text_features: 文本特征 [K, C]

        Returns:
            loss: 跨模态对比损失

        """
        image_features = F.normalize(image_features, dim=-1)
        point_features = F.normalize(point_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logits_image = image_features @ text_features.t()
        logits_image = logits_image * self.logit_scale.exp()

        logits_point = point_features @ text_features.t()
        logits_point = logits_point * self.logit_scale.exp()

        B = image_features.shape[0]
        labels = torch.arange(B, device=image_features.device)

        loss_image = F.cross_entropy(logits_image, labels)
        loss_point = F.cross_entropy(logits_point, labels)

        return (loss_image + loss_point) / 2

    def forward(
        self,
        image_global: torch.Tensor,
        point_global: torch.Tensor,
        text_global: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        point_features: Optional[torch.Tensor] = None,
        bev_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """

        前向传播

        Args:
            image_global: 图像全局特征 [B, C]
            point_global: 点云全局特征 [B, C]
            text_global: 文本全局特征 [K, C]
            image_features: 图像局部特征 [B, C, H, W]
            point_features: 点云局部特征 [B, C, H, W]
            bev_features: BEV 局部特征 [B, C, H, W]

        Returns:
            loss_dict: 损失字典

        """
        loss_dict = {}

        loss_global = self.global_contrast(image_global, point_global, text_global)
        loss_dict['contrast_global'] = loss_global * self.global_weight

        if all(x is not None for x in [image_features, point_features, bev_features]):
            loss_local = self.local_contrast(image_features, point_features, bev_features)
            loss_dict['contrast_local'] = loss_local * self.local_weight

        loss_cross = self.cross_modal_contrast(image_global, point_global, text_global)
        loss_dict['contrast_cross_modal'] = loss_cross * self.cross_modal_weight

        total_loss = sum(loss_dict.values())
        loss_dict['total_contrast_loss'] = total_loss

        return loss_dict


class FocalLoss(nn.Module):
    """

    Focal Loss

    用于类别不平衡的分割任务

    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        """

        初始化 Focal Loss

        Args:
            alpha: 类别权重 [C]
            gamma: 聚焦参数
            reduction: 归约方式

        """
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """

        前向传播

        Args:
            inputs: 预测概率 [B, C, H, W]
            targets: 真实标签 [B, H, W]

        Returns:
            loss: Focal Loss

        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-ce_loss)

        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            focal_weight = alpha[targets] * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class DiceLoss(nn.Module):
    """

    Dice Loss

    用于分割任务的 Dice 系数损失

    """

    def __init__(
        self,
        smooth: float = 1e-6,
        reduction: str = 'mean',
    ):
        """

        初始化 Dice Loss

        Args:
            smooth: 平滑参数
            reduction: 归约方式

        """
        super().__init__()

        self.smooth = smooth
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """

        前向传播

        Args:
            inputs: 预测概率 [B, C, H, W]
            targets: 真实标签 [B, H, W]

        Returns:
            loss: Dice Loss

        """
        B, C, H, W = inputs.shape

        inputs = F.softmax(inputs, dim=1)

        targets_one_hot = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()

        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        loss = 1 - dice

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class SegmentationLoss(nn.Module):
    """

    语义分割损失

    组合多种分割损失

    """

    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,
        use_focal: bool = True,
        use_dice: bool = True,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        """

        初始化分割损失

        Args:
            num_classes: 类别数
            class_weights: 类别权重
            use_focal: 是否使用 Focal Loss
            use_dice: 是否使用 Dice Loss
            focal_weight: Focal Loss 权重
            dice_weight: Dice Loss 权重

        """
        super().__init__()

        self.num_classes = num_classes
        self.use_focal = use_focal
        self.use_dice = use_dice

        if use_focal:
            self.focal_loss = FocalLoss(
                alpha=class_weights,
                gamma=2.0,
                reduction='mean',
            )

        if use_dice:
            self.dice_loss = DiceLoss(
                smooth=1e-6,
                reduction='mean',
            )

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """

        前向传播

        Args:
            inputs: 预测 [B, C, H, W]
            targets: 标签 [B, H, W]

        Returns:
            loss_dict: 损失字典

        """
        loss_dict = {}

        if self.use_focal:
            loss_focal = self.focal_loss(inputs, targets)
            loss_dict['seg_focal'] = loss_focal * self.focal_weight

        if self.use_dice:
            loss_dice = self.dice_loss(inputs, targets)
            loss_dict['seg_dice'] = loss_dice * self.dice_weight

        total_loss = sum(loss_dict.values())
        loss_dict['total_seg_loss'] = total_loss

        return loss_dict


class BEVTextCLIPLoss(nn.Module):
    """

    BEV-TextCLIP 总损失函数

    组合分割损失和对比损失

    """

    def __init__(
        self,
        num_classes: int,
        contrastive_weight: float = 0.5,
        segmentation_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """

        初始化总损失

        Args:
            num_classes: 类别数
            contrastive_weight: 对比损失权重
            segmentation_weight: 分割损失权重
            class_weights: 类别权重

        """
        super().__init__()

        self.contrastive_weight = contrastive_weight
        self.segmentation_weight = segmentation_weight

        self.contrastive_loss = MultiModalContrastiveLoss(
            temperature=0.07,
            global_weight=1.0,
            local_weight=0.5,
            cross_modal_weight=0.5,
        )

        self.segmentation_loss = SegmentationLoss(
            num_classes=num_classes,
            class_weights=class_weights,
            use_focal=True,
            use_dice=True,
        )

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """

        前向传播

        Args:
            outputs: 模型输出
            targets: 目标标签

        Returns:
            loss_dict: 损失字典

        """
        loss_dict = {}

        if self.contrastive_weight > 0:
            # 简化对比损失计算，跳过耗显存的部分
            contrastive_output = {}
            
            # 只计算全局对比损失
            image_global = outputs.get('image_global')
            point_global = outputs.get('point_global')
            text_global = outputs.get('text_global')
            
            if image_global is not None and point_global is not None:
                global_loss = self.contrastive_loss.global_contrast(
                    image_global, point_global, text_global
                )
                contrastive_output['global'] = global_loss
                contrastive_output['local'] = torch.tensor(0.0, device=image_global.device)
                contrastive_output['cross_modal'] = torch.tensor(0.0, device=image_global.device)
                contrastive_output['total_contrast_loss'] = global_loss
            
            for k, v in contrastive_output.items():
                if isinstance(v, torch.Tensor):
                    loss_dict[f'contrast/{k}'] = v * self.contrastive_weight

        if self.segmentation_weight > 0:
            seg_output = self.segmentation_loss(
                inputs=outputs['segmentation_logits'],
                targets=targets['labels'],
            )
            for k, v in seg_output.items():
                loss_dict[f'seg/{k}'] = v * self.segmentation_weight

        total_loss = sum(loss_dict.values())
        loss_dict['total_loss'] = total_loss

        return loss_dict
