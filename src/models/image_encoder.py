#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: BEV-TextCLIP 图像编码器模块，支持 ResNet/ViT 和 LSS
日期: 2026年1月22日
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
from collections import OrderedDict


class ResNetEncoder(nn.Module):
    """ResNet image encoder backbone"""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 512,
        pretrained: bool = True,
        freeze: bool = True,
    ):
        """Initialize ResNet encoder
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            pretrained: Use pretrained weights
            freeze: Freeze parameters
        """
        super().__init__()
        
        from torchvision.models import resnet50, ResNet50_Weights
        
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        
        self.stages = nn.ModuleDict()
        self.stages['layer0'] = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.stages['layer1'] = backbone.layer1
        self.stages['layer2'] = backbone.layer2
        self.stages['layer3'] = backbone.layer3
        self.stages['layer4'] = backbone.layer4
        
        if out_channels != 2048:
            self.adjust_channel = nn.Conv2d(2048, out_channels, kernel_size=1)
        else:
            self.adjust_channel = nn.Identity()
        
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            features: Feature dictionary
        """
        features = {}
        
        x = self.stages['layer0'](x)
        features['layer0'] = x
        
        x = self.stages['layer1'](x)
        features['layer1'] = x
        
        x = self.stages['layer2'](x)
        features['layer2'] = x
        
        x = self.stages['layer3'](x)
        features['layer3'] = x
        
        x = self.stages['layer4'](x)
        features['layer4'] = self.adjust_channel(x)
        
        return features
    
    def get_output_shape(self, in_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Get output feature map shape
        
        Args:
            in_shape: Input image shape [H, W]
            
        Returns:
            out_shape: Output shape [H', W']
        """
        h, w = in_shape
        h //= 32
        w //= 32
        return (h, w)


class ViTEncoder(nn.Module):
    """Vision Transformer encoder backbone"""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 512,
        img_size: int = 224,
        patch_size: int = 16,
        pretrained: bool = True,
        freeze: bool = True,
    ):
        """Initialize ViT encoder
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            img_size: Input image size
            patch_size: Patch size
            pretrained: Use pretrained weights
            freeze: Freeze parameters
        """
        super().__init__()
        
        try:
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
            self.use_torchvision = True
        except ImportError:
            from timm.models import vit_base_patch16_224
            backbone = vit_base_patch16_224(pretrained=pretrained)
            self.use_torchvision = False
        
        self.patch_size = patch_size
        self.embed_dim = backbone.embed_dim
        
        if hasattr(backbone, 'patch_embed'):
            self.patch_embed = backbone.patch_embed
        else:
            self.patch_embed = nn.Conv2d(
                in_channels, self.embed_dim, kernel_size=patch_size, stride=patch_size
            )
        
        self.pos_embed = backbone.pos_embed if hasattr(backbone, 'pos_embed') else None
        self.blocks = backbone.blocks if hasattr(backbone, 'blocks') else nn.ModuleList()
        self.norm = backbone.norm if hasattr(backbone, 'norm') else nn.LayerNorm(self.embed_dim)
        
        if out_channels != self.embed_dim:
            self.adjust_channel = nn.Linear(self.embed_dim, out_channels)
        else:
            self.adjust_channel = nn.Identity()
        
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            features: Feature dictionary
        """
        B, C, H, W = x.shape
        
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, :x.size(1)]
        
        x = x.flatten(2).transpose(1, 2)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = self.adjust_channel(x)
        
        return {
            'patch_embed': x,
            'spatial_shape': (H, W),
        }
    
    def get_output_shape(self, in_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Get output feature map shape
        
        Args:
            in_shape: Input image shape [H, W]
            
        Returns:
            out_shape: Output shape [H', W']
        """
        h = in_shape[0] // self.patch_size
        w = in_shape[1] // self.patch_size
        return (h, w)


class DepthNet(nn.Module):
    """Depth estimation network for LSS"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        depth_bins: int = 4,
        hidden_channels: int = 96,
    ):
        """Initialize depth estimation network
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            depth_bins: Number of depth bins
            hidden_channels: Hidden channels
        """
        super().__init__()
        
        self.depth_bins = depth_bins
        self.out_channels = out_channels
        
        self.depth_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, depth_bins, 1),
        )
        
        self.context_net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass
        
        Args:
            x: Input features [B, C, H, W]
            
        Returns:
            depth_prob: Depth probability [B, D, H, W]
            context: Context features [B, C, H, W]
        """
        depth_prob = self.depth_net(x)
        depth_prob = F.softmax(depth_prob, dim=1)
        
        context = self.context_net(x)
        
        return depth_prob, context


class VoxelPooling(nn.Module):
    """Voxel pooling module for LSS"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_size: Tuple[int, int, int],
        voxel_size: Tuple[float, float, float],
        point_cloud_range: Tuple[float, float, float],
    ):
        """Initialize voxel pooling
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            grid_size: BEV grid size [X, Y, Z]
            voxel_size: Voxel size [dx, dy, dz]
            point_cloud_range: Point cloud range
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        
        self.x_min, self.y_min, self.z_min = point_cloud_range[:3]
        self.x_max, self.y_max, self.z_max = point_cloud_range[3:]
        
        self.dx, self.dy, self.dz = voxel_size
        
        self.frustum_size = grid_size[2]
    
    def create_frustum_grid(
        self,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create frustum grid
        
        Args:
            intrinsics: Camera intrinsics [B, N, 3, 3]
            extrinsics: Camera extrinsics [B, N, 4, 4]
            image_shape: Image shape [H, W]
            
        Returns:
            frustum_coords: Frustum coordinates [B, N, D, H, W, 3]
            depth_bins: Depth bins [D]
        """
        B, N = intrinsics.shape[:2]
        D = self.frustum_size
        H, W = image_shape
        
        device = intrinsics.device
        
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        pixel_coords = torch.stack([x_coords, y_coords, torch.ones_like(x_coords)], dim=-1)
        
        intrinsics_inv = torch.inverse(intrinsics.view(-1, 3, 3)).view(B, N, 3, 3)
        
        pixel_coords = pixel_coords.view(1, 1, H, W, 3).expand(B, N, -1, -1, -1)
        
        camera_coords = torch.einsum('bnhwc,bncd->bnhdw', pixel_coords, intrinsics_inv)
        camera_coords = camera_coords.permute(0, 1, 2, 4, 3)
        
        depth_bins = torch.linspace(self.z_min, self.z_max, D, device=device)
        
        frustum_coords = camera_coords.unsqueeze(2) * depth_bins.view(1, 1, D, 1, 1, 1)
        
        extrinsics_inv = torch.inverse(extrinsics.view(-1, 4, 4)).view(B, N, 4, 4)
        ones = torch.ones_like(frustum_coords[..., :1])
        homogeneous_coords = torch.cat([frustum_coords, ones], dim=-1)
        
        world_coords_list = []
        for b in range(B):
            for n in range(N):
                homo = homogeneous_coords[b, n]
                world = torch.matmul(homo, extrinsics_inv[b, n])
                world_coords_list.append(world)
        world_coords = torch.stack(world_coords_list, dim=0)
        world_coords = world_coords.view(B, N, D, H, W, 4)
        
        world_coords = world_coords[..., :3]
        
        return world_coords, depth_bins
    
    def voxel_pooling(
        self,
        bev_indices: torch.Tensor,
        bev_weights: torch.Tensor,
        out_channels: int,
    ) -> torch.Tensor:
        """Voxel pooling operation
        
        Args:
            bev_indices: BEV grid indices [B, N, D, H, W, 3]
            bev_weights: BEV weights [B, N, D, H, W, C]
            out_channels: Output channels
            
        Returns:
            bev_features: BEV features [B, C, X, Y]
        """
        B, N, D, H, W, C = bev_weights.shape
        X, Y, Z = self.grid_size
        
        bev_features = torch.zeros(B, out_channels, X, Y, device=bev_weights.device)
        
        bev_indices_flat = bev_indices.reshape(B, -1, 3)
        bev_weights_flat = bev_weights.reshape(B, -1, C)
        
        valid_mask = (
            (bev_indices_flat[..., 0] >= 0) &
            (bev_indices_flat[..., 0] < X) &
            (bev_indices_flat[..., 1] >= 0) &
            (bev_indices_flat[..., 1] < Y) &
            (bev_indices_flat[..., 2] >= 0) &
            (bev_indices_flat[..., 2] < Z)
        )
        
        for b in range(B):
            indices_b = bev_indices_flat[b]
            weights_b = bev_weights_flat[b]
            valid_b = valid_mask[b]
            
            indices_valid = indices_b[valid_b]
            weights_valid = weights_b[valid_b]
            
            x_coords = indices_valid[:, 0].long()
            y_coords = indices_valid[:, 1].long()
            
            for i in range(len(x_coords)):
                x = x_coords[i].item()
                y = y_coords[i].item()
                if 0 <= x < X and 0 <= y < Y:
                    bev_features[b, :, x, y] += weights_valid[i]
        
        return bev_features
    
    def forward(
        self,
        context: torch.Tensor,
        depth_prob: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Forward pass
        
        Args:
            context: Context features [B, C, H, W]
            depth_prob: Depth probability [B, D, H, W]
            intrinsics: Camera intrinsics [B, N, 3, 3]
            extrinsics: Camera extrinsics [B, N, 4, 4]
            image_shape: Image shape [H, W]
            
        Returns:
            bev_features: BEV features [B, C, X, Y]
        """
        B, C, H, W = context.shape
        D = self.frustum_size
        
        frustum_coords, depth_bins = self.create_frustum_grid(
            intrinsics, extrinsics, image_shape
        )
        
        context_expanded = context.permute(0, 2, 3, 1).unsqueeze(1).unsqueeze(2)
        depth_prob_expanded = depth_prob.unsqueeze(-1)
        
        bev_weights = context_expanded * depth_prob_expanded
        
        bev_indices = frustum_coords.clone()
        bev_indices[..., 0] = (bev_indices[..., 0] - self.x_min) / self.dx
        bev_indices[..., 1] = (bev_indices[..., 1] - self.y_min) / self.dy
        bev_indices = bev_indices.long()
        
        bev_features = self.voxel_pooling(
            bev_indices, bev_weights, C
        )
        
        return bev_features


class LiftSplatTransform(nn.Module):
    """Lift-Splat-Shoot view transformation module"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        grid_size: Tuple[int, int, int] = (200, 200, 4),
        voxel_size: Tuple[float, float, float] = (0.2, 0.2, 1.0),
        point_cloud_range: Tuple[float, float, float, float, float, float] = (
            -20.0, -20.0, -2.0, 20.0, 20.0, 6.0
        ),
        depth_bins: int = 4,
    ):
        """Initialize LSS transform
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            grid_size: BEV grid size [X, Y, Z]
            voxel_size: Voxel size [dx, dy, dz]
            point_cloud_range: Point cloud range
            depth_bins: Number of depth bins
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.depth_bins = depth_bins
        
        self.depth_net = DepthNet(
            in_channels=in_channels,
            out_channels=out_channels,
            depth_bins=depth_bins,
        )
        
        self.voxel_pooling = VoxelPooling(
            in_channels=out_channels,
            out_channels=out_channels,
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
        )
        
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(
        self,
        image_features: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Forward pass
        
        Args:
            image_features: Image features [B, N, C, H, W]
            intrinsics: Camera intrinsics [B, N, 3, 3]
            extrinsics: Camera extrinsics [B, N, 4, 4]
            image_shape: Image shape [H, W]
            
        Returns:
            bev_features: BEV features [B, C, X, Y]
        """
        B, N, C, H, W = image_features.shape
        
        image_flat = image_features.view(B * N, C, H, W)

        depth_prob, context = self.depth_net(image_flat)

        depth_prob = depth_prob.view(B, N, self.depth_bins, H, W)
        context = context.view(B, N, self.out_channels, H, W)
        
        context = context.mean(dim=1)
        
        bev_features = self.voxel_pooling(
            context,
            depth_prob,
            intrinsics,
            extrinsics,
            image_shape,
        )
        
        bev_features = self.bev_encoder(bev_features)
        
        return bev_features


class ImageEncoder(nn.Module):
    """Image encoder wrapper with ResNet/ViT and LSS"""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 256,
        image_encoder_type: str = "resnet50",
        bev_grid_size: Tuple[int, int] = (200, 200),
        bev_depth_bins: int = 4,
        point_cloud_range: Tuple[float, float, float, float, float, float] = (
            -20.0, -20.0, -2.0, 20.0, 20.0, 6.0
        ),
        pretrained: bool = True,
        freeze: bool = True,
    ):
        """Initialize image encoder
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            image_encoder_type: Encoder type (resnet50 or vit)
            bev_grid_size: BEV grid size [X, Y]
            bev_depth_bins: BEV depth bins
            point_cloud_range: Point cloud range
            pretrained: Use pretrained weights
            freeze: Freeze parameters
        """
        super().__init__()
        
        self.out_channels = out_channels
        
        if image_encoder_type == "resnet50":
            self.backbone = ResNetEncoder(
                in_channels=in_channels,
                out_channels=out_channels,
                pretrained=pretrained,
                freeze=freeze,
            )
            backbone_out_channels = out_channels
        elif image_encoder_type == "vit":
            self.backbone = ViTEncoder(
                in_channels=in_channels,
                out_channels=out_channels,
                pretrained=pretrained,
                freeze=freeze,
            )
            backbone_out_channels = out_channels
        else:
            raise ValueError(f"Unknown image encoder type: {image_encoder_type}")
        
        self.lss_transform = LiftSplatTransform(
            in_channels=backbone_out_channels,
            out_channels=out_channels,
            grid_size=(bev_grid_size[0], bev_grid_size[1], bev_depth_bins),
            voxel_size=(
                (point_cloud_range[3] - point_cloud_range[0]) / bev_grid_size[0],
                (point_cloud_range[4] - point_cloud_range[1]) / bev_grid_size[1],
                (point_cloud_range[5] - point_cloud_range[2]) / bev_depth_bins,
            ),
            point_cloud_range=point_cloud_range,
            depth_bins=bev_depth_bins,
        )
    
    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass
        
        Args:
            images: Multi-view images [B, N, 3, H, W]
            intrinsics: Camera intrinsics [B, N, 3, 3]
            extrinsics: Camera extrinsics [B, N, 4, 4]
            
        Returns:
            bev_features: BEV features [B, C, X, Y]
        """
        B, N, C, H, W = images.shape
        
        images_flat = images.view(B * N, C, H, W)
        features = self.backbone(images_flat)
        
        if isinstance(features, dict):
            if 'layer4' in features:
                image_features = features['layer4']
            else:
                image_features = features.get('patch_embed')
                if image_features is not None:
                    B, N, C = image_features.shape
                    spatial_shape = features.get('spatial_shape', (H // 32, W // 32))
                    image_features = image_features.transpose(1, 2).view(B, N, C, *spatial_shape)
        else:
            image_features = features
        
        _, C_feat, H_feat, W_feat = image_features.shape
        image_features = image_features.view(B, N, C_feat, H_feat, W_feat)
        
        image_shape = (H_feat, W_feat)
        
        bev = self.lss_transform(
            image_features,
            intrinsics,
            extrinsics,
            image_shape,
        )
        
        return bev
    
    def get_feature_maps(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get multi-scale feature maps
        
        Args:
            images: Input images [B, 3, H, W]
            
        Returns:
            feature_maps: Multi-scale feature map dictionary
        """
        return self.backbone(images)
