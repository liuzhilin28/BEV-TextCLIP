#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: BEV-TextCLIP 点云编码器模块，支持 PointPillar/VoxelNet
日期: 2026年1月22日
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import numpy as np


class Voxelization(nn.Module):
    """Voxelization module for point cloud"""
    
    def __init__(
        self,
        voxel_size: Tuple[float, float, float] = (0.05, 0.05, 0.05),
        point_cloud_range: Tuple[float, float, float, float, float, float] = (
            -20.0, -20.0, -2.0, 20.0, 20.0, 6.0
        ),
        max_points_per_voxel: int = 32,
        max_voxels: int = 60000,
    ):
        """Initialize voxelization
        
        Args:
            voxel_size: Voxel size [dx, dy, dz]
            point_cloud_range: Point cloud range
            max_points_per_voxel: Maximum points per voxel
            max_voxels: Maximum number of voxels
        """
        super().__init__()
        
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels
        
        self.x_min, self.y_min, self.z_min = point_cloud_range[:3]
        self.x_max, self.y_max, self.z_max = point_cloud_range[3:]
        
        self.grid_size = [
            int((self.x_max - self.x_min) / voxel_size[0]),
            int((self.y_max - self.y_min) / voxel_size[1]),
            int((self.z_max - self.z_min) / voxel_size[2]),
        ]
    
    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Voxelize point cloud
        
        Args:
            points: Point cloud [N, 4] (x, y, z, intensity)
            
        Returns:
            voxels: Voxel features [V, C, max_points]
            indices: Voxel indices [V, 3]
            num_points: Number of points per voxel [V]
        """
        points_np = points.cpu().numpy()
        
        voxel_indices, num_points_per_voxel = self._points_to_voxel_indices(
            points_np,
            self.grid_size,
            self.point_cloud_range,
            self.voxel_size,
            self.max_points_per_voxel,
            self.max_voxels,
        )
        
        valid_mask = num_points_per_voxel > 0
        
        voxel_indices = voxel_indices[valid_mask]
        num_points_per_voxel = num_points_per_voxel[valid_mask]
        
        V = len(voxel_indices)
        if V == 0:
            voxels = torch.zeros((1, 4, self.max_points_per_voxel), device=points.device)
            indices = torch.zeros((1, 3), device=points.device, dtype=torch.long)
            num_points = torch.ones(1, device=points.device, dtype=torch.long)
            return voxels, indices, num_points
        
        max_c = points.shape[1]
        voxels = torch.zeros((V, max_c, self.max_points_per_voxel), device=points.device)
        
        for i, (v_idx, num_pts) in enumerate(zip(voxel_indices, num_points_per_voxel)):
            point_mask = (
                ((points_np[:, 0] - self.x_min) / self.voxel_size[0] == v_idx[0]) &
                ((points_np[:, 1] - self.y_min) / self.voxel_size[1] == v_idx[1]) &
                ((points_np[:, 2] - self.z_min) / self.voxel_size[2] == v_idx[2])
            )
            point_indices = np.where(point_mask)[0][:self.max_points_per_voxel]
            voxels[i, :, :len(point_indices)] = torch.from_numpy(
                points_np[point_indices].T
            ).to(voxels.device)
        
        indices = torch.from_numpy(voxel_indices).to(points.device).long()
        num_points_tensor = torch.from_numpy(num_points_per_voxel).to(points.device).long()
        
        return voxels, indices, num_points_tensor
    
    def _points_to_voxel_indices(
        self,
        points: np.ndarray,
        grid_size: List[int],
        point_cloud_range: Tuple[float, float, float, float, float, float],
        voxel_size: Tuple[float, float, float],
        max_points_per_voxel: int,
        max_voxels: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert points to voxel indices"""
        x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
        dx, dy, dz = voxel_size
        
        points_x = points[:, 0]
        points_y = points[:, 1]
        points_z = points[:, 2]
        
        voxel_x = ((points_x - x_min) / dx).astype(np.int32)
        voxel_y = ((points_y - y_min) / dy).astype(np.int32)
        voxel_z = ((points_z - z_min) / dz).astype(np.int32)
        
        valid_mask = (
            (voxel_x >= 0) & (voxel_x < grid_size[0]) &
            (voxel_y >= 0) & (voxel_y < grid_size[1]) &
            (voxel_z >= 0) & (voxel_z < grid_size[2])
        )
        
        points = points[valid_mask]
        voxel_x = voxel_x[valid_mask]
        voxel_y = voxel_y[valid_mask]
        voxel_z = voxel_z[valid_mask]
        
        if len(points) == 0:
            return np.zeros((0, 3), dtype=np.int32), np.zeros(0, dtype=np.int32)
        
        grid_index = voxel_x + voxel_y * grid_size[0] + voxel_z * grid_size[0] * grid_size[1]
        
        sort_idx = np.argsort(grid_index)
        sorted_grid_index = grid_index[sort_idx]
        sorted_points = points[sort_idx]
        sorted_voxel_x = voxel_x[sort_idx]
        sorted_voxel_y = voxel_y[sort_idx]
        sorted_voxel_z = voxel_z[sort_idx]
        
        unique_grid_index, counts = np.unique(sorted_grid_index, return_counts=True)
        
        num_voxels = min(len(unique_grid_index), max_voxels)
        
        voxel_indices = np.zeros((num_voxels, 3), dtype=np.int32)
        num_points_per_voxel = np.zeros(num_voxels, dtype=np.int32)
        
        for i in range(num_voxels):
            grid_val = unique_grid_index[i]
            mask = sorted_grid_index == grid_val
            
            voxel_indices[i, 0] = sorted_voxel_x[mask][0]
            voxel_indices[i, 1] = sorted_voxel_y[mask][0]
            voxel_indices[i, 2] = sorted_voxel_z[mask][0]
            
            num_points = min(counts[i], max_points_per_voxel)
            num_points_per_voxel[i] = num_points
        
        return voxel_indices, num_points_per_voxel


class SparseConv3d(nn.Module):
    """Simplified 3D convolution for voxels"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        """Initialize 3D convolution
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            x: Input [B, C, D, H, W]
            
        Returns:
            out: Output [B, C', D, H, W]
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PseudoSparseConv3d(nn.Module):
    """Pseudo sparse 3D convolution"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_size: Tuple[int, int, int],
        kernel_size: int = 3,
    ):
        """Initialize pseudo sparse convolution
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            grid_size: 3D grid size [D, H, W]
            kernel_size: Kernel size
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.dense_conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(
        self,
        voxels: torch.Tensor,
        indices: torch.Tensor,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Forward pass
        
        Args:
            voxels: Voxel features [V, C, max_points]
            indices: Voxel indices [V, 3]
            batch_size: Batch size
            
        Returns:
            out: Output [B, C', D, H, W]
        """
        D, H, W = self.grid_size
        C = self.in_channels
        
        dense_voxel = torch.zeros(
            (batch_size, C, D, H, W),
            device=voxels.device,
            dtype=voxels.dtype,
        )
        
        voxel_features = voxels.mean(dim=2)
        
        for b in range(batch_size):
            batch_mask = indices[:, 0] >= 0
            batch_indices = indices[batch_mask]
            batch_voxels = voxel_features[batch_mask]
            
            for i, (idx, feat) in enumerate(zip(batch_indices, batch_voxels)):
                d, h, w = idx[2], idx[1], idx[0]
                if 0 <= d < D and 0 <= h < H and 0 <= w < W:
                    dense_voxel[b, :, d, h, w] = feat
        
        out = self.dense_conv(dense_voxel)
        out = self.bn(out)
        out = self.relu(out)
        
        return out


class HeightPooling(nn.Module):
    """Height pooling module for 3D to 2D"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_size: Tuple[int, int, int],
    ):
        """Initialize height pooling
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            grid_size: 3D grid size [D, H, W]
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size
        
        self.conv = nn.Conv2d(
            in_channels * grid_size[0],
            out_channels,
            kernel_size=1,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            x: Input [B, C*D, H, W]
            
        Returns:
            out: Output [B, C', H, W]
        """
        x = self.conv(x)
        return x


class VoxelNet(nn.Module):
    """VoxelNet point cloud encoder"""
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 256,
        voxel_size: Tuple[float, float, float] = (0.05, 0.05, 0.05),
        point_cloud_range: Tuple[float, float, float, float, float, float] = (
            -20.0, -20.0, -2.0, 20.0, 20.0, 6.0
        ),
        grid_size: Tuple[int, int, int] = (200, 200, 4),
        hidden_channels: int = 64,
        max_voxels: int = 60000,
        max_points_per_voxel: int = 32,
    ):
        """Initialize VoxelNet
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            voxel_size: Voxel size
            point_cloud_range: Point cloud range
            grid_size: BEV grid size [X, Y, Z]
            hidden_channels: Hidden channels
            max_voxels: Maximum voxels
            max_points_per_voxel: Maximum points per voxel
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size
        self.max_voxels = max_voxels
        self.max_points_per_voxel = max_points_per_voxel
        
        self.voxelization = Voxelization(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_points_per_voxel=max_points_per_voxel,
            max_voxels=max_voxels,
        )
        
        grid_D, grid_H, grid_W = grid_size
        
        self.pseudo_sparse_conv = PseudoSparseConv3d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            grid_size=grid_size,
            kernel_size=3,
        )
        
        self.conv3d_1 = SparseConv3d(hidden_channels, hidden_channels, kernel_size=3)
        self.conv3d_2 = SparseConv3d(hidden_channels, hidden_channels, kernel_size=3)
        
        self.height_pooling = HeightPooling(
            in_channels=hidden_channels,
            out_channels=out_channels,
            grid_size=grid_size,
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
        point_cloud: torch.Tensor,
        point_cloud_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass
        
        Args:
            point_cloud: Point cloud [B, N, 4] or [N, 4]
            point_cloud_lengths: Point cloud lengths [B]
            
        Returns:
            bev_features: BEV features [B, C, X, Y]
        """
        if point_cloud.dim() == 2:
            point_cloud = point_cloud.unsqueeze(0)
        
        B, N, C = point_cloud.shape
        
        all_bev_features = []
        
        for b in range(B):
            if point_cloud_lengths is not None:
                num_points = point_cloud_lengths[b].item()
                points = point_cloud[b, :num_points]
            else:
                points = point_cloud[b]
            
            if points.shape[0] == 0:
                bev = torch.zeros(
                    (1, self.out_channels, self.grid_size[0], self.grid_size[1]),
                    device=point_cloud.device,
                )
                all_bev_features.append(bev)
                continue
            
            voxels, indices, num_points_per_voxel = self.voxelization(points)
            
            x_3d = self.pseudo_sparse_conv(voxels, indices, batch_size=1)
            
            x_3d = self.conv3d_1(x_3d)
            x_3d = self.conv3d_2(x_3d)
            
            D, H, W = self.grid_size
            x_3d = x_3d.view(1, -1, H, W)
            
            bev = self.height_pooling(x_3d)
            bev = self.bev_encoder(bev)
            
            all_bev_features.append(bev)
        
        bev_features = torch.stack(all_bev_features, dim=0).squeeze(1)
        
        return bev_features


class PointPillarScatter(nn.Module):
    """PointPillar scatter layer"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_size: Tuple[int, int],
    ):
        """Initialize scatter layer
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            grid_size: BEV grid size [X, Y]
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(
        self,
        pillar_features: torch.Tensor,
        voxel_indices: torch.Tensor,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Forward pass
        
        Args:
            pillar_features: Pillar features [V, C]
            voxel_indices: Voxel indices [V, 2]
            batch_size: Batch size
            
        Returns:
            bev_features: BEV features [B, C', X, Y]
        """
        X, Y = self.grid_size
        
        bev_features = torch.zeros(
            (batch_size, self.in_channels, X, Y),
            device=pillar_features.device,
            dtype=pillar_features.dtype,
        )
        
        for b in range(batch_size):
            batch_mask = voxel_indices[:, 0] >= 0
            batch_indices = voxel_indices[batch_mask]
            batch_features = pillar_features[batch_mask]
            
            for i, (x, y) in enumerate(batch_indices):
                x_int = int(x.item())
                y_int = int(y.item())
                if 0 <= x_int < X and 0 <= y_int < Y:
                    bev_features[b, :, x_int, y_int] = batch_features[i]
        
        bev_features = self.conv(bev_features)
        
        return bev_features
        
        bev_features = self.conv(bev_features)
        
        return bev_features


class PointPillarEncoder(nn.Module):
    """PointPillar point cloud encoder"""
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 256,
        voxel_size: Tuple[float, float, float] = (0.05, 0.05, 0.05),
        point_cloud_range: Tuple[float, float, float, float, float, float] = (
            -20.0, -20.0, -2.0, 20.0, 20.0, 6.0
        ),
        grid_size: Tuple[int, int] = (200, 200),
        hidden_channels: List[int] = [64, 64],
        max_voxels: int = 60000,
    ):
        """Initialize PointPillar encoder
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            voxel_size: Voxel size
            point_cloud_range: Point cloud range
            grid_size: BEV grid size [X, Y]
            hidden_channels: Hidden channels list
            max_voxels: Maximum voxels
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size
        self.max_voxels = max_voxels
        
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        
        self.x_min, self.y_min, self.z_min = point_cloud_range[:3]
        self.x_max, self.y_max, self.z_max = point_cloud_range[3:]
        
        self.grid_D = 1
        
        self.pillar_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels[0], hidden_channels[1]),
            nn.ReLU(inplace=True),
        )
        
        self.scatter = PointPillarScatter(
            in_channels=hidden_channels[-1],
            out_channels=out_channels,
            grid_size=grid_size,
        )
        
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def _points_to_pillars(
        self,
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert points to pillars
        
        Args:
            points: Point cloud [N, 4]
            
        Returns:
            pillars: Pillar features [V, max_points, C]
            indices: Pillar indices [V, 2]
            num_points: Number of points per pillar [V]
        """
        dx = (self.x_max - self.x_min) / self.grid_size[0]
        dy = (self.y_max - self.y_min) / self.grid_size[1]
        
        pillar_x = ((points[:, 0] - self.x_min) / dx).astype(np.int32)
        pillar_y = ((points[:, 1] - self.y_min) / dy).astype(np.int32)
        
        valid_mask = (
            (pillar_x >= 0) & (pillar_x < self.grid_size[0]) &
            (pillar_y >= 0) & (pillar_y < self.grid_size[1])
        )
        
        points = points[valid_mask]
        pillar_x = pillar_x[valid_mask]
        pillar_y = pillar_y[valid_mask]
        
        if len(points) == 0:
            return np.zeros((1, 1, self.in_channels)), np.zeros((1, 2), dtype=np.int32), np.ones(1, dtype=np.int32)
        
        pillar_grid_index = pillar_x + pillar_y * self.grid_size[0]
        
        sort_idx = np.argsort(pillar_grid_index)
        sorted_points = points[sort_idx]
        sorted_pillar_x = pillar_x[sort_idx]
        sorted_pillar_y = pillar_y[sort_idx]
        sorted_grid_index = pillar_grid_index[sort_idx]
        
        unique_indices, counts = np.unique(sorted_grid_index, return_counts=True)
        
        num_pillars = min(len(unique_indices), self.max_voxels)
        
        pillars = np.zeros((num_pillars, 32, self.in_channels), dtype=np.float32)
        indices = np.zeros((num_pillars, 2), dtype=np.int32)
        num_points = np.zeros(num_pillars, dtype=np.int32)
        
        for i in range(num_pillars):
            grid_val = unique_indices[i]
            mask = sorted_grid_index == grid_val
            
            pillar_points = sorted_points[mask]
            num_pts = min(len(pillar_points), 32)
            
            pillars[i, :num_pts] = pillar_points[:num_pts]
            indices[i, 0] = sorted_pillar_x[mask][0]
            indices[i, 1] = sorted_pillar_y[mask][0]
            num_points[i] = num_pts
        
        return pillars, indices, num_points
    
    def forward(
        self,
        point_cloud: torch.Tensor,
        point_cloud_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass
        
        Args:
            point_cloud: Point cloud [B, N, 4]
            point_cloud_lengths: Point cloud lengths [B]
            
        Returns:
            bev_features: BEV features [B, C, X, Y]
        """
        if point_cloud.dim() == 2:
            point_cloud = point_cloud.unsqueeze(0)
        
        B, N, C = point_cloud.shape
        
        all_bev_features = []
        
        for b in range(B):
            if point_cloud_lengths is not None:
                num_points = point_cloud_lengths[b].item()
                points = point_cloud[b, :num_points].cpu().numpy()
            else:
                points = point_cloud[b].cpu().numpy()
            
            pillars, indices, num_points = self._points_to_pillars(points)
            
            pillars_tensor = torch.from_numpy(pillars).to(point_cloud.device)
            num_points_tensor = torch.from_numpy(num_points).to(point_cloud.device)
            
            max_pts = pillars_tensor.shape[1]
            masked_features = pillars_tensor * (num_points_tensor > 0).unsqueeze(-1).unsqueeze(-1)
            
            pillar_features = self.pillar_encoder(masked_features)
            
            max_pts_per_pillar = pillar_features.size(1)
            pooled_features = pillar_features.mean(dim=1)
            
            bev = self.scatter(pooled_features, torch.from_numpy(indices).to(point_cloud.device), batch_size=1)
            bev = self.bev_encoder(bev)
            
            all_bev_features.append(bev)
        
        if len(all_bev_features) == 1:
            return all_bev_features[0]
        
        bev_features = torch.cat(all_bev_features, dim=0)
        
        return bev_features


class PointEncoder(nn.Module):
    """Point cloud encoder wrapper"""
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 256,
        encoder_type: str = "pointpillar",
        voxel_size: Tuple[float, float, float] = (0.05, 0.05, 0.05),
        point_cloud_range: Tuple[float, float, float, float, float, float] = (
            -20.0, -20.0, -2.0, 20.0, 20.0, 6.0
        ),
        grid_size: Tuple[int, int] = (200, 200),
        hidden_channels: int = 64,
    ):
        """Initialize point encoder
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            encoder_type: Encoder type (pointpillar or voxelnet)
            voxel_size: Voxel size
            point_cloud_range: Point cloud range
            grid_size: BEV grid size [X, Y]
            hidden_channels: Hidden channels
        """
        super().__init__()
        
        if encoder_type == "voxelnet":
            self.encoder = VoxelNet(
                in_channels=in_channels,
                out_channels=out_channels,
                voxel_size=voxel_size,
                point_cloud_range=point_cloud_range,
                grid_size=(grid_size[0], grid_size[1], 4),
                hidden_channels=hidden_channels,
            )
        elif encoder_type == "pointpillar":
            self.encoder = PointPillarEncoder(
                in_channels=in_channels,
                out_channels=out_channels,
                voxel_size=voxel_size,
                point_cloud_range=point_cloud_range,
                grid_size=grid_size,
                hidden_channels=[hidden_channels, hidden_channels],
            )
        else:
            raise ValueError(f"Unknown point encoder type: {encoder_type}")
    
    def forward(
        self,
        point_cloud: torch.Tensor,
        point_cloud_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass
        
        Args:
            point_cloud: Point cloud [B, N, 4] or [N, 4]
            point_cloud_lengths: Point cloud lengths [B]
            
        Returns:
            bev_features: BEV features [B, C, X, Y]
        """
        return self.encoder(point_cloud, point_cloud_lengths)
