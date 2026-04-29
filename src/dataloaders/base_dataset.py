#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: BEV-TextCLIP 数据集基类和具体数据集实现
日期: 2026年1月22日
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
import random
import os
import json


class BEVBaseDataset(Dataset, ABC):
    """

    BEV-TextCLIP 数据集基类

    Attributes:
        config: BEVTextCLIPConfig 配置对象
        class_names: 类别名称列表
        data_list: 数据列表
        transform: 数据增强变换

    """

    def __init__(
        self,
        config,
        data_root: str = "./data",
        split: str = "train",
        transform: Optional[Any] = None,
    ):
        """

        初始化数据集基类

        Args:
            config: BEVTextCLIPConfig 配置对象
            data_root: 数据根目录
            split: 数据集划分 ('train', 'val', 'test')
            transform: 数据增强变换

        """
        self.config = config
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.class_names = config.class_names
        self.num_classes = config.num_classes
        self.ignore_index = -100

        self.data_list = self._load_data_list()

    def _get_bev_resolution(self) -> Tuple[int, int]:
        """Return the configured BEV label resolution."""
        bev_resolution = getattr(self.config, 'bev_resolution', (200, 200))
        return int(bev_resolution[0]), int(bev_resolution[1])

    def _get_default_label_index(self) -> int:
        """Use the catch-all class for empty cells when available."""
        if hasattr(self, 'class_names') and 'other' in self.class_names:
            return self.class_names.index('other')
        return max(self.num_classes - 1, 0)

    def _create_empty_bev_labels(self) -> np.ndarray:
        """Create an empty BEV label canvas using ignore_index for unknown cells."""
        bev_h, bev_w = self._get_bev_resolution()
        return np.full((bev_h, bev_w), self.ignore_index, dtype=np.int64)

    def _rasterize_point_labels_to_bev(
        self,
        point_cloud: np.ndarray,
        point_labels: np.ndarray,
    ) -> np.ndarray:
        """Project point-level labels to the configured BEV grid via majority vote."""
        bev_h, bev_w = self._get_bev_resolution()
        bev_labels = self._create_empty_bev_labels()

        if point_cloud.size == 0 or point_labels.size == 0:
            return bev_labels

        point_cloud_range = getattr(
            self.config,
            'point_cloud_range',
            (-20.0, -20.0, -2.0, 20.0, 20.0, 6.0),
        )
        x_min, y_min, _, x_max, y_max, _ = point_cloud_range

        x_coords = point_cloud[:, 0]
        y_coords = point_cloud[:, 1]
        labels = point_labels.astype(np.int64, copy=False)

        valid_mask = (
            np.isfinite(x_coords) &
            np.isfinite(y_coords) &
            (x_coords >= x_min) & (x_coords < x_max) &
            (y_coords >= y_min) & (y_coords < y_max) &
            (labels >= 0) & (labels < self.num_classes)
        )

        if not np.any(valid_mask):
            return bev_labels

        x_valid = x_coords[valid_mask]
        y_valid = y_coords[valid_mask]
        labels_valid = labels[valid_mask]

        x_indices = np.floor((x_valid - x_min) / (x_max - x_min) * bev_h).astype(np.int64)
        y_indices = np.floor((y_valid - y_min) / (y_max - y_min) * bev_w).astype(np.int64)
        x_indices = np.clip(x_indices, 0, bev_h - 1)
        y_indices = np.clip(y_indices, 0, bev_w - 1)

        flat_indices = x_indices * bev_w + y_indices
        vote_table = np.zeros((bev_h * bev_w, self.num_classes), dtype=np.int32)
        np.add.at(vote_table, (flat_indices, labels_valid), 1)

        occupied_mask = vote_table.sum(axis=1) > 0
        bev_flat = bev_labels.reshape(-1)
        bev_flat[occupied_mask] = vote_table[occupied_mask].argmax(axis=1)

        return bev_labels

    def _ensure_bev_labels(
        self,
        point_cloud: np.ndarray,
        labels: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Convert point-level labels to BEV labels when needed."""
        if labels is None:
            return None

        labels_array = np.asarray(labels)
        if labels_array.ndim == 2:
            return labels_array.astype(np.int64, copy=False)

        if labels_array.ndim != 1:
            return labels_array.astype(np.int64, copy=False)

        point_cloud_array = np.asarray(point_cloud)
        if point_cloud_array.ndim != 2 or point_cloud_array.shape[0] != labels_array.shape[0]:
            return self._create_empty_bev_labels()

        return self._rasterize_point_labels_to_bev(point_cloud_array, labels_array)

    @abstractmethod
    def _load_data_list(self) -> List[Dict]:
        """

        加载数据列表 (子类实现)

        Returns:
            data_list: 数据字典列表

        """
        pass

    @abstractmethod
    def _load_point_cloud(self, data_path: str) -> np.ndarray:
        """

        加载点云数据 (子类实现)

        Args:
            data_path: 点云文件路径

        Returns:
            points: 点云数据 [N, 3+?] (x, y, z, ...)

        """
        pass

    @abstractmethod
    def _load_images(self, data_path: str) -> List[np.ndarray]:
        """

        加载多视角图像 (子类实现)

        Args:
            data_path: 图像目录路径

        Returns:
            images: 图像列表 [N_views, H, W, 3]

        """
        pass

    @abstractmethod
    def _load_camera_params(self, data_path: str) -> Dict[str, np.ndarray]:
        """

        加载相机参数 (子类实现)

        Args:
            data_path: 参数文件路径

        Returns:
            params: {
                'intrinsics': [N, 3, 3],
                'extrinsics': [N, 4, 4],
                'image_shape': [H, W]
            }

        """
        pass

    @abstractmethod
    def _load_labels(self, data_path: str) -> np.ndarray:
        """

        加载语义标签 (子类实现)

        Args:
            data_path: 标签文件路径

        Returns:
            labels: 标签数据 [N] 或 [H, W, ...]

        """
        pass

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """

        获取数据项

        Args:
            idx: 数据索引

        Returns:
            data_dict: {
                'point_cloud': [N, 4] (x, y, z, intensity),
                'images': [N_views, 3, H, W],
                'intrinsics': [N_views, 3, 3],
                'extrinsics': [N_views, 4, 4],
                'labels': Optional [N] 或 [H, W],
                'image_shape': [H, W],
                'sample_token': str,
            }

        """
        data_info = self.data_list[idx]
        data_path = data_info['data_path']

        point_cloud = self._load_point_cloud(data_info['point_cloud_path'])
        images = self._load_images(data_info['images_path'])
        camera_params = self._load_camera_params(data_info['camera_path'])
        labels = self._load_labels(data_info['labels_path'])
        labels = self._ensure_bev_labels(point_cloud, labels)

        data_dict = {
            'point_cloud': point_cloud.astype(np.float32),
            'images': images,
            'intrinsics': camera_params['intrinsics'].astype(np.float32),
            'extrinsics': camera_params['extrinsics'].astype(np.float32),
            'labels': labels.astype(np.int64, copy=False) if labels is not None else None,
            'image_shape': camera_params['image_shape'],
            'sample_token': data_info.get('sample_token', str(idx)),
        }

        if self.transform is not None:
            data_dict = self.transform(data_dict)

        return data_dict


class DataCollator:
    """

    数据批处理收集器

    将多个数据项组合成批次

    """

    def __init__(self, config):
        """

        初始化数据收集器

        Args:
            config: BEVTextCLIPConfig 配置对象

        """
        self.config = config
        self.ignore_index = getattr(config, 'ignore_index', -100)
        image_size = getattr(config, 'image_size', (224, 224))
        self.target_image_size = (int(image_size[0]), int(image_size[1]))

    def _resize_images_and_intrinsics(
        self,
        images: np.ndarray,
        intrinsics: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize multi-view images to the training size and scale intrinsics consistently."""
        target_h, target_w = self.target_image_size
        _, _, original_h, original_w, _ = images.shape

        if (original_h, original_w) == (target_h, target_w):
            return images, intrinsics

        image_tensor = torch.from_numpy(images).permute(0, 1, 4, 2, 3).float()
        batch_size, num_cameras = image_tensor.shape[:2]
        image_tensor = image_tensor.view(batch_size * num_cameras, 3, original_h, original_w)
        image_tensor = F.interpolate(
            image_tensor,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False,
        )
        image_tensor = image_tensor.view(batch_size, num_cameras, 3, target_h, target_w)
        resized_images = image_tensor.permute(0, 1, 3, 4, 2).contiguous().numpy()

        scale_x = target_w / float(original_w)
        scale_y = target_h / float(original_h)
        resized_intrinsics = intrinsics.copy()
        resized_intrinsics[..., 0, 0] *= scale_x
        resized_intrinsics[..., 1, 1] *= scale_y
        resized_intrinsics[..., 0, 2] *= scale_x
        resized_intrinsics[..., 1, 2] *= scale_y

        return resized_images, resized_intrinsics

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        """

        收集批次数据

        Args:
            batch: 数据项列表

        Returns:
            batch_dict: 批次数据字典

        """
        batch_size = len(batch)

        point_clouds = [item['point_cloud'] for item in batch]
        images_lists = [item['images'] for item in batch]
        intrinsics = [item['intrinsics'] for item in batch]
        extrinsics = [item['extrinsics'] for item in batch]
        labels = [item['labels'] for item in batch]
        image_shapes = [item['image_shape'] for item in batch]
        sample_tokens = [item['sample_token'] for item in batch]

        max_points = max(pc.shape[0] if hasattr(pc, 'shape') else len(pc) for pc in point_clouds)
        num_cameras = len(images_lists[0])

        padded_point_clouds = []
        point_cloud_lengths = []
        for pc in point_clouds:
            if hasattr(pc, 'shape'):
                pc_shape = pc.shape
            else:
                pc = np.array(pc)
                pc_shape = pc.shape
            
            if pc_shape[0] < max_points:
                pad_size = max_points - pc_shape[0]
                padding = np.zeros((pad_size, pc_shape[1]), dtype=np.float32)
                pc = np.vstack([pc, padding])
            padded_point_clouds.append(pc)
            point_cloud_lengths.append(pc_shape[0])

        all_images = []
        for images_list in images_lists:
            for img in images_list:
                if hasattr(img, 'shape'):
                    all_images.append(img)
                else:
                    all_images.append(np.array(img))
        
        batch_images = np.stack(all_images, axis=0)
        
        num_cameras = len(images_lists[0])
        B = batch_size
        total_images = batch_images.shape[0]
        H = batch_images.shape[1]
        W = batch_images.shape[2]
        C = batch_images.shape[3]
        batch_images = batch_images.reshape(B, num_cameras, H, W, C)

        batch_intrinsics = np.stack(intrinsics, axis=0)
        batch_extrinsics = np.stack(extrinsics, axis=0)
        batch_labels = np.stack(labels, axis=0) if labels[0] is not None else None

        batch_images, batch_intrinsics = self._resize_images_and_intrinsics(
            batch_images,
            batch_intrinsics,
        )
        resized_h, resized_w = batch_images.shape[2], batch_images.shape[3]

        batch_images = np.transpose(batch_images, (0, 1, 4, 2, 3))
        batch_images = batch_images.astype(np.float32) / 255.0
        
        image_shapes_tensor = torch.from_numpy(
            np.repeat(np.array([[resized_h, resized_w]], dtype=np.int64), batch_size, axis=0)
        )

        batch_dict = {
            'point_cloud': torch.from_numpy(np.stack(padded_point_clouds, axis=0)),
            'point_cloud_lengths': torch.tensor(point_cloud_lengths, dtype=torch.long),
            'images': torch.from_numpy(batch_images),
            'intrinsics': torch.from_numpy(batch_intrinsics),
            'extrinsics': torch.from_numpy(batch_extrinsics),
            'labels': torch.from_numpy(batch_labels) if batch_labels is not None else None,
            'label_mask': torch.from_numpy((batch_labels != self.ignore_index).astype(np.bool_)) if batch_labels is not None else None,
            'image_shapes': image_shapes_tensor,
            'sample_tokens': sample_tokens,
        }

        return batch_dict


def create_data_loaders(
    config,
    data_root: str = "./data",
    batch_size: int = 4,
    num_workers: int = 4,
    train_transform: Optional[Any] = None,
    val_transform: Optional[Any] = None,
) -> Tuple[DataLoader, DataLoader]:
    """

    创建训练和验证数据加载器

    Args:
        config: BEVTextCLIPConfig 配置对象
        data_root: 数据根目录
        batch_size: 批次大小
        num_workers: 工作进程数
        train_transform: 训练数据增强
        val_transform: 验证数据增强

    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器

    """
    dataset_name = getattr(config, 'dataset', 'nuscenes')

    dataset_map = {
        'nuscenes': NuScenesDataset,
        'scannet': ScanNetDataset,
        'kitti': KITTIDataset,
        'dummy': DummyDataset,
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_map.keys())}")

    DatasetClass = dataset_map[dataset_name]

    train_dataset = DatasetClass(
        config=config,
        data_root=data_root,
        split="train",
        transform=train_transform,
    )

    val_dataset = DatasetClass(
        config=config,
        data_root=data_root,
        split="val",
        transform=val_transform,
    )

    collator = DataCollator(config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


class NuScenesDataset(BEVBaseDataset):
    """

    nuScenes 数据集实现

    nuScenes: https://www.nuscenes.org/
    室外自动驾驶数据集

    """

    def __init__(
        self,
        config,
        data_root: str = "./data",
        split: str = "train",
        transform: Optional[Any] = None,
    ):
        """

        初始化nuScenes数据集

        Args:
            config: BEVTextCLIPConfig 配置对象
            data_root: 数据根目录
            split: 数据集划分
            transform: 数据增强

        """
        self.requested_version = getattr(config, 'nuscenes_version', 'auto')
        self.version = self._resolve_nuscenes_version(data_root, split, self.requested_version)
        self._nuscenes_label_lookup = None
        super().__init__(config, data_root, split, transform)

    def _resolve_nuscenes_version(
        self,
        data_root: str,
        split: str,
        requested_version: str,
    ) -> str:
        """Resolve the local nuScenes version to use for the current split."""
        nuscenes_root = os.path.join(data_root, 'nuscenes')

        def version_is_available(version: str) -> bool:
            return os.path.exists(os.path.join(nuscenes_root, version, 'attribute.json'))

        if requested_version != 'auto' and version_is_available(requested_version):
            return requested_version

        if split in ('train', 'val'):
            preferred_versions = ['v1.0-trainval', 'v1.0-mini']
        elif split == 'test':
            preferred_versions = ['v1.0-test', 'v1.0-mini', 'v1.0-trainval']
        else:
            preferred_versions = ['v1.0-trainval', 'v1.0-mini', 'v1.0-test']

        if requested_version != 'auto':
            preferred_versions = [requested_version] + [
                version for version in preferred_versions if version != requested_version
            ]

        for version in preferred_versions:
            if version_is_available(version):
                if requested_version not in ('auto', version):
                    print(
                        f"Warning: Requested nuScenes version '{requested_version}' is unavailable. "
                        f"Falling back to '{version}'."
                    )
                return version

        return requested_version if requested_version != 'auto' else 'v1.0-mini'

    def _build_nuscenes_label_lookup(self) -> np.ndarray:
        """Build a lookup table from raw nuScenes lidarseg ids to training ids."""
        if self._nuscenes_label_lookup is not None:
            return self._nuscenes_label_lookup

        default_label = self._get_default_label_index()
        category_files = [
            os.path.join(self.data_root, 'nuscenes', 'v1.0-trainval', 'category.json'),
            os.path.join(self.data_root, 'nuscenes', 'v1.0-mini', 'category.json'),
            os.path.join(self.data_root, 'nuscenes', 'v1.0-test', 'category.json'),
        ]

        category_path = next((path for path in category_files if os.path.exists(path)), None)
        if category_path is None:
            self._nuscenes_label_lookup = np.arange(self.num_classes, dtype=np.int64)
            return self._nuscenes_label_lookup

        with open(category_path, 'r', encoding='utf-8') as f:
            categories = json.load(f)

        max_category_index = max(category.get('index', 0) for category in categories)
        lookup = np.full(max_category_index + 1, default_label, dtype=np.int64)

        class_name_to_index = {name: idx for idx, name in enumerate(self.class_names)}

        def category_to_train_index(category_name: str) -> int:
            if category_name.startswith('human.pedestrian'):
                return class_name_to_index.get('pedestrian', default_label)
            if category_name == 'movable_object.barrier':
                return class_name_to_index.get('barrier', default_label)
            if category_name == 'movable_object.trafficcone':
                return class_name_to_index.get('traffic_cone', default_label)
            if category_name == 'vehicle.bicycle':
                return class_name_to_index.get('bicycle', default_label)
            if category_name.startswith('vehicle.bus'):
                return class_name_to_index.get('bus', default_label)
            if category_name == 'vehicle.car':
                return class_name_to_index.get('car', default_label)
            if category_name == 'vehicle.construction':
                return class_name_to_index.get('construction_vehicle', default_label)
            if category_name == 'vehicle.motorcycle':
                return class_name_to_index.get('motorcycle', default_label)
            if category_name == 'vehicle.trailer':
                return class_name_to_index.get('trailer', default_label)
            if category_name == 'vehicle.truck':
                return class_name_to_index.get('truck', default_label)
            if category_name == 'flat.driveable_surface':
                return class_name_to_index.get('driveable_surface', default_label)
            if category_name == 'flat.sidewalk':
                return class_name_to_index.get('sidewalk', default_label)
            if category_name == 'flat.terrain':
                return class_name_to_index.get('terrain', default_label)
            if category_name == 'static.manmade':
                return class_name_to_index.get('manmade', default_label)
            if category_name == 'static.vegetation':
                return class_name_to_index.get('vegetation', default_label)
            return default_label

        for category in categories:
            lookup[category['index']] = category_to_train_index(category['name'])

        self._nuscenes_label_lookup = lookup
        return self._nuscenes_label_lookup

    def _map_nuscenes_labels_to_training_ids(self, labels: np.ndarray) -> np.ndarray:
        """Map raw nuScenes lidarseg ids to the model's training taxonomy."""
        lookup = self._build_nuscenes_label_lookup()
        mapped_labels = np.full(labels.shape, self._get_default_label_index(), dtype=np.int64)
        valid_mask = (labels >= 0) & (labels < lookup.shape[0])
        mapped_labels[valid_mask] = lookup[labels[valid_mask]]
        return mapped_labels

    def _resolve_lidarseg_path(
        self,
        nusc,
        data_root: str,
        version: str,
        lidar_sample_data: Dict[str, Any],
    ) -> Optional[str]:
        """Resolve a lidarseg label path with SDK lookup first and file-path fallbacks second."""
        try:
            lidarseg_record = nusc.get('lidarseg', lidar_sample_data['token'])
            candidate_path = os.path.join(data_root, lidarseg_record['filename'])
            if os.path.exists(candidate_path):
                return candidate_path
        except Exception:
            pass

        candidate_versions = [version]
        if version == 'v1.0-mini':
            candidate_versions.append('v1.0-trainval')

        sample_token = lidar_sample_data['token']
        for candidate_version in candidate_versions:
            candidate_path = os.path.join(
                data_root,
                'lidarseg',
                candidate_version,
                f'{sample_token}_lidarseg.bin',
            )
            if os.path.exists(candidate_path):
                return candidate_path

        return None

    @staticmethod
    def _record_has_labels(record: Dict[str, Any]) -> bool:
        """Return whether a sample record has a usable label path."""
        labels_path = record.get('labels_path')
        return labels_path is not None and os.path.exists(labels_path)

    def _split_labeled_records_for_train_val(
        self,
        labeled_records: List[Dict[str, Any]],
        val_ratio: float,
    ) -> List[Dict[str, Any]]:
        """Split labeled records deterministically for fallback train/val usage."""
        if not labeled_records:
            return []

        sorted_records = sorted(labeled_records, key=lambda item: item.get('sample_token', ''))
        if len(sorted_records) == 1:
            return sorted_records if self.split == 'val' else []

        val_count = int(round(len(sorted_records) * val_ratio))
        val_count = max(1, min(val_count, len(sorted_records) - 1))

        if self.split == 'train':
            return sorted_records[:-val_count]
        if self.split == 'val':
            return sorted_records[-val_count:]
        return sorted_records

    def _select_split_records(
        self,
        all_records: List[Dict[str, Any]],
        split_scenes: Dict[str, List[str]],
        version: str,
    ) -> List[Dict[str, Any]]:
        """Select split records and fall back to a deterministic labeled split when needed."""
        split_key_map = {
            ('v1.0-mini', 'train'): 'mini_train',
            ('v1.0-mini', 'val'): 'mini_val',
            ('v1.0-mini', 'test'): 'mini_val',
            ('v1.0-trainval', 'train'): 'train',
            ('v1.0-trainval', 'val'): 'val',
            ('v1.0-test', 'test'): 'test',
        }

        if self.split not in ('train', 'val'):
            split_key = split_key_map.get((version, self.split))
            allowed_scene_names = set(split_scenes.get(split_key, [])) if split_key is not None else set()
            if not allowed_scene_names:
                return all_records
            return [record for record in all_records if record.get('scene_name') in allowed_scene_names]

        train_split_key = split_key_map.get((version, 'train'))
        val_split_key = split_key_map.get((version, 'val'))
        train_scene_names = set(split_scenes.get(train_split_key, [])) if train_split_key is not None else set()
        val_scene_names = set(split_scenes.get(val_split_key, [])) if val_split_key is not None else set()

        official_train_records = [
            record for record in all_records if not train_scene_names or record.get('scene_name') in train_scene_names
        ]
        official_val_records = [
            record for record in all_records if not val_scene_names or record.get('scene_name') in val_scene_names
        ]
        official_train_labeled = [record for record in official_train_records if self._record_has_labels(record)]
        official_val_labeled = [record for record in official_val_records if self._record_has_labels(record)]

        if official_train_labeled and official_val_labeled:
            selected_records = official_train_labeled if self.split == 'train' else official_val_labeled
            skipped_unlabeled = (
                len(official_train_records) - len(official_train_labeled)
                if self.split == 'train'
                else len(official_val_records) - len(official_val_labeled)
            )
            if skipped_unlabeled > 0:
                print(
                    f"Warning: Dropped {skipped_unlabeled} unlabeled nuScenes {self.split} samples "
                    f"for version '{version}'."
                )
            return selected_records

        labeled_records = [record for record in all_records if self._record_has_labels(record)]
        if not labeled_records:
            print(
                f"Warning: No labeled nuScenes samples found for split '{self.split}' "
                f"under version '{version}'."
            )
            return []

        official_total = len(official_train_records) + len(official_val_records)
        if official_total > 0 and len(official_val_records) > 0:
            val_ratio = len(official_val_records) / float(official_total)
        else:
            val_ratio = 0.2

        print(
            f"Warning: Official nuScenes split '{version}' has incomplete labels "
            f"(train_labeled={len(official_train_labeled)}, val_labeled={len(official_val_labeled)}). "
            f"Falling back to deterministic labeled split with val_ratio={val_ratio:.3f}."
        )
        return self._split_labeled_records_for_train_val(labeled_records, val_ratio)

    def _load_data_list(self) -> List[Dict]:
        """

        加载nuScenes数据列表

        Returns:
            data_list: 数据字典列表

        """
        import os

        data_list = []
        data_root = os.path.join(self.data_root, 'nuscenes')

        try:
            from nuscenes.nuscenes import NuScenes
            from nuscenes.utils.splits import create_splits_scenes
            
            version = self.version
            nusc = NuScenes(version=version, dataroot=data_root, verbose=False)
            split_scenes = create_splits_scenes()
            
            for sample in nusc.sample:
                scene = nusc.get('scene', sample['scene_token'])

                lidar_sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                lidar_path = os.path.join(data_root, lidar_sample_data['filename'])
                
                cam_paths = []
                for cam in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
                    if cam in sample['data']:
                        cam_sd = nusc.get('sample_data', sample['data'][cam])
                        cam_paths.append(os.path.join(data_root, cam_sd['filename']))
                
                labels_path = self._resolve_lidarseg_path(
                    nusc,
                    data_root,
                    version,
                    lidar_sample_data,
                )

                data_list.append({
                    'data_path': lidar_path,
                    'point_cloud_path': lidar_path,
                    'images_path': cam_paths,
                    'camera_path': lidar_path,
                    'labels_path': labels_path,
                    'scene_name': scene['name'],
                    'sample_token': sample['token'],
                })
                
        except Exception as e:
            print(f"Warning: Could not load nuScenes dataset: {e}")
            return self._create_dummy_data_list()

        return self._select_split_records(data_list, split_scenes, version)

    def _create_dummy_data_list(self) -> List[Dict]:
        """创建虚拟数据列表用于测试"""
        return [
            {
                'data_path': f'dummy_data_{i}',
                'point_cloud_path': f'dummy_lidar_{i}',
                'images_path': f'dummy_camera_{i}',
                'camera_path': f'dummy_camera_params_{i}',
                'labels_path': f'dummy_labels_{i}',
                'sample_token': f'token_{i}',
            }
            for i in range(100)
        ]

    def _load_point_cloud(self, data_path: str) -> np.ndarray:
        """

        加载点云数据

        Args:
            data_path: 点云文件路径

        Returns:
            points: [N, 4] (x, y, z, intensity)

        """
        import os

        if not os.path.exists(data_path):
            dummy_points = np.random.randn(10000, 4).astype(np.float32)
            dummy_points[:, :3] *= 20
            return dummy_points

        if data_path.endswith('.bin'):
            points = np.fromfile(data_path, dtype=np.float32)
            if points.size % 5 == 0:
                points = points.reshape(-1, 5)[:, :4]
            else:
                points = points.reshape(-1, 4)
        elif data_path.endswith('.npy'):
            points = np.load(data_path)
        else:
            points = np.random.randn(10000, 4).astype(np.float32)

        return points

    def _load_images(self, data_path: str) -> List[np.ndarray]:
        """

        加载多视角图像

        Args:
            data_path: 图像路径列表或目录路径

        Returns:
            images: [N_views, H, W, 3]

        """
        import os
        from PIL import Image

        images = []

        if isinstance(data_path, list):
            for cam_path in data_path:
                if os.path.exists(cam_path):
                    img = np.array(Image.open(cam_path))
                    images.append(img)
                else:
                    dummy_img = np.random.randint(0, 255, (370, 1224, 3), dtype=np.uint8)
                    images.append(dummy_img)
            return images

        cam_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                     'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

        if not os.path.exists(data_path):
            dummy_images = []
            for _ in cam_names:
                dummy_img = np.random.randint(0, 255, (370, 1224, 3), dtype=np.uint8)
                dummy_images.append(dummy_img)
            return dummy_images

        for cam_name in cam_names:
            cam_path = os.path.join(data_path, f"{cam_name}.jpg")
            if os.path.exists(cam_path):
                img = np.array(Image.open(cam_path))
                images.append(img)
            else:
                dummy_img = np.random.randint(0, 255, (370, 1224, 3), dtype=np.uint8)
                images.append(dummy_img)

        return images

    def _load_camera_params(self, data_path: str) -> Dict[str, np.ndarray]:
        """

        加载相机参数

        Args:
            data_path: 参数文件路径

        Returns:
            params: 相机参数字典

        """
        import os

        if not os.path.exists(data_path):
            intrinsics = np.eye(3, dtype=np.float32)
            extrinsics = np.eye(4, dtype=np.float32)
            return {
                'intrinsics': np.stack([intrinsics] * 6, axis=0),
                'extrinsics': np.stack([extrinsics] * 6, axis=0),
                'image_shape': np.array([370, 1224]),
            }

        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.eye(4, dtype=np.float32)

        return {
            'intrinsics': np.stack([intrinsics] * 6, axis=0),
            'extrinsics': np.stack([extrinsics] * 6, axis=0),
            'image_shape': np.array([370, 1224]),
        }

    def _load_labels(self, data_path: str) -> np.ndarray:
        """

        加载语义标签

        Args:
            data_path: 标签文件路径

        Returns:
            labels: [N] 点级标签

        """
        if data_path is None or not os.path.exists(data_path):
            return self._create_empty_bev_labels()

        if data_path.endswith('.bin'):
            raw_labels = np.fromfile(data_path, dtype=np.uint8).astype(np.int64)
            labels = self._map_nuscenes_labels_to_training_ids(raw_labels)
        elif data_path.endswith('.npy'):
            labels = np.load(data_path)
            if labels.ndim == 1 and labels.max(initial=-1) >= self.num_classes:
                labels = self._map_nuscenes_labels_to_training_ids(labels.astype(np.int64, copy=False))
        else:
            return self._create_empty_bev_labels()

        return labels.astype(np.int64, copy=False)


class ScanNetDataset(BEVBaseDataset):
    """

    ScanNet 数据集实现

    ScanNet: http://www.scan-net.org/
    室内场景数据集

    """

    def __init__(
        self,
        config,
        data_root: str = "./data",
        split: str = "train",
        transform: Optional[Any] = None,
    ):
        """

        初始化ScanNet数据集

        Args:
            config: BEVTextCLIPConfig 配置对象
            data_root: 数据根目录
            split: 数据集划分
            transform: 数据增强

        """
        super().__init__(config, data_root, split, transform)

    def _load_data_list(self) -> List[Dict]:
        """

        加载ScanNet数据列表

        Returns:
            data_list: 数据字典列表

        """
        import os

        data_list = []
        split_file = os.path.join(self.data_root, f"scannet_{self.split}.txt")

        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                scene_ids = [line.strip() for line in f]

            for scene_id in scene_ids:
                data_list.append({
                    'data_path': os.path.join(self.data_root, 'scans', scene_id),
                    'point_cloud_path': os.path.join(self.data_root, 'scans', scene_id, f'{scene_id}_vh_clean_2.ply'),
                    'images_path': os.path.join(self.data_root, 'scans', scene_id, 'color'),
                    'camera_path': os.path.join(self.data_root, 'scans', scene_id, 'intrinsics.txt'),
                    'labels_path': os.path.join(self.data_root, 'scans', scene_id, f'{scene_id}_vh_clean_2.labels.ply'),
                    'sample_token': scene_id,
                })
        else:
            self._create_dummy_data_list()

        return data_list

    def _create_dummy_data_list(self):
        """创建虚拟数据列表用于测试"""
        self.data_list = [
            {
                'data_path': f'dummy_scene_{i}',
                'point_cloud_path': f'dummy_point_cloud_{i}',
                'images_path': f'dummy_images_{i}',
                'camera_path': f'dummy_camera_{i}',
                'labels_path': f'dummy_labels_{i}',
                'sample_token': f'scene_{i}',
            }
            for i in range(100)
        ]

    def _load_point_cloud(self, data_path: str) -> np.ndarray:
        """

        加载点云数据

        Args:
            data_path: 点云文件路径

        Returns:
            points: [N, 3] (x, y, z)

        """
        import os

        if not os.path.exists(data_path):
            dummy_points = np.random.randn(50000, 3).astype(np.float32) * 5
            return dummy_points

        if data_path.endswith('.ply'):
            import trimesh
            mesh = trimesh.load(data_path)
            points = mesh.vertices.astype(np.float32)
        else:
            dummy_points = np.random.randn(50000, 3).astype(np.float32) * 5
            points = dummy_points

        return points

    def _load_images(self, data_path: str) -> List[np.ndarray]:
        """

        加载图像 (室内场景为单视角)

        Args:
            data_path: 图像路径

        Returns:
            images: [1, H, W, 3]

        """
        import os
        from PIL import Image

        if not os.path.exists(data_path):
            dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            return [dummy_img]

        images = []
        image_files = sorted([f for f in os.listdir(data_path) if f.endswith('.jpg')])

        for img_file in image_files[:1]:
            img_path = os.path.join(data_path, img_file)
            img = np.array(Image.open(img_path))
            images.append(img)

        if not images:
            dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            images = [dummy_img]

        return images

    def _load_camera_params(self, data_path: str) -> Dict[str, np.ndarray]:
        """

        加载相机参数

        Args:
            data_path: 参数文件路径

        Returns:
            params: 相机参数字典

        """
        import os

        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.eye(4, dtype=np.float32)

        if os.path.exists(data_path):
            try:
                with open(data_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 9:
                        fx, fy, cx, cy = [float(x) for x in lines[:4]]
                        intrinsics[0, 0] = fx
                        intrinsics[1, 1] = fy
                        intrinsics[0, 2] = cx
                        intrinsics[1, 2] = cy
            except Exception:
                pass

        return {
            'intrinsics': intrinsics[np.newaxis, :, :],
            'extrinsics': extrinsics[np.newaxis, :, :],
            'image_shape': np.array([480, 640]),
        }

    def _load_labels(self, data_path: str) -> np.ndarray:
        """

        加载语义标签

        Args:
            data_path: 标签文件路径

        Returns:
            labels: [N] 点级标签

        """
        import os

        if data_path is None or not os.path.exists(data_path):
            return self._create_empty_bev_labels()

        if data_path.endswith('.ply'):
            import trimesh
            mesh = trimesh.load(data_path)
            if hasattr(mesh, 'labels'):
                labels = mesh.labels.astype(np.int64)
            else:
                labels = np.zeros(len(mesh.vertices), dtype=np.int64)
        else:
            return self._create_empty_bev_labels()

        return labels


class DummyDataset(BEVBaseDataset):
    """

    虚拟数据集 (用于快速测试)

    """

    def __init__(
        self,
        config,
        data_root: str = "./data",
        split: str = "train",
        transform: Optional[Any] = None,
        num_samples: int = 100,
    ):
        """

        初始化虚拟数据集

        Args:
            config: BEVTextCLIPConfig 配置对象
            data_root: 数据根目录
            split: 数据集划分
            transform: 数据增强
            num_samples: 样本数量

        """
        self.config = config
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.class_names = config.class_names
        self.num_classes = config.num_classes
        self.num_samples = num_samples
        self.bev_resolution = getattr(config, 'bev_resolution', (200, 200))
        self.ignore_index = getattr(config, 'ignore_index', -100)
        
        self.data_list = self._load_data_list()

    def _load_data_list(self) -> List[Dict]:
        """创建虚拟数据列表"""
        return [
            {
                'data_path': f'dummy_sample_{i}',
                'point_cloud_path': f'dummy_point_cloud_{i}',
                'images_path': f'dummy_images_{i}',
                'camera_path': f'dummy_camera_{i}',
                'labels_path': f'dummy_labels_{i}',
                'sample_token': f'token_{i}',
            }
            for i in range(self.num_samples)
        ]

    def _load_point_cloud(self, data_path: str) -> np.ndarray:
        """生成虚拟点云"""
        num_points = random.randint(8000, 15000)
        points = np.random.randn(num_points, 4).astype(np.float32) * 10
        return points

    def _load_images(self, data_path: str) -> List[np.ndarray]:
        """生成虚拟图像"""
        images = []
        for _ in range(6):
            img = np.random.randint(0, 255, (370, 1224, 3), dtype=np.uint8)
            images.append(img)
        return images

    def _load_camera_params(self, data_path: str) -> Dict[str, np.ndarray]:
        """生成虚拟相机参数"""
        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.eye(4, dtype=np.float32)

        return {
            'intrinsics': np.stack([intrinsics] * 6, axis=0),
            'extrinsics': np.stack([extrinsics] * 6, axis=0),
            'image_shape': np.array([370, 1224]),
        }

    def _load_labels(self, data_path: str) -> np.ndarray:
        """生成虚拟标签
        
        Returns:
            labels: BEV 格式标签 [H, W] 或 点云格式标签 [N]
        """
        if hasattr(self, 'bev_resolution'):
            bev_h, bev_w = self.bev_resolution
            labels = np.random.randint(0, self.num_classes, size=(bev_h, bev_w), dtype=np.int64)
        else:
            num_points = random.randint(8000, 15000)
            labels = np.random.randint(0, self.num_classes, size=num_points, dtype=np.int64)
        return labels


def get_dataset_by_name(dataset_name: str, **kwargs) -> BEVBaseDataset:
    """

    根据数据集名称获取数据集类

    Args:
        dataset_name: 数据集名称 ('nuscenes', 'scannet', 'kitti', 'dummy')

    Returns:
        dataset: 数据集实例

    """
    dataset_map = {
        'nuscenes': NuScenesDataset,
        'scannet': ScanNetDataset,
        'kitti': KITTIDataset,
        'dummy': DummyDataset,
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_map.keys())}")

    return dataset_map[dataset_name](**kwargs)


class KITTIDataset(BEVBaseDataset):
    """

    KITTI 数据集实现

    KITTI: https://www.cvlibs.net/datasets/kitti/
    自动驾驶数据集

    """

    CAM_NAMES = ['image_02', 'image_03']  # 左视角和右视角

    def __init__(
        self,
        config,
        data_root: str = "./data",
        split: str = "train",
        transform: Optional[Any] = None,
    ):
        """

        初始化 KITTI 数据集

        Args:
            config: BEVTextCLIPConfig 配置对象
            data_root: 数据根目录
            split: 数据集划分
            transform: 数据增强

        """
        super().__init__(config, data_root, split, transform)

    def _load_data_list(self) -> List[Dict]:
        """

        加载 KITTI 数据列表

        Returns:
            data_list: 数据字典列表

        """
        import os

        data_list = []
        split_file = os.path.join(self.data_root, f"kitti_{self.split}.txt")

        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                frame_ids = [line.strip() for line in f]

            for frame_id in frame_ids:
                data_list.append({
                    'data_path': frame_id,
                    'point_cloud_path': os.path.join(self.data_root, 'velodyne_points', f'{frame_id}.bin'),
                    'images_path': os.path.join(self.data_root, 'image_02', f'{frame_id}.jpg'),
                    'camera_path': frame_id,
                    'labels_path': os.path.join(self.data_root, 'label_2', f'{frame_id}.txt'),
                    'sample_token': frame_id,
                })
        else:
            self._create_dummy_data_list()

        return data_list

    def _create_dummy_data_list(self):
        """创建虚拟数据列表用于测试"""
        self.data_list = [
            {
                'data_path': f'dummy_kitti_{i}',
                'point_cloud_path': f'dummy_velodyne_{i}',
                'images_path': f'dummy_image_{i}',
                'camera_path': f'dummy_calib_{i}',
                'labels_path': f'dummy_label_{i}',
                'sample_token': f'kitti_frame_{i}',
            }
            for i in range(100)
        ]

    def _load_point_cloud(self, data_path: str) -> np.ndarray:
        """

        加载点云数据

        Args:
            data_path: 点云文件路径

        Returns:
            points: [N, 4] (x, y, z, intensity)

        """
        import os

        if not os.path.exists(data_path):
            dummy_points = np.random.randn(50000, 4).astype(np.float32)
            dummy_points[:, :3] *= 30
            return dummy_points

        if data_path.endswith('.bin'):
            points = np.fromfile(data_path, dtype=np.float32)
            points = points.reshape(-1, 4)
        elif data_path.endswith('.npy'):
            points = np.load(data_path)
        else:
            dummy_points = np.random.randn(50000, 4).astype(np.float32)
            points = dummy_points

        return points

    def _load_images(self, data_path: str) -> List[np.ndarray]:
        """

        加载多视角图像 (KITTI 只有左视角图像)

        Args:
            data_path: 图像路径

        Returns:
            images: [1, H, W, 3]

        """
        import os
        from PIL import Image

        images = []

        if not os.path.exists(data_path):
            dummy_img = np.random.randint(0, 255, (370, 1224, 3), dtype=np.uint8)
            return [dummy_img]

        if os.path.exists(data_path):
            if data_path.endswith('.jpg') or data_path.endswith('.png'):
                img = np.array(Image.open(data_path))
                images.append(img)
            else:
                dummy_img = np.random.randint(0, 255, (370, 1224, 3), dtype=np.uint8)
                images.append(dummy_img)

        if not images:
            dummy_img = np.random.randint(0, 255, (370, 1224, 3), dtype=np.uint8)
            images = [dummy_img]

        return images

    def _load_camera_params(self, data_path: str) -> Dict[str, np.ndarray]:
        """

        加载相机参数

        Args:
            data_path: 参数文件路径 (KITTI 使用 calib 文件)

        Returns:
            params: 相机参数字典

        """
        import os

        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.eye(4, dtype=np.float32)

        calib_file = os.path.join(self.data_root, 'calib', f'{data_path}.txt')

        if os.path.exists(calib_file):
            try:
                with open(calib_file, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    if line.startswith('P2:'):
                        values = line.split(':')[1].strip().split()
                        p2 = np.array([float(x) for x in values]).reshape(3, 4)
                        intrinsics = p2[:3, :3]
                        break
            except Exception:
                pass

        return {
            'intrinsics': intrinsics[np.newaxis, :, :],
            'extrinsics': extrinsics[np.newaxis, :, :],
            'image_shape': np.array([370, 1224]),
        }

    def _load_labels(self, data_path: str) -> np.ndarray:
        """

        加载语义标签

        Args:
            data_path: 标签文件路径

        Returns:
            labels: [N] 点级标签 (KITTI 使用文本标签文件)

        """
        import os

        if data_path is None or not os.path.exists(data_path):
            return self._create_empty_bev_labels()

        if data_path.endswith('.txt'):
            labels = np.zeros(50000, dtype=np.int64)
            try:
                with open(data_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            obj_type = parts[0]
                            if obj_type in ['Car', 'Van', 'Truck']:
                                labels[0] = 1
                            elif obj_type in ['Pedestrian', 'Person_sitting']:
                                labels[0] = 2
                            elif obj_type in ['Cyclist']:
                                labels[0] = 3
                            break
            except Exception:
                pass
        elif data_path.endswith('.npy'):
            labels = np.load(data_path)
        else:
            return self._create_empty_bev_labels()

        return labels
