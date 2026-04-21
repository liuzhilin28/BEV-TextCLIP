#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: BEV-TextCLIP 数据集基类和具体数据集实现
日期: 2026年1月22日
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from abc import ABC, abstractmethod
import random


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

        self.data_list = self._load_data_list()

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

        data_dict = {
            'point_cloud': point_cloud.astype(np.float32),
            'images': images,
            'intrinsics': camera_params['intrinsics'].astype(np.float32),
            'extrinsics': camera_params['extrinsics'].astype(np.float32),
            'labels': labels,
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
        batch_images = np.transpose(batch_images, (0, 1, 4, 2, 3))
        batch_images = batch_images.astype(np.float32) / 255.0

        batch_intrinsics = np.stack(intrinsics, axis=0)
        batch_extrinsics = np.stack(extrinsics, axis=0)
        batch_labels = np.stack(labels, axis=0) if labels[0] is not None else None
        
        if isinstance(image_shapes[0], (list, tuple)):
            image_shapes_tensor = torch.tensor(image_shapes, dtype=torch.long)
        else:
            image_shapes_tensor = torch.tensor(image_shapes, dtype=torch.long)

        batch_dict = {
            'point_cloud': torch.from_numpy(np.stack(padded_point_clouds, axis=0)),
            'point_cloud_lengths': torch.tensor(point_cloud_lengths, dtype=torch.long),
            'images': torch.from_numpy(batch_images),
            'intrinsics': torch.from_numpy(batch_intrinsics),
            'extrinsics': torch.from_numpy(batch_extrinsics),
            'labels': torch.from_numpy(batch_labels) if batch_labels is not None else None,
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
        super().__init__(config, data_root, split, transform)
        self.version = getattr(config, 'nuscenes_version', 'v1.0-trainval')

    def _load_data_list(self) -> List[Dict]:
        """

        加载nuScenes数据列表

        Returns:
            data_list: 数据字典列表

        """
        import os
        from nuscenes.nuscenes import NuScenes

        data_list = []
        data_root = self.data_root
        split_file = os.path.join(data_root, f"nuscenes_infos_{self.split}.pkl")

        if os.path.exists(split_file):
            import pickle
            with open(split_file, 'rb') as f:
                infos = pickle.load(f)
            for info in infos['infos']:
                data_list.append({
                    'data_path': info['lidar_path'],
                    'point_cloud_path': info['lidar_path'],
                    'images_path': info['camera_path'],
                    'camera_path': info['camera_intrinsics_path'],
                    'labels_path': info['label_path'],
                    'sample_token': info['token'],
                })
        else:
            try:
                nusc = NuScenes(
                    version=self.version,
                    dataroot=data_root,
                    verbose=False
                )
                for sample in nusc.sample:
                    data_list.append({
                        'data_path': sample['data']['lidar'],
                        'point_cloud_path': sample['data']['lidar'],
                        'images_path': sample['data']['camera'],
                        'camera_path': sample['data']['camera'],
                        'labels_path': sample.get('labels', None),
                        'sample_token': sample['token'],
                    })
            except Exception as e:
                print(f"Warning: Could not load nuScenes dataset: {e}")
                self._create_dummy_data_list()

        return data_list

    def _create_dummy_data_list(self):
        """创建虚拟数据列表用于测试"""
        self.data_list = [
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
            data_path: 图像路径

        Returns:
            images: [N_views, H, W, 3]

        """
        import os
        from PIL import Image

        images = []
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
            dummy_labels = np.zeros(10000, dtype=np.int64)
            return dummy_labels

        if data_path.endswith('.npy'):
            labels = np.load(data_path)
        else:
            labels = np.zeros(10000, dtype=np.int64)

        return labels


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
            dummy_labels = np.zeros(50000, dtype=np.int64)
            return dummy_labels

        if data_path.endswith('.ply'):
            import trimesh
            mesh = trimesh.load(data_path)
            if hasattr(mesh, 'labels'):
                labels = mesh.labels.astype(np.int64)
            else:
                labels = np.zeros(len(mesh.vertices), dtype=np.int64)
        else:
            labels = np.zeros(50000, dtype=np.int64)

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
        self.bev_resolution = (200, 200)
        
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
            dummy_labels = np.zeros(50000, dtype=np.int64)
            return dummy_labels

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
            labels = np.zeros(50000, dtype=np.int64)

        return labels
