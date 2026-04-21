#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: BEV-TextCLIP 配置文件模块
日期: 2026年1月22日
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import yaml


@dataclass
class BEVTextCLIPConfig:
    """BEV-TextCLIP configuration class"""
    
    dataset: str = "nuscenes"
    num_classes: int = 16
    
    class_names: List[str] = field(default_factory=lambda: [
        "vehicle", "pedestrian", "motorcycle", "bicycle",
        "traffic cone", "barrier", "driveable surface",
        "sidewalk", "other flat", "vegetation", "manmade"
    ])
    
    bev_resolution: Tuple[int, int] = (200, 200)
    bev_channels: int = 256
    num_depth_bins: int = 4
    
    bev_range: Tuple[float, float, float, float] = (-20.0, -20.0, 20.0, 20.0)
    
    image_encoder_type: str = "resnet50"
    image_pretrained: bool = True
    image_freeze: bool = True
    image_out_channels: int = 512
    
    use_lss: bool = True
    lss_depth_net_hidden: int = 96
    
    point_encoder_type: str = "pointpillar"
    voxel_size: Tuple[float, float, float] = (0.05, 0.05, 0.05)
    point_cloud_range: Tuple[float, float, float, float, float, float] = (
        -20.0, -20.0, -2.0, 20.0, 20.0, 6.0
    )
    use_height_pooling: bool = True
    
    text_encoder_type: str = "local_clip"
    text_pretrained: bool = False
    text_freeze: bool = False
    text_embedding_dim: int = 256
    
    fusion_type: str = "gated_attention"
    use_bidirectional_attention: bool = True
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    fusion_hidden_dim: int = 256
    
    use_contrastive: bool = True
    contrastive_weight: float = 0.5
    temperature: float = 0.07
    use_memory_bank: bool = False
    memory_size: int = 65536
    
    contrast_global: bool = True
    contrast_local: bool = True
    contrast_cross_modal: bool = True
    
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 100
    warmup_epochs: int = 5
    
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine"
    
    use_amp: bool = True
    
    eval_interval: int = 5
    save_interval: int = 10
    
    use_augmentation: bool = True
    rotation_range: Tuple[float, float] = (-0.1, 0.1)
    translation_range: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    
    def __post_init__(self):
        """Post initialization"""
        if self.dataset == "scannet":
            self.num_classes = 20
            self.class_names = [
                'wall', 'floor', 'cabinet', 'bed', 'chair',
                'sofa', 'table', 'door', 'window', 'bookshelf',
                'picture', 'counter', 'desk', 'curtain',
                'refrigerator', 'shower curtain', 'toilet',
                'sink', 'bathtub', 'other furniture'
            ]
            self.bev_range = (-10.0, -10.0, 10.0, 10.0)
            self.point_cloud_range = (-10.0, -10.0, -2.0, 10.0, 10.0, 6.0)
            self.voxel_size = (0.02, 0.02, 0.02)
        
        elif self.dataset == "nuscenes":
            self.num_classes = 16
            self.class_names = [
                "vehicle", "pedestrian", "motorcycle", "bicycle",
                "traffic cone", "barrier", "driveable surface",
                "sidewalk", "other flat", "vegetation", "manmade"
            ]
            self.bev_range = (-20.0, -20.0, 20.0, 20.0)
            self.point_cloud_range = (-20.0, -20.0, -2.0, 20.0, 20.0, 6.0)
            self.voxel_size = (0.05, 0.05, 0.05)
        
        elif self.dataset == "kitti":
            self.num_classes = 19
            self.bev_range = (-40.0, -40.0, 40.0, 40.0)
            self.point_cloud_range = (-40.0, -40.0, -3.0, 40.0, 40.0, 3.0)
            self.voxel_size = (0.1, 0.1, 0.1)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "BEVTextCLIPConfig":
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, config_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, indent=2, default_flow_style=False)
    
    def get_class_embeddings_path(self) -> str:
        """Get class embeddings file path"""
        return f"checkpoints/{self.dataset}_class_embeddings.pt"
    
    def get_pretraining_path(self) -> str:
        """Get pre-training weights path"""
        return f"checkpoints/{self.dataset}_pretrain.pt"
    
    def get_finetuning_path(self) -> str:
        """Get fine-tuning weights path"""
        return f"checkpoints/{self.dataset}_finetune.pt"


def get_config(dataset: str = "nuscenes") -> BEVTextCLIPConfig:
    """Get configuration for specified dataset"""
    return BEVTextCLIPConfig(dataset=dataset)


def create_nuscenes_config() -> BEVTextCLIPConfig:
    """Create nuScenes dataset configuration"""
    return BEVTextCLIPConfig(
        dataset="nuscenes",
        num_classes=16,
        bev_resolution=(200, 200),
        bev_channels=256,
        point_encoder_type="pointpillar",
        use_contrastive=True,
        contrastive_weight=0.5,
    )


def create_scannet_config() -> BEVTextCLIPConfig:
    """Create ScanNet dataset configuration"""
    return BEVTextCLIPConfig(
        dataset="scannet",
        num_classes=20,
        bev_resolution=(100, 100),
        bev_channels=128,
        point_encoder_type="pointpillar",
        use_contrastive=True,
        contrastive_weight=0.3,
    )
