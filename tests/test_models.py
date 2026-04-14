#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说明: BEV-TextCLIP单元测试，覆盖所有模块
日期: 2026年1月22日
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.configs.bev_textclip_config import BEVTextCLIPConfig, get_config


class TestConfig:
    """配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = BEVTextCLIPConfig()
        assert config.dataset == "nuscenes"
        assert config.num_classes == 16
        assert config.bev_resolution == (200, 200)
        assert config.bev_channels == 256

    def test_nuscenes_config(self):
        """测试 nuScenes 配置"""
        config = get_config("nuscenes")
        assert config.dataset == "nuscenes"
        assert config.num_classes == 16
        assert "vehicle" in config.class_names

    def test_scannet_config(self):
        """测试 ScanNet 配置"""
        config = get_config("scannet")
        assert config.dataset == "scannet"
        assert config.num_classes == 20
        assert "wall" in config.class_names


class TestImageEncoder:
    """图像编码器测试"""

    @pytest.fixture
    def config(self):
        return BEVTextCLIPConfig()

    def test_resnet_encoder_output_shape(self, config):
        """测试 ResNet 输出形状"""
        from src.models.image_encoder import ResNetEncoder

        encoder = ResNetEncoder(
            in_channels=3,
            out_channels=512,
            pretrained=False,
            freeze=True,
        )

        x = torch.randn(2, 3, 224, 224)
        features = encoder(x)

        assert 'layer4' in features
        assert features['layer4'].shape == (2, 512, 7, 7)

    def test_image_encoder_output_shape(self, config):
        """测试图像编码器输出形状"""
        from src.models.image_encoder import ImageEncoder

        encoder = ImageEncoder(
            in_channels=3,
            out_channels=256,
            image_encoder_type="resnet50",
            bev_grid_size=(200, 200),
            bev_depth_bins=4,
            point_cloud_range=config.point_cloud_range,
            pretrained=False,
            freeze=True,
        )

        images = torch.randn(2, 6, 3, 224, 224)
        intrinsics = torch.eye(3).unsqueeze(0).expand(2, 6, -1, -1)
        extrinsics = torch.eye(4).unsqueeze(0).expand(2, 6, -1, -1)

        bev = encoder(images, intrinsics, extrinsics)

        assert bev.shape == (2, 256, 200, 200)


class TestPointEncoder:
    """点云编码器测试"""

    def test_point_encoder_output_shape(self):
        """测试点云编码器输出形状"""
        from src.models.point_encoder import PointEncoder

        encoder = PointEncoder(
            in_channels=4,
            out_channels=256,
            encoder_type="pointpillar",
            grid_size=(200, 200),
        )

        point_cloud = torch.randn(2, 10000, 4)

        bev = encoder(point_cloud)

        assert bev.shape == (2, 256, 200, 200)

    def test_point_encoder_empty_points(self):
        """测试空点云输入"""
        from src.models.point_encoder import PointEncoder

        encoder = PointEncoder(
            in_channels=4,
            out_channels=256,
            encoder_type="pointpillar",
            grid_size=(200, 200),
        )

        point_cloud = torch.randn(1, 0, 4)

        bev = encoder(point_cloud)

        assert bev.shape[1] == 256
        assert bev.shape[2] == 200
        assert bev.shape[3] == 200


class TestFusionModule:
    """融合模块测试"""

    def test_gated_attention_fusion(self):
        """测试门控注意力融合"""
        from src.models.fusion_module import GatedFeatureFusion

        fusion = GatedFeatureFusion(
            in_channels=256,
            out_channels=256,
        )

        image_bev = torch.randn(2, 256, 200, 200)
        point_bev = torch.randn(2, 256, 200, 200)

        fused = fusion(image_bev, point_bev)

        assert fused.shape == (2, 256, 200, 200)

    def test_concatenation_fusion(self):
        """测试拼接融合"""
        from src.models.fusion_module import ConcatenationFusion

        fusion = ConcatenationFusion(
            in_channels=256,
            out_channels=256,
        )

        image_bev = torch.randn(2, 256, 200, 200)
        point_bev = torch.randn(2, 256, 200, 200)

        fused = fusion(image_bev, point_bev)

        assert fused.shape == (2, 256, 200, 200)

    def test_addition_fusion(self):
        """测试相加融合"""
        from src.models.fusion_module import AdditionFusion

        fusion = AdditionFusion(
            in_channels=256,
            out_channels=256,
        )

        image_bev = torch.randn(2, 256, 200, 200)
        point_bev = torch.randn(2, 256, 200, 200)

        fused = fusion(image_bev, point_bev)

        assert fused.shape == (2, 256, 200, 200)


class TestCrossAttention:
    """交叉注意力测试"""

    def test_bev_text_cross_attention(self):
        """测试 BEV-文本交叉注意力"""
        from src.models.cross_attention import BEVTextCrossAttention

        attn = BEVTextCrossAttention(
            bev_channels=256,
            text_channels=256,
            num_heads=8,
            use_bidirectional=True,
        )

        bev_features = torch.randn(2, 40000, 256)
        text_embeddings = torch.randn(16, 256)

        enhanced, weights = attn(bev_features, text_embeddings)

        assert enhanced.shape == bev_features.shape
        assert weights.shape[0] == 2


class TestLosses:
    """损失函数测试"""

    def test_info_nce(self):
        """测试 InfoNCE 损失"""
        from src.models.losses import InfoNCE

        loss_fn = InfoNCE(temperature=0.07)

        query = torch.randn(4, 128)
        positive = torch.randn(4, 128)
        negatives = torch.randn(4, 32, 128)

        loss = loss_fn(query, positive, negatives)

        assert loss.item() >= 0

    def test_focal_loss(self):
        """测试 Focal Loss"""
        from src.models.losses import FocalLoss

        loss_fn = FocalLoss(gamma=2.0)

        inputs = torch.randn(2, 16, 200, 200)
        targets = torch.randint(0, 16, (2, 200, 200))

        loss = loss_fn(inputs, targets)

        assert loss.item() >= 0

    def test_dice_loss(self):
        """测试 Dice Loss"""
        from src.models.losses import DiceLoss

        loss_fn = DiceLoss()

        inputs = torch.randn(2, 16, 200, 200)
        targets = torch.randint(0, 16, (2, 200, 200))

        loss = loss_fn(inputs, targets)

        assert loss.item() >= 0


class TestMainModel:
    """主模型测试"""

    @pytest.fixture
    def config(self):
        return BEVTextCLIPConfig()

    def test_model_creation(self, config):
        """测试模型创建"""
        from src.models.bev_textclip import BEVTextCLIP

        model = BEVTextCLIP(
            num_classes=config.num_classes,
            class_names=config.class_names,
            image_encoder_type="resnet50",
            point_encoder_type="pointpillar",
            pretrained=False,
            freeze_image_encoder=False,
            freeze_point_encoder=False,
            freeze_text_encoder=True,
        )

        assert model is not None
        assert model.num_classes == config.num_classes

    def test_model_forward(self, config):
        """测试模型前向传播"""
        from src.models.bev_textclip import BEVTextCLIP

        model = BEVTextCLIP(
            num_classes=config.num_classes,
            class_names=config.class_names,
            image_encoder_type="resnet50",
            point_encoder_type="pointpillar",
            pretrained=False,
            freeze_image_encoder=False,
            freeze_point_encoder=False,
            freeze_text_encoder=True,
        )

        images = torch.randn(1, 6, 3, 224, 224)
        intrinsics = torch.eye(3).unsqueeze(0).expand(1, 6, -1, -1)
        extrinsics = torch.eye(4).unsqueeze(0).expand(1, 6, -1, -1)
        point_cloud = torch.randn(1, 10000, 4)

        output = model(
            images=images,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            point_cloud=point_cloud,
        )

        assert 'segmentation_logits' in output
        assert output['segmentation_logits'].shape == (1, config.num_classes, 200, 200)


class TestDataset:
    """数据集测试"""

    def test_dummy_dataset(self):
        """测试虚拟数据集"""
        from src.dataloaders.base_dataset import DummyDataset
        from src.configs.bev_textclip_config import get_config

        config = get_config("nuscenes")
        dataset = DummyDataset(
            config=config,
            num_samples=10,
        )

        assert len(dataset) == 10

        sample = dataset[0]
        assert 'point_cloud' in sample
        assert 'images' in sample
        assert 'labels' in sample


class TestIntegration:
    """集成测试"""

    def test_bev_textclip_pipeline(self):
        """
        端到端测试：完整推理流程

        测试步骤：
        1. 加载配置
        2. 创建模型
        3. 前向传播
        4. 计算损失
        5. 验证输出形状
        """
        from src.configs.bev_textclip_config import BEVTextCLIPConfig
        from src.models.bev_textclip import BEVTextCLIP

        config = BEVTextCLIPConfig()

        model = BEVTextCLIP(
            num_classes=config.num_classes,
            class_names=config.class_names,
            image_encoder_type="resnet50",
            point_encoder_type="pointpillar",
            pretrained=False,
            freeze_image_encoder=False,
            freeze_point_encoder=False,
            freeze_text_encoder=True,
        )

        images = torch.randn(1, 6, 3, 224, 224)
        intrinsics = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(1, 6, -1, -1)
        extrinsics = torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(1, 6, -1, -1)
        point_cloud = torch.randn(1, 10000, 4)
        labels = torch.randint(0, config.num_classes, (1, 200, 200))

        output = model(
            images=images,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            point_cloud=point_cloud,
            labels=labels,
        )

        assert 'segmentation_logits' in output
        assert output['segmentation_logits'].shape == (1, config.num_classes, 200, 200)
        assert 'loss' in output
        assert output['loss'].item() >= 0

    def test_model_predict(self):
        """
        测试模型预测模式

        验证 predict 方法正确返回预测结果和概率
        """
        from src.configs.bev_textclip_config import BEVTextCLIPConfig
        from src.models.bev_textclip import BEVTextCLIP

        config = BEVTextCLIPConfig()

        model = BEVTextCLIP(
            num_classes=config.num_classes,
            class_names=config.class_names,
            image_encoder_type="resnet50",
            point_encoder_type="pointpillar",
            pretrained=False,
            freeze_image_encoder=False,
            freeze_point_encoder=False,
            freeze_text_encoder=True,
        )

        images = torch.randn(1, 6, 3, 224, 224)
        intrinsics = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(1, 6, -1, -1)
        extrinsics = torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(1, 6, -1, -1)
        point_cloud = torch.randn(1, 10000, 4)

        result = model.predict(
            images=images,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            point_cloud=point_cloud,
        )

        assert 'predictions' in result
        assert 'probabilities' in result
        assert result['predictions'].shape == (1, 200, 200)
        assert result['probabilities'].shape == (1, config.num_classes, 200, 200)

    def test_fusion_types(self):
        """
        测试不同融合类型

        验证模型支持多种融合策略
        """
        from src.configs.bev_textclip_config import BEVTextCLIPConfig
        from src.models.bev_textclip import BEVTextCLIP

        config = BEVTextCLIPConfig()
        fusion_types = ["gated_attention", "concatenation", "addition"]

        images = torch.randn(1, 6, 3, 224, 224)
        intrinsics = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(1, 6, -1, -1)
        extrinsics = torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(1, 6, -1, -1)
        point_cloud = torch.randn(1, 10000, 4)

        for fusion_type in fusion_types:
            model = BEVTextCLIP(
                num_classes=config.num_classes,
                class_names=config.class_names,
                image_encoder_type="resnet50",
                point_encoder_type="pointpillar",
                fusion_type=fusion_type,
                pretrained=False,
                freeze_image_encoder=False,
                freeze_point_encoder=False,
                freeze_text_encoder=True,
            )

            output = model(
                images=images,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                point_cloud=point_cloud,
            )

            assert output['segmentation_logits'].shape == (1, config.num_classes, 200, 200)


class TestOptimization:
    """优化工具测试"""

    def test_gradient_checkpointing(self):
        """测试梯度检查点"""
        from src.utils.optimization import GradientCheckpointing

        checkpointing = GradientCheckpointing(enabled=True)
        assert checkpointing.enabled is True

        checkpointing.enabled = False
        assert checkpointing.enabled is False

    def test_mixed_precision(self):
        """测试混合精度"""
        from src.utils.optimization import MixedPrecisionManager

        amp = MixedPrecisionManager(enabled=False)
        assert amp.enabled is False

        amp = MixedPrecisionManager(enabled=True, dtype=torch.float16)
        assert amp.enabled is True
        assert amp.dtype == torch.float16

    def test_memory_optimizer(self):
        """测试内存优化"""
        from src.utils.optimization import MemoryOptimizer

        mem_opt = MemoryOptimizer(enabled=False)
        assert mem_opt.enabled is False

        info = mem_opt.get_memory_info()
        assert 'allocated_gb' in info
        assert 'reserved_gb' in info

    def test_inference_optimizer(self):
        """测试推理优化"""
        from src.utils.optimization import InferenceOptimizer

        opt = InferenceOptimizer()
        assert len(opt.optimizations_applied) == 0

        opt.enable_cudnn_benchmark(True)
        assert "cudnn.benchmark" in opt.optimizations_applied


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
