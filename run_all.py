#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
六大创新点完整测试与训练
"""

import argparse
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_quick_test():
    """快速测试所有模块"""
    print("=" * 60)
    print("六大创新点快速测试")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    batch_size = 2
    bev_channels = 256
    num_classes = 11
    class_names = ["vehicle", "pedestrian", "motorcycle", "bicycle",
                   "traffic cone", "barrier", "driveable surface",
                   "sidewalk", "other flat", "vegetation", "manmade"]
    
    # 1. 创新点1: BEV表征学习
    print("\n[创新点1] BEV空间表征学习...")
    from src.models.image_encoder import ImageEncoder
    from src.models.point_encoder import PointEncoder
    from src.models.fusion_module import BevFusion
    
    image_enc = ImageEncoder(image_encoder_type="resnet50", out_channels=bev_channels, freeze=False).to(device)
    point_enc = PointEncoder(in_channels=4, out_channels=bev_channels, encoder_type="pointpillar").to(device)
    fusion = BevFusion(in_channels=bev_channels, out_channels=bev_channels, fusion_type="gated_attention").to(device)
    
    images = torch.randn(batch_size, 6, 3, 224, 224).to(device)
    intrinsics = torch.eye(3).unsqueeze(0).expand(batch_size, 6, -1, -1).to(device)
    extrinsics = torch.eye(4).unsqueeze(0).expand(batch_size, 6, -1, -1).to(device)
    point_cloud = torch.randn(batch_size, 1000, 4).to(device)
    
    image_bev = image_enc(images, intrinsics, extrinsics)
    point_bev = point_enc(point_cloud)
    fused_bev = fusion(image_bev, point_bev)
    print(f"  BEV shape: {fused_bev.shape} ✓")
    
    # 2. 创新点2: 文本编码
    print("\n[创新点2] 动态文本编码...")
    from src.models.text_encoder import TextEncoder
    
    text_enc = TextEncoder(
        encoder_type="local_clip",
        model_path="./models/clip_random",
        output_dim=bev_channels,
        freeze=False,
        device=device,
    )
    class_embeddings = text_enc.generate_class_embeddings(class_names, device)
    print(f"  文本嵌入 shape: {class_embeddings.shape} ✓")
    
    # 3. 创新点3: 跨模态注意力
    print("\n[创新点3] 跨模态注意力...")
    from src.models.cross_attention import BevTextInteraction
    
    cross_attn = BevTextInteraction(
        bev_channels=bev_channels,
        text_channels=bev_channels,
        use_bidirectional=True,
    ).to(device)
    
    interaction_output = cross_attn(fused_bev, class_embeddings)
    print(f"  增强BEV shape: {interaction_output['enhanced_bev'].shape} ✓")
    
    # 4. 创新点4: 对比损失
    print("\n[创新点4] 多层级对比损失...")
    from src.models.losses import MultiModalContrastiveLoss
    
    # 清理显存
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    contrastive_loss = MultiModalContrastiveLoss().to(device)
    image_global = image_bev.mean(dim=[2, 3])
    point_global = point_bev.mean(dim=[2, 3])
    bev_global = interaction_output['enhanced_bev'].mean(dim=[2, 3])
    
    # 只测试全局对比损失，跳过需要大量显存的局部对比
    loss_global = contrastive_loss.global_contrast(
        image_global=image_global,
        point_global=point_global,
        text_global=class_embeddings,
    )
    print(f"  全局对比损失: {loss_global.item():.4f} ✓")
    
    # 5. 完整模型测试
    print("\n[完整模型] 测试...")
    from src.models.bev_textclip import BEVTextCLIP
    
    model = BEVTextCLIP(
        num_classes=num_classes,
        class_names=class_names,
        text_encoder_type="local_clip",
        image_encoder_type="resnet50",
        point_encoder_type="pointpillar",
        fusion_type="gated_attention",
        freeze_image_encoder=False,
        freeze_text_encoder=False,
    ).to(device)
    
    labels = torch.randint(0, num_classes, (batch_size, 200, 200)).to(device)
    output = model(images, intrinsics, extrinsics, point_cloud, labels=labels)
    
    print(f"  分割输出 shape: {output['segmentation_logits'].shape}")
    print(f"  损失: {output['loss'].item():.4f} ✓")
    
    print("\n" + "=" * 60)
    print("[OK] 六大创新点快速测试全部通过!")
    print("=" * 60)

def run_training(num_epochs=100, batch_size=2, learning_rate=1e-4):
    """运行完整训练"""
    from src.configs.bev_textclip_config import get_config
    from src.models.bev_textclip import BEVTextCLIP
    from torch.utils.data import DataLoader
    from src.dataloaders.base_dataset import DummyDataset, DataCollator
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    config = get_config("nuscenes")
    
    # 创建数据集
    collator = DataCollator(config)
    train_dataset = DummyDataset(config, num_samples=100, split='train')
    val_dataset = DummyDataset(config, num_samples=20, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                               num_workers=0, collate_fn=collator, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, collate_fn=collator, pin_memory=True)
    
    # 创建模型
    model = BEVTextCLIP(
        num_classes=config.num_classes,
        class_names=config.class_names,
        text_encoder_type="local_clip",
        freeze_text_encoder=False,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}, 可训练: {trainable_params:,}")
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            images = batch['images'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            extrinsics = batch['extrinsics'].to(device)
            point_cloud = batch['point_cloud'].to(device)
            labels = batch['labels'].to(device)
            
            if 'point_cloud_lengths' in batch:
                point_cloud_lengths = batch['point_cloud_lengths'].to(device)
            else:
                point_cloud_lengths = None
            
            optimizer.zero_grad()
            output = model(images, intrinsics, extrinsics, point_cloud, 
                          point_cloud_lengths, labels)
            loss = output['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        scheduler.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # 验证
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['images'].to(device)
                    intrinsics = batch['intrinsics'].to(device)
                    extrinsics = batch['extrinsics'].to(device)
                    point_cloud = batch['point_cloud'].to(device)
                    labels = batch['labels'].to(device)
                    
                    output = model(images, intrinsics, extrinsics, point_cloud)
                    preds = output['segmentation_logits'].argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.numel()
            
            accuracy = correct / total
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
    
    print("\n训练完成!")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/best_model.pt")
    print("模型已保存: checkpoints/best_model.pt")

def main():
    parser = argparse.ArgumentParser(description='六大创新点完整测试与训练')
    parser.add_argument('--quick_test', action='store_true', help='快速测试所有模块')
    parser.add_argument('--skip_tests', action='store_true', help='跳过测试，直接训练')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    args = parser.parse_args()
    
    if args.quick_test:
        run_quick_test()
    elif args.skip_tests:
        run_training(args.num_epochs, args.batch_size, args.learning_rate)
    else:
        run_quick_test()
        run_training(args.num_epochs, args.batch_size, args.learning_rate)

if __name__ == "__main__":
    main()
