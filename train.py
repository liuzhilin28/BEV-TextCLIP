#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.configs.bev_textclip_config import BEVTextCLIPConfig, get_config
from src.models.bev_textclip import BEVTextCLIP, create_bev_textclip_model
from src.dataloaders.base_dataset import DummyDataset, DataCollator


def setup_logging(log_dir: str):
    """Setup logging"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        try:
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

            output = model(
                images=images,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                point_cloud=point_cloud,
                point_cloud_lengths=point_cloud_lengths,
                labels=labels,
            )

            if 'loss' in output:
                loss = output['loss']
            else:
                loss = output['segmentation_logits'].sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            avg_loss = total_loss / num_batches
            pbar.set_postfix(loss=f"{loss.item():.4f}",avg=f"{avg_loss:.4f}")

        except Exception as e:
            logger.error(f'Error in batch {batch_idx}: {e}')
            continue

    avg_loss = total_loss / max(num_batches, 1)
    logger.info(f'Epoch {epoch} - Avg Loss: {avg_loss:.4f}')
    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    logger: logging.Logger,
):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    for batch in tqdm(val_loader, desc='Evaluating'):
        try:
            images = batch['images'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            extrinsics = batch['extrinsics'].to(device)
            point_cloud = batch['point_cloud'].to(device)
            labels = batch['labels'].to(device)

            if 'point_cloud_lengths' in batch:
                point_cloud_lengths = batch['point_cloud_lengths'].to(device)
            else:
                point_cloud_lengths = None

            output = model(
                images=images,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                point_cloud=point_cloud,
                point_cloud_lengths=point_cloud_lengths,
            )

            predictions = output['segmentation_logits'].argmax(dim=1)
            accuracy = (predictions == labels).float().mean()

            if 'loss' in output:
                total_loss += output['loss'].item()

            total_accuracy += accuracy.item()
            num_batches += 1

        except Exception as e:
            logger.error(f'Evaluation error: {e}')
            continue

    avg_loss = total_loss / max(num_batches, 1)
    avg_accuracy = total_accuracy / max(num_batches, 1)
    logger.info(f'Val Loss: {avg_loss:.4f}, Val Accuracy: {avg_accuracy:.4f}')
    return avg_loss, avg_accuracy


def main():
    parser = argparse.ArgumentParser(description='Train BEV-TextCLIP')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(args.log_dir)

    logger.info(f'Device: {device}')
    logger.info(f'Args: {args}')

    config = get_config('nuscenes')
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay

    logger.info(f'Config: num_classes={config.num_classes}, bev_resolution={config.bev_resolution}')

    collator = DataCollator(config)

    train_dataset = DummyDataset(config, num_samples=100, split='train')
    val_dataset = DummyDataset(config, num_samples=20, split='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        pin_memory=True,
        drop_last=False,
    )

    logger.info(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')

    model = create_bev_textclip_model(config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total params: {total_params:,}, Trainable: {trainable_params:,}')

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6,
    )

    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume and os.path.exists(args.resume):
        logger.info(f'Resume from {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f'Epoch {epoch}/{args.num_epochs}')

        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, logger
        )

        val_loss, val_accuracy = evaluate(
            model, val_loader, device, logger
        )

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'config': config,
            }, checkpoint_path)
            logger.info(f'Saved best model to {checkpoint_path}')

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            logger.info(f'Saved checkpoint to {checkpoint_path}')

    logger.info('Training completed!')


if __name__ == '__main__':
    main()
