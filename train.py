#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import os
import sys
import gc
from contextlib import nullcontext
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
from src.dataloaders.base_dataset import NuScenesDataset, DataCollator


def clear_cuda_memory():
    """Clear CUDA cache after recoverable failures such as OOM."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def resolve_repo_path(repo_root: str, path: str) -> str:
    """Resolve relative paths against the repository root."""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(repo_root, path))


def count_labeled_samples(dataset) -> int:
    """Count dataset samples that already resolved to a concrete label file."""
    return sum(
        1 for item in getattr(dataset, 'data_list', [])
        if item.get('labels_path') is not None and os.path.exists(item['labels_path'])
    )


def setup_logging(log_dir: str):
    """Setup logging"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log'), mode='w'),
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
    use_amp: bool = True,
):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    amp_enabled = use_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
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

            optimizer.zero_grad(set_to_none=True)

            autocast_context = torch.amp.autocast(device_type='cuda', dtype=torch.float16) if amp_enabled else nullcontext()
            with autocast_context:
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

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1

            avg_loss = total_loss / num_batches
            pbar.set_postfix(loss=f"{loss.item():.4f}",avg=f"{avg_loss:.4f}")

        except Exception as e:
            if 'out of memory' in str(e).lower():
                clear_cuda_memory()
            logger.error(f'Error in batch {batch_idx}: {e}')
            continue

    avg_loss = total_loss / max(num_batches, 1)
    logger.info(f'Epoch {epoch} - Avg Loss: {avg_loss:.4f}')
    return avg_loss


def update_confusion_matrix(confmat: torch.Tensor,
                            preds: torch.Tensor,
                            labels: torch.Tensor,
                            mask: torch.Tensor,
                            num_classes: int) -> torch.Tensor:
    """
    preds:  [B, H, W]
    labels: [B, H, W]
    mask:   [B, H, W]  True 表示有效像素
    """
    preds = preds[mask].view(-1).long()
    labels = labels[mask].view(-1).long()

    valid = (labels >= 0) & (labels < num_classes)
    preds = preds[valid]
    labels = labels[valid]

    if preds.numel() == 0:
        return confmat

    inds = labels * num_classes + preds
    confmat += torch.bincount(
        inds, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)

    return confmat


def compute_iou_from_confmat(confmat: torch.Tensor):
    """
    confmat[i, j] = GT为i、预测为j 的像素数
    """
    confmat = confmat.float()

    tp = torch.diag(confmat)
    fp = confmat.sum(dim=0) - tp
    fn = confmat.sum(dim=1) - tp

    denom = tp + fp + fn
    per_class_iou = tp / torch.clamp(denom, min=1.0)

    valid_classes = denom > 0
    miou = per_class_iou[valid_classes].mean().item() if valid_classes.any() else 0.0

    return miou, per_class_iou, valid_classes

"""@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    logger: logging.Logger,
    use_amp: bool = True,
):
  
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    skipped_batches = 0
    amp_enabled = use_amp and device.type == 'cuda'

    for batch in tqdm(val_loader, desc='Evaluating'):
        try:
            images = batch['images'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            extrinsics = batch['extrinsics'].to(device)
            point_cloud = batch['point_cloud'].to(device)
            labels = batch['labels'].to(device)
            label_mask = batch['label_mask'].to(device) if batch.get('label_mask') is not None else (labels >= 0)

            if 'point_cloud_lengths' in batch:
                point_cloud_lengths = batch['point_cloud_lengths'].to(device)
            else:
                point_cloud_lengths = None

            autocast_context = torch.amp.autocast(device_type='cuda', dtype=torch.float16) if amp_enabled else nullcontext()
            with autocast_context:
                output = model(
                    images=images,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    point_cloud=point_cloud,
                    point_cloud_lengths=point_cloud_lengths,
                    labels=labels,
                )

            predictions = output['segmentation_logits'].argmax(dim=1)
            valid_count = label_mask.sum()
            if valid_count.item() == 0:
                skipped_batches += 1
                continue
            accuracy = (predictions[label_mask] == labels[label_mask]).float().mean()

            if 'loss' in output:
                total_loss += output['loss'].item()

            total_accuracy += accuracy.item()
            num_batches += 1

        except Exception as e:
            if 'out of memory' in str(e).lower():
                clear_cuda_memory()
            logger.error(f'Evaluation error: {e}')
            continue

    avg_loss = total_loss / max(num_batches, 1)
    avg_accuracy = total_accuracy / max(num_batches, 1)
    if num_batches == 0:
        logger.warning(
            f'Validation skipped: no batches with valid labels. skipped_batches={skipped_batches}'
        )
    logger.info(f'Val Loss: {avg_loss:.4f}, Val Accuracy: {avg_accuracy:.4f}')
    return avg_loss, avg_accuracy"""
@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    logger: logging.Logger,
    num_classes: int,
    class_names=None,
    use_amp: bool = True,
):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    skipped_batches = 0
    amp_enabled = use_amp and device.type == 'cuda'

    confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    for batch in tqdm(val_loader, desc='Evaluating'):
        try:
            images = batch['images'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            extrinsics = batch['extrinsics'].to(device)
            point_cloud = batch['point_cloud'].to(device)
            labels = batch['labels'].to(device)
            label_mask = batch['label_mask'].to(device) if batch.get('label_mask') is not None else (labels >= 0)

            if 'point_cloud_lengths' in batch:
                point_cloud_lengths = batch['point_cloud_lengths'].to(device)
            else:
                point_cloud_lengths = None

            autocast_context = torch.amp.autocast(device_type='cuda', dtype=torch.float16) if amp_enabled else nullcontext()
            with autocast_context:
                output = model(
                    images=images,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    point_cloud=point_cloud,
                    point_cloud_lengths=point_cloud_lengths,
                    labels=labels,
                )

            predictions = output['segmentation_logits'].argmax(dim=1)

            valid_count = label_mask.sum()
            if valid_count.item() == 0:
                skipped_batches += 1
                continue

            accuracy = (predictions[label_mask] == labels[label_mask]).float().mean()

            if 'loss' in output:
                total_loss += output['loss'].item()

            total_accuracy += accuracy.item()
            num_batches += 1

            confmat = update_confusion_matrix(
                confmat, predictions, labels, label_mask, num_classes
            )

        except Exception as e:
            if 'out of memory' in str(e).lower():
                clear_cuda_memory()
            logger.error(f'Evaluation error: {e}')
            continue

    avg_loss = total_loss / max(num_batches, 1)
    avg_accuracy = total_accuracy / max(num_batches, 1)

    miou, per_class_iou, valid_classes = compute_iou_from_confmat(confmat)

    if num_batches == 0:
        logger.warning(
            f'Validation skipped: no batches with valid labels. skipped_batches={skipped_batches}'
        )

    logger.info(
        f'Val Loss: {avg_loss:.4f}, '
        f'Val Accuracy: {avg_accuracy:.4f}, '
        f'Val mIoU: {miou:.4f}'
    )

    if class_names is not None:
        for i, name in enumerate(class_names):
            if valid_classes[i]:
                logger.info(f'  IoU[{name}]: {per_class_iou[i].item():.4f}')
            else:
                logger.info(f'  IoU[{name}]: N/A')

    return avg_loss, avg_accuracy, miou, per_class_iou.detach().cpu()

def main():
    parser = argparse.ArgumentParser(description='Train BEV-TextCLIP')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--unfreeze_image_encoder', action='store_true')
    parser.add_argument('--unfreeze_text_encoder', action='store_true')
    parser.add_argument('--disable_amp', action='store_true')
    parser.add_argument('--text_model_path', type=str, default=None)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(args.log_dir)

    logger.info(f'Device: {device}')
    logger.info(f'Args: {args}')

    config = get_config('nuscenes')
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    if args.unfreeze_image_encoder:
        config.image_freeze = False
    if args.unfreeze_text_encoder:
        config.text_freeze = False
    config.use_amp = not args.disable_amp
    if args.text_model_path:
        config.text_model_path = args.text_model_path
    if config.text_encoder_type == 'local_clip':
        config.text_model_path = resolve_repo_path(repo_root, config.text_model_path)

    logger.info(f'Config: num_classes={config.num_classes}, bev_resolution={config.bev_resolution}')
    logger.info(
        f'Training: image_freeze={config.image_freeze}, '
        f'text_freeze={config.text_freeze}, amp={config.use_amp}, image_size={config.image_size}'
    )
    if config.text_encoder_type == 'local_clip':
        logger.info(
            f'Text encoder: type={config.text_encoder_type}, '
            f'pretrained={config.text_pretrained}, model_path={config.text_model_path}'
        )
        if not os.path.isdir(config.text_model_path):
            logger.error(
                f'Local text model path not found: {config.text_model_path}. '
                'Please sync the pretrained model directory to the server.'
            )
            return
    else:
        logger.info(
            f'Text encoder: type={config.text_encoder_type}, '
            f'pretrained={config.text_pretrained}, model_name={config.text_model_name}'
        )

    collator = DataCollator(config)

    train_dataset = NuScenesDataset(config, data_root=args.data_root, split='train')
    val_dataset = NuScenesDataset(config, data_root=args.data_root, split='val')

    logger.info(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')
    logger.info(
        f'Labeled samples: train={count_labeled_samples(train_dataset)}, '
        f'val={count_labeled_samples(val_dataset)}'
    )

    if len(train_dataset) == 0:
        logger.error('No training data found! Please check data path.')
        return

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
    best_val_miou = -1.0

    if args.resume and os.path.exists(args.resume):
        logger.info(f'Resume from {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_miou = checkpoint.get('best_val_miou', -1.0)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f'Epoch {epoch}/{args.num_epochs}')

        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, logger, use_amp=config.use_amp
        )

        """val_loss, val_accuracy = evaluate(
            model, val_loader, device, logger, use_amp=config.use_amp
        )"""

        val_loss, val_accuracy, val_miou, per_class_iou = evaluate(
            model,
            val_loader,
            device,
            logger,
            num_classes=config.num_classes,
            class_names=config.class_names,
            use_amp=config.use_amp,
        )

        scheduler.step()

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'val_miou': val_miou,
                'per_class_iou': per_class_iou,
                'best_val_miou': best_val_miou,
                'config': config,
            }, checkpoint_path)
            logger.info(f'Saved best model to {checkpoint_path}')

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            """torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)"""
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'val_miou': val_miou,
                'per_class_iou': per_class_iou,
                'config': config,
            }, checkpoint_path)
            logger.info(f'Saved checkpoint to {checkpoint_path}')

    logger.info('Training completed!')


if __name__ == '__main__':
    main()
