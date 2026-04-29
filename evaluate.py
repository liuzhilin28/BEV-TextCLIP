#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate BEV-TextCLIP checkpoints on real validation data."""

import argparse
import json
import os
import sys
from contextlib import nullcontext
from typing import Any, Dict, List

os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from src.configs.bev_textclip_config import BEVTextCLIPConfig
from src.dataloaders.base_dataset import DataCollator, NuScenesDataset
from src.models.bev_textclip import create_bev_textclip_model


def resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def fail_on_dummy_records(dataset: NuScenesDataset) -> None:
    """Refuse evaluation when the dataset fell back to generated dummy samples."""
    dummy_records = [
        record for record in getattr(dataset, 'data_list', [])
        if str(record.get('data_path', '')).startswith('dummy_')
        or str(record.get('point_cloud_path', '')).startswith('dummy_')
        or str(record.get('labels_path', '')).startswith('dummy_')
    ]
    if dummy_records:
        raise RuntimeError(
            "NuScenesDataset fell back to dummy data. Evaluation for paper metrics "
            "is refused because it would not use real validation samples."
        )


def count_labeled_records(dataset: NuScenesDataset) -> int:
    count = 0
    for record in getattr(dataset, 'data_list', []):
        labels_path = record.get('labels_path')
        if labels_path is not None and os.path.exists(labels_path):
            count += 1
    return count


def read_checkpoint(checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        return {'model_state_dict': checkpoint}
    return checkpoint


def get_config_value(config_obj: Any, name: str) -> Any:
    if config_obj is None:
        return None
    if isinstance(config_obj, dict):
        return config_obj.get(name)
    return getattr(config_obj, name, None)


def validate_checkpoint_compatibility(config: BEVTextCLIPConfig, checkpoint: Dict[str, Any]) -> None:
    checkpoint_config = checkpoint.get('config')
    checked_fields = [
        'fusion_type',
        'bev_resolution',
        'bev_channels',
        'num_classes',
        'text_encoder_type',
        'image_encoder_type',
        'point_encoder_type',
    ]
    mismatches = []
    for field_name in checked_fields:
        checkpoint_value = get_config_value(checkpoint_config, field_name)
        if checkpoint_value is None:
            continue
        current_value = getattr(config, field_name)
        if isinstance(checkpoint_value, (list, tuple)):
            values_match = list(checkpoint_value) == list(current_value)
        else:
            values_match = checkpoint_value == current_value
        if not values_match:
            mismatches.append((field_name, checkpoint_value, current_value))

    if mismatches:
        details = '; '.join(
            f"{name}: checkpoint={old!r}, config={new!r}"
            for name, old, new in mismatches
        )
        raise RuntimeError(
            "Checkpoint architecture does not match --config. "
            f"{details}. Use the matching config/checkpoint pair or retrain the checkpoint "
            "with the current config before reporting evaluation metrics."
        )


def load_checkpoint(model: torch.nn.Module, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint does not contain a model state dict.")

    if state_dict and all(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {key[len('module.'):]: value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def update_confusion_matrix(
    confmat: torch.Tensor,
    preds: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    preds = preds[mask].reshape(-1).long()
    labels = labels[mask].reshape(-1).long()

    valid = (
        (labels >= 0) &
        (labels < num_classes) &
        (preds >= 0) &
        (preds < num_classes)
    )
    preds = preds[valid]
    labels = labels[valid]

    if preds.numel() == 0:
        return confmat

    indices = labels * num_classes + preds
    confmat += torch.bincount(
        indices,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    return confmat


def compute_metrics(confmat: torch.Tensor, class_names: List[str]) -> Dict[str, Any]:
    confmat = confmat.cpu().to(torch.float64)
    tp = torch.diag(confmat)
    fp = confmat.sum(dim=0) - tp
    fn = confmat.sum(dim=1) - tp
    support = confmat.sum(dim=1)
    denom = tp + fp + fn
    valid_classes = denom > 0

    per_class_iou = torch.zeros_like(tp)
    per_class_iou[valid_classes] = tp[valid_classes] / denom[valid_classes]

    accuracy = tp.sum() / torch.clamp(confmat.sum(), min=1.0)
    miou = (
        per_class_iou[valid_classes].mean()
        if valid_classes.any()
        else torch.tensor(0.0, dtype=torch.float64)
    )

    class_iou = {}
    for idx, name in enumerate(class_names):
        value = None
        if idx < per_class_iou.numel() and valid_classes[idx]:
            value = float(per_class_iou[idx].item())
        class_iou[name] = value

    return {
        'accuracy': float(accuracy.item()),
        'mIoU': float(miou.item()),
        'class_IoU': class_iou,
        'valid_classes': [
            class_names[idx]
            for idx in range(min(len(class_names), valid_classes.numel()))
            if valid_classes[idx]
        ],
        'class_support': {
            class_names[idx]: int(support[idx].item())
            for idx in range(min(len(class_names), support.numel()))
        },
        'confusion_matrix': confmat.to(torch.int64).tolist(),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: BEVTextCLIPConfig,
) -> Dict[str, Any]:
    model.eval()
    model.use_contrastive = False

    confmat = torch.zeros(
        (config.num_classes, config.num_classes),
        dtype=torch.int64,
        device=device,
    )
    amp_enabled = config.use_amp and device.type == 'cuda'
    evaluated_batches = 0
    skipped_batches = 0
    evaluated_pixels = 0

    for batch in tqdm(dataloader, desc='Evaluating'):
        labels = batch.get('labels')
        if labels is None:
            skipped_batches += 1
            continue

        label_mask = batch.get('label_mask')
        if label_mask is None:
            label_mask = labels != config.ignore_index

        valid_count = int(label_mask.sum().item())
        if valid_count == 0:
            skipped_batches += 1
            continue

        images = batch['images'].to(device, non_blocking=True)
        intrinsics = batch['intrinsics'].to(device, non_blocking=True)
        extrinsics = batch['extrinsics'].to(device, non_blocking=True)
        point_cloud = batch['point_cloud'].to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        label_mask = label_mask.to(device, non_blocking=True)
        point_cloud_lengths = batch.get('point_cloud_lengths')
        if point_cloud_lengths is not None:
            point_cloud_lengths = point_cloud_lengths.to(device, non_blocking=True)

        autocast_context = (
            torch.amp.autocast(device_type='cuda', dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with autocast_context:
            output = model(
                images=images,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                point_cloud=point_cloud,
                point_cloud_lengths=point_cloud_lengths,
                labels=None,
            )

        predictions = output['segmentation_logits'].argmax(dim=1)
        confmat = update_confusion_matrix(
            confmat,
            predictions,
            labels,
            label_mask,
            config.num_classes,
        )
        evaluated_batches += 1
        evaluated_pixels += valid_count

    if evaluated_batches == 0:
        raise RuntimeError(
            "No labeled validation batches were evaluated. Check data_root, split, "
            "and lidarseg labels. Random labels are intentionally not generated."
        )

    metrics = compute_metrics(confmat, config.class_names)
    metrics.update({
        'evaluated_batches': evaluated_batches,
        'skipped_batches': skipped_batches,
        'evaluated_pixels': evaluated_pixels,
    })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate BEV-TextCLIP on real validation data')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output', type=str, default='results/metrics.json')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--disable_amp', action='store_true')
    args = parser.parse_args()

    config = BEVTextCLIPConfig.from_yaml(args.config)
    config.batch_size = args.batch_size
    config.use_amp = not args.disable_amp
    if config.text_encoder_type == 'local_clip':
        config.text_model_path = resolve_repo_path(config.text_model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = NuScenesDataset(config, data_root=args.data_root, split='val')
    fail_on_dummy_records(dataset)

    if len(dataset) == 0:
        raise RuntimeError(
            "Validation dataset is empty. Evaluation stopped instead of using dummy or random data."
        )

    labeled_records = count_labeled_records(dataset)
    if labeled_records == 0:
        raise RuntimeError(
            "Validation dataset has no real label files. Evaluation stopped instead of using random labels."
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=DataCollator(config),
        pin_memory=device.type == 'cuda',
        drop_last=False,
    )

    checkpoint_meta = read_checkpoint(args.checkpoint, device)
    validate_checkpoint_compatibility(config, checkpoint_meta)

    model = create_bev_textclip_model(config).to(device)
    load_checkpoint(model, checkpoint_meta)
    metrics = evaluate(model, dataloader, device, config)
    metrics.update({
        'config': args.config,
        'checkpoint': args.checkpoint,
        'data_root': args.data_root,
        'dataset': config.dataset,
        'num_classes': config.num_classes,
        'class_names': config.class_names,
        'num_val_samples': len(dataset),
        'num_labeled_val_samples': labeled_records,
        'checkpoint_epoch': to_jsonable(checkpoint_meta.get('epoch')),
        'checkpoint_val_miou': to_jsonable(checkpoint_meta.get('val_miou')),
    })

    output_path = resolve_repo_path(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"mIoU: {metrics['mIoU']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Saved metrics to {output_path}")


if __name__ == '__main__':
    main()
