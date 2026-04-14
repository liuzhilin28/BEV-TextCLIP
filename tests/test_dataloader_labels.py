#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.configs.bev_textclip_config import get_config
from src.dataloaders.base_dataset import DataCollator, DummyDataset, NuScenesDataset


def test_point_labels_are_rasterized_to_bev_grid():
    config = get_config("nuscenes")
    dataset = DummyDataset(config=config, num_samples=1)

    car_idx = config.class_names.index("car")
    truck_idx = config.class_names.index("truck")

    point_cloud = np.array(
        [
            [-19.9, -19.9, 0.0, 1.0],
            [-19.8, -19.8, 0.0, 1.0],
            [19.9, 19.9, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    point_labels = np.array([car_idx, car_idx, truck_idx], dtype=np.int64)

    bev_labels = dataset._ensure_bev_labels(point_cloud, point_labels)

    assert bev_labels.shape == config.bev_resolution
    assert bev_labels[0, 0] == car_idx
    assert bev_labels[-1, -1] == truck_idx


def test_missing_nuscenes_labels_return_empty_bev_grid(tmp_path):
    config = get_config("nuscenes")
    data_root = tmp_path / "data"
    dataset = NuScenesDataset(config=config, data_root=str(data_root), split="train")

    bev_labels = dataset._load_labels(str(tmp_path / "missing_lidarseg.bin"))

    assert bev_labels.shape == config.bev_resolution
    assert np.all(bev_labels == config.ignore_index)


def test_nuscenes_raw_labels_map_to_training_taxonomy(tmp_path):
    config = get_config("nuscenes")
    category_dir = tmp_path / "data" / "nuscenes" / "v1.0-mini"
    category_dir.mkdir(parents=True, exist_ok=True)
    category_path = category_dir / "category.json"
    category_path.write_text(
        json.dumps(
            [
                {"index": 0, "name": "noise"},
                {"index": 1, "name": "vehicle.car"},
                {"index": 2, "name": "flat.driveable_surface"},
                {"index": 3, "name": "human.pedestrian.adult"},
            ]
        ),
        encoding="utf-8",
    )

    dataset = NuScenesDataset(config=config, data_root=str(tmp_path / "data"), split="train")
    mapped = dataset._map_nuscenes_labels_to_training_ids(np.array([0, 1, 2, 3], dtype=np.int64))

    assert mapped.tolist() == [
        config.class_names.index("other"),
        config.class_names.index("car"),
        config.class_names.index("driveable_surface"),
        config.class_names.index("pedestrian"),
    ]


def test_data_collator_resizes_images_to_training_shape():
    config = get_config("nuscenes")
    dataset = DummyDataset(config=config, num_samples=2)
    collator = DataCollator(config)

    batch = collator([dataset[0], dataset[1]])

    assert batch["images"].shape == (2, 6, 3, config.image_size[0], config.image_size[1])
    assert batch["image_shapes"].tolist() == [list(config.image_size), list(config.image_size)]
    assert str(batch["label_mask"].dtype) == "torch.bool"


def test_nuscenes_version_falls_back_to_mini_when_trainval_is_missing(tmp_path):
    config = get_config("nuscenes")
    nuscenes_root = tmp_path / "data" / "nuscenes" / "v1.0-mini"
    nuscenes_root.mkdir(parents=True, exist_ok=True)
    (nuscenes_root / "attribute.json").write_text("[]", encoding="utf-8")

    dataset = NuScenesDataset(config=config, data_root=str(tmp_path / "data"), split="train")

    assert dataset.version == "v1.0-mini"


def test_nuscenes_dummy_data_list_is_returned_on_load_failure(tmp_path):
    config = get_config("nuscenes")
    dataset = NuScenesDataset(config=config, data_root=str(tmp_path / "data"), split="train")

    assert len(dataset.data_list) == 100
