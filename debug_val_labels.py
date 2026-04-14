import os
import numpy as np
from src.configs.bev_textclip_config import get_config
from src.dataloaders.base_dataset import NuScenesDataset

config = get_config("nuscenes")
dataset = NuScenesDataset(config, data_root="./data", split="val")

print("val size:", len(dataset))

missing = 0
existing = 0

for i in range(min(20, len(dataset))):
    info = dataset.data_list[i]
    lp = info["labels_path"]
    ok = (lp is not None) and os.path.exists(lp)
    print(f"[{i}] sample_token={info['sample_token']}")
    print("labels_path =", lp)
    print("exists =", ok)

    if ok:
        existing += 1
    else:
        missing += 1

    item = dataset[i]
    labels = item["labels"]
    labels = np.asarray(labels)

    print("labels shape:", labels.shape)
    print("labels min/max:", labels.min(), labels.max())
    print("unique (first 20):", np.unique(labels)[:20])
    print("valid pixels:", (labels != -100).sum())
    print("-" * 60)

print("existing label paths:", existing)
print("missing label paths:", missing)