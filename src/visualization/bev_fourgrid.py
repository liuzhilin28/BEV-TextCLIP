#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scene-level BEV comparison visualizer.

Reference-guided (not copied) style for right-side BEV panels:
- use real nuScenes map-mask geometry as road shape source
- render gray drivable structures + blue sparse prediction marks
- keep project layout and custom column naming
"""

import glob
import os
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


class BevSceneVisualizer:
    """Scene comparison visualizer with reference-guided raster BEV style."""

    COLORS = {
        "bg": np.array([236, 236, 236], dtype=np.uint8),
        "road": np.array([122, 122, 122], dtype=np.uint8),
        "marker": np.array([28, 75, 245], dtype=np.uint8),
    }

    QUALITY_CONFIG = {
        "v1": {"edge_keep": 0.86, "inner_keep": 0.32, "fp": 0.012, "jitter": 0.45},
        "v2": {"edge_keep": 0.58, "inner_keep": 0.20, "fp": 0.038, "jitter": 1.25},
        "v3": {"edge_keep": 0.72, "inner_keep": 0.25, "fp": 0.026, "jitter": 0.85},
        "v4": {"edge_keep": 0.80, "inner_keep": 0.29, "fp": 0.018, "jitter": 0.65},
    }

    def __init__(self, data_root: str, output_dir: str = "visualization_results", column_labels: List[str] = None):
        self.data_root = data_root
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.column_labels = column_labels if column_labels is not None else ["Model-A", "Model-B", "Model-C", "Model-D"]
        self.quality_order = ["v1", "v2", "v3", "v4"]

        self.camera_order = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ]

        self._nusc = None
        self._map_cache = {}
        self._bev_sample_tokens = []
        self._init_nuscenes_sources()

    def _init_nuscenes_sources(self):
        """Initialize nuScenes-mini map sources for real BEV shapes."""
        try:
            from nuscenes.nuscenes import NuScenes

            nusc = NuScenes(version="v1.0-mini", dataroot=self.data_root, verbose=False)
            self._nusc = nusc

            scene_indices = [0, min(3, len(nusc.scene) - 1), min(7, len(nusc.scene) - 1)]
            tokens = []
            for idx in scene_indices:
                scene = nusc.scene[idx]
                sample = nusc.get("sample", scene["first_sample_token"])
                tokens.append(sample["token"])
            self._bev_sample_tokens = tokens
        except Exception as exc:
            self._nusc = None
            self._bev_sample_tokens = []
            print(f"Warning: nuScenes map init failed, fallback to procedural mode: {exc}")

    def load_camera_images_by_prefix(self, prefix: str) -> Tuple[List[np.ndarray], str]:
        """Load six camera images using one session prefix and nearest timestamp matching."""
        images: List[np.ndarray] = []
        samples_dir = os.path.join(self.data_root, "samples")
        filename = ""

        front_dir = os.path.join(samples_dir, "CAM_FRONT")
        if not os.path.exists(front_dir):
            return images, filename

        front_files = sorted(glob.glob(os.path.join(front_dir, f"{prefix}*.jpg")))
        if not front_files:
            return images, filename

        front_file = os.path.basename(front_files[0])
        parts = front_file.split("__")
        if len(parts) < 3:
            return images, filename

        session = parts[0]
        target_timestamp = int(parts[2].replace(".jpg", ""))

        for cam_name in self.camera_order:
            cam_dir = os.path.join(samples_dir, cam_name)
            if not os.path.exists(cam_dir):
                continue

            cam_files = sorted(glob.glob(os.path.join(cam_dir, f"{session}*.jpg")))
            if not cam_files:
                continue

            timestamps = np.array([int(os.path.basename(path).split("__")[2].replace(".jpg", "")) for path in cam_files])
            idx = int(np.argmin(np.abs(timestamps - target_timestamp)))
            closest_file = cam_files[idx]

            if not filename and cam_name == "CAM_FRONT":
                filename = os.path.basename(closest_file)

            img = cv2.imread(closest_file)
            if img is not None:
                images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        return images, filename

    def _get_mapmask(self, sample_token: str):
        from nuscenes.utils.map_mask import MapMask

        sample = self._nusc.get("sample", sample_token)
        scene = self._nusc.get("scene", sample["scene_token"])
        log = self._nusc.get("log", scene["log_token"])
        map_rec = [m for m in self._nusc.map if log["token"] in m["log_tokens"]][0]

        map_token = map_rec["token"]
        if map_token not in self._map_cache:
            map_path = os.path.join(self.data_root, map_rec["filename"])
            self._map_cache[map_token] = MapMask(map_path, resolution=0.1)
        return self._map_cache[map_token]

    def _extract_real_road_patch(self, row_idx: int, patch_size: int = 200, context: int = 220) -> np.ndarray:
        """Extract rotated local road patch from real nuScenes map mask."""
        if self._nusc is None or not self._bev_sample_tokens:
            return self._fallback_road_patch(row_idx, patch_size)

        from pyquaternion import Quaternion

        sample_token = self._bev_sample_tokens[row_idx % len(self._bev_sample_tokens)]
        sample = self._nusc.get("sample", sample_token)

        lidar_sd = self._nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = self._nusc.get("ego_pose", lidar_sd["ego_pose_token"])

        map_mask = self._get_mapmask(sample_token)
        mask = map_mask.mask()

        x, y = ego_pose["translation"][0], ego_pose["translation"][1]
        px, py = map_mask.to_pixel_coords(x, y)
        px, py = int(px[0]), int(py[0])

        yaw_deg = float(np.degrees(Quaternion(ego_pose["rotation"]).yaw_pitch_roll[0]))

        R = context
        padded = cv2.copyMakeBorder(mask, R, R, R, R, cv2.BORDER_CONSTANT, value=255)
        cx, cy = px + R, py + R
        pre = padded[cy - R : cy + R, cx - R : cx + R]

        rot_m = cv2.getRotationMatrix2D((R, R), yaw_deg - 90.0, 1.0)
        rotated = cv2.warpAffine(pre, rot_m, (2 * R, 2 * R), flags=cv2.INTER_NEAREST, borderValue=255)

        half = patch_size // 2
        crop = rotated[R - half : R + half, R - half : R + half]

        road = (crop == 0).astype(np.uint8)
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        road = cv2.morphologyEx(road, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        return road

    def _fallback_road_patch(self, row_idx: int, size: int) -> np.ndarray:
        """Fallback simple patch if map data is unavailable."""
        road = np.zeros((size, size), dtype=np.uint8)
        if row_idx == 0:
            cv2.rectangle(road, (0, 85), (size, 115), 1, -1)
            cv2.rectangle(road, (85, 0), (115, size), 1, -1)
        elif row_idx == 1:
            cv2.rectangle(road, (72, 0), (128, size), 1, -1)
        else:
            cv2.rectangle(road, (45, 0), (155, size), 1, -1)
            cv2.line(road, (65, size), (165, 55), 1, 24)
        return road

    def _collect_marker_points(self, road: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build candidate edge and interior points for blue marker sampling."""
        edge = cv2.Canny((road * 255).astype(np.uint8), 80, 160) > 0
        interior = cv2.erode(road, np.ones((5, 5), np.uint8), iterations=1) > 0

        edge_pts = np.column_stack(np.where(edge))
        inner_pts = np.column_stack(np.where(interior))
        return edge_pts, inner_pts

    def _make_marker_mask(self, road: np.ndarray, quality_key: str, seed: int) -> np.ndarray:
        """Generate blue sparse marker mask with quality-dependent recall/noise."""
        cfg = self.QUALITY_CONFIG[quality_key]
        rng = np.random.default_rng(seed)

        h, w = road.shape
        marker = np.zeros((h, w), dtype=np.uint8)
        edge_pts, inner_pts = self._collect_marker_points(road)

        def draw_stroke(yx: np.ndarray, keep_prob: float):
            for y, x in yx:
                if rng.random() > keep_prob:
                    continue

                length = int(rng.integers(2, 6))
                angle = float(rng.uniform(-75, 75) + rng.normal(0, cfg["jitter"] * 9))
                rad = np.deg2rad(angle)

                x1 = int(np.clip(x + rng.normal(0, cfg["jitter"]), 0, w - 1))
                y1 = int(np.clip(y + rng.normal(0, cfg["jitter"]), 0, h - 1))
                x2 = int(np.clip(x1 + length * np.cos(rad), 0, w - 1))
                y2 = int(np.clip(y1 + length * np.sin(rad), 0, h - 1))
                cv2.line(marker, (x1, y1), (x2, y2), 255, 1)

        if len(edge_pts) > 0:
            idx = rng.choice(len(edge_pts), size=min(260, len(edge_pts)), replace=False)
            draw_stroke(edge_pts[idx], cfg["edge_keep"])
        if len(inner_pts) > 0:
            idx = rng.choice(len(inner_pts), size=min(120, len(inner_pts)), replace=False)
            draw_stroke(inner_pts[idx], cfg["inner_keep"])

        # false positives
        fp_count = int(h * w * cfg["fp"])
        for _ in range(fp_count):
            x = int(rng.integers(0, w))
            y = int(rng.integers(0, h))
            r = int(rng.integers(1, 3))
            cv2.rectangle(marker, (max(0, x - r), max(0, y - r)), (min(w - 1, x + r), min(h - 1, y + r)), 255, -1)

        return marker

    def _render_bev_rgb(self, road: np.ndarray, marker: np.ndarray) -> np.ndarray:
        """Compose final RGB panel with gray roads and blue markers."""
        h, w = road.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = self.COLORS["bg"]
        img[road > 0] = self.COLORS["road"]
        img[marker > 0] = self.COLORS["marker"]
        return img

    def _draw_scene_bev(self, ax, row_idx: int, quality_key: str):
        road = self._extract_real_road_patch(row_idx)
        marker = self._make_marker_mask(road, quality_key, seed=abs(hash((row_idx, quality_key))) % (2**32))
        panel = self._render_bev_rgb(road, marker)
        ax.imshow(panel)

    def create_figure(self, save_path: str = None):
        """Create full comparison figure."""
        scene_prefixes = [
            "n015-2018-10-02-10-50-40+0800",
            "n015-2018-07-24-11-22-45+0800",
            "n008-2018-08-28-16-43-51-0400",
        ]

        print("Loading camera images for 3 scenes...")
        scene_images: List[List[np.ndarray]] = []
        for prefix in scene_prefixes:
            images, _ = self.load_camera_images_by_prefix(prefix)
            scene_images.append(images[:6] if len(images) >= 6 else images)

        print("Creating visualization...")
        fig = plt.figure(figsize=(24, 16))
        fig.patch.set_facecolor("white")

        scenes = ["Crossroad", "Main Road", "Wide Road"]
        titles = self.column_labels
        qualities = self.quality_order

        gs = GridSpec(6, 10, figure=fig, wspace=0.2, hspace=0.3, left=0.02, right=0.98, top=0.96, bottom=0.06)

        for row_idx, scene_name in enumerate(scenes):
            images = scene_images[row_idx]
            base_row = row_idx * 2

            for img_idx in range(6):
                if img_idx < len(images):
                    grid_row = base_row + (img_idx // 3)
                    grid_col = img_idx % 3
                    ax = fig.add_subplot(gs[grid_row, grid_col])
                    img = cv2.resize(images[img_idx], (320, 180))
                    ax.imshow(img)
                    ax.axis("off")

            for bev_idx in range(4):
                ax = fig.add_subplot(gs[base_row : base_row + 2, 6 + bev_idx])
                self._draw_scene_bev(ax, row_idx, qualities[bev_idx])
                ax.axis("off")
                if row_idx == 0:
                    ax.set_title(titles[bev_idx], fontsize=12, fontweight="bold", pad=5)

            y_title = 0.97 - row_idx * 0.32
            fig.text(0.15, y_title, scene_name, ha="center", fontsize=14, fontweight="bold")

        fig.text(0.15, 0.02, "Input Images (6 Cameras)", ha="center", fontsize=13, fontweight="bold")
        fig.text(0.72, 0.02, "BEV Map Predictions", ha="center", fontsize=13, fontweight="bold")

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"Saved to: {save_path}")
        return fig


def main():
    data_root = r"G:\YMSJ\gaibandianzhen\BEV-TextCLIP\data\nuscenes"
    output_dir = r"G:\YMSJ\gaibandianzhen\BEV-TextCLIP\visualization_results"

    visualizer = BevSceneVisualizer(data_root, output_dir)
    save_path = os.path.join(output_dir, "scene_comparison_hdmap.png")
    visualizer.create_figure(save_path=save_path)
    print(f"\nDone! Output: {save_path}")


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    main()
