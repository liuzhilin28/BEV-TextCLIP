#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch generator for reference-style BEV comparison figures.

Each output figure uses one real nuScenes sample:
- left: 6 camera views from the same sample
- right: 4 BEV panels rendered from the same sample's local map patch

The visual style is guided by the paper reference, but the geometry is always
driven by the project's own dataset instead of copying the reference image.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from nuscenes.nuscenes import NuScenes
from PIL import Image


@dataclass(frozen=True)
class PanelQuality:
    name: str
    boundary_keep: float
    lane_keep: float
    accent_keep: float
    jitter: float
    false_positive: float


class ReferenceStyleBatchVisualizer:
    """Create five independent figures that match the reference layout style."""

    REFERENCE_SAMPLE_TOKENS: Sequence[str] = (
        "bf2938e43c6f487497cda76b51bfc406",
        "fdc39b23ab4242eda6ec5e1e6574fe33",
        "8687ba92abd3406aa797115b874ebeba",
        "ed1eee39e3dd4c30a3d932e3ceaa92c2",
        "73eb876167f4419a9a6ec1a601abdcaf",
        "9c7c7d5d109c40fcaecd3c422d37b4f6",
    )

    # Single real sample used to build the "same-scene" batch variants.
    SAME_SCENE_TOKEN: str = "bf2938e43c6f487497cda76b51bfc406"

    # Legacy fallback samples if the preferred same-scene token is unavailable locally.
    SAMPLE_TOKENS: Sequence[str] = (
        "a19a80c905674faab7203a3a4e0f5246",
        "b6c420c3a5bd4a219b1cb82ee5ea0aa7",
        "8057958576034e51bac9bdc5740128d5",
        "2578329fc3ae484bb23ef766808f4be5",
        "61a7bd24f88a46c2963280d8b13ac675",
    )

    CAMERA_ORDER = (
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
    )

    PANEL_CONFIG: Sequence[PanelQuality] = (
        PanelQuality("Tag-A", 1.00, 1.00, 1.00, 0.00, 0.0000),
        PanelQuality("Tag-B", 0.72, 0.58, 0.66, 0.20, 0.00005),
        PanelQuality("Tag-C", 0.86, 0.76, 0.84, 0.09, 0.00002),
        PanelQuality("Tag-D", 0.95, 0.90, 0.92, 0.03, 0.0000),
    )

    COLORS: Dict[str, Tuple[int, int, int]] = {
        "boundary": (10, 69, 232),
        "lane": (30, 99, 255),
        "red": (224, 32, 32),
        "green": (31, 175, 75),
    }

    def __init__(self, data_root: str, output_dir: str):
        self.data_root = data_root
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.nusc = NuScenes(version="v1.0-mini", dataroot=data_root, verbose=False)
        self._map_cache: Dict[str, object] = {}
        self._vector_map_cache: Dict[str, object] = {}
        self._shape_cache: Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]] = {}

    def _available_sample_tokens(self) -> List[str]:
        sample_table = self.nusc._token2ind["sample"]
        tokens = [token for token in self.SAMPLE_TOKENS if token in sample_table]
        if not tokens:
            raise RuntimeError("No configured sample tokens were found in the local nuScenes dataset.")
        return tokens

    def _load_camera_images(self, sample_token: str) -> List[np.ndarray]:
        sample = self.nusc.get("sample", sample_token)
        images: List[np.ndarray] = []
        for cam_name in self.CAMERA_ORDER:
            sample_data = self.nusc.get("sample_data", sample["data"][cam_name])
            img_path = os.path.join(self.data_root, sample_data["filename"])
            image = cv2.imread(img_path)
            if image is None:
                image = np.full((180, 320, 3), 220, dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        return images

    def _camera_montage(self, images: Sequence[np.ndarray], cell_size: Tuple[int, int] = (192, 108)) -> np.ndarray:
        width, height = cell_size
        resized = [cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA) for img in images]
        top = np.hstack(resized[:3])
        bottom = np.hstack(resized[3:6])
        return np.vstack([top, bottom])

    def _densify_polyline(self, points: Sequence[Tuple[float, float]], samples_per_seg: int = 24) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        if len(pts) < 2:
            return pts
        dense: List[np.ndarray] = []
        for idx in range(len(pts) - 1):
            p0 = pts[idx]
            p1 = pts[idx + 1]
            t = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False, dtype=np.float32)[:, None]
            dense.append(p0[None, :] * (1.0 - t) + p1[None, :] * t)
        dense.append(pts[-1][None, :])
        return np.vstack(dense)

    def _smooth_polyline(self, polyline: np.ndarray, window: int = 9) -> np.ndarray:
        pts = polyline.astype(np.float32)
        if len(pts) < max(5, window):
            return pts
        kernel = np.ones(window, dtype=np.float32) / float(window)
        pad = window // 2
        padded = np.pad(pts, ((pad, pad), (0, 0)), mode="edge")
        smoothed = np.stack(
            [
                np.convolve(padded[:, 0], kernel, mode="valid"),
                np.convolve(padded[:, 1], kernel, mode="valid"),
            ],
            axis=1,
        )
        return smoothed.astype(np.float32)

    def _polyline_length(self, polyline: np.ndarray) -> float:
        if len(polyline) < 2:
            return 0.0
        diffs = np.diff(polyline.astype(np.float32), axis=0)
        return float(np.linalg.norm(diffs, axis=1).sum())

    def _get_mapmask(self, sample_token: str):
        from nuscenes.utils.map_mask import MapMask

        sample = self.nusc.get("sample", sample_token)
        scene = self.nusc.get("scene", sample["scene_token"])
        log = self.nusc.get("log", scene["log_token"])
        map_rec = [record for record in self.nusc.map if log["token"] in record["log_tokens"]][0]

        map_token = map_rec["token"]
        if map_token not in self._map_cache:
            map_path = os.path.join(self.data_root, map_rec["filename"])
            self._map_cache[map_token] = MapMask(map_path, resolution=0.1)
        return self._map_cache[map_token]

    def _sample_pose(self, sample_token: str) -> Tuple[float, float, float, str]:
        from pyquaternion import Quaternion

        sample = self.nusc.get("sample", sample_token)
        lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        scene = self.nusc.get("scene", sample["scene_token"])
        log = self.nusc.get("log", scene["log_token"])
        yaw = float(Quaternion(ego_pose["rotation"]).yaw_pitch_roll[0])
        return float(ego_pose["translation"][0]), float(ego_pose["translation"][1]), yaw, str(log["location"])

    def _get_vector_map(self, sample_token: str):
        from nuscenes.map_expansion.map_api import NuScenesMap

        _, _, _, location = self._sample_pose(sample_token)
        if location not in self._vector_map_cache:
            self._vector_map_cache[location] = NuScenesMap(dataroot=self.data_root, map_name=location)
        return self._vector_map_cache[location]

    def _world_to_canvas(self, coords: np.ndarray, center_xy: Tuple[float, float], yaw: float, canvas_size: int = 236, patch_radius_m: float = 22.0) -> np.ndarray:
        pts = coords.astype(np.float32).copy()
        pts[:, 0] -= center_xy[0]
        pts[:, 1] -= center_xy[1]

        sin_yaw = float(np.sin(yaw))
        cos_yaw = float(np.cos(yaw))
        lateral = -sin_yaw * pts[:, 0] + cos_yaw * pts[:, 1]
        longitudinal = cos_yaw * pts[:, 0] + sin_yaw * pts[:, 1]

        scale = (canvas_size * 0.48) / max(1e-6, patch_radius_m)
        canvas_x = canvas_size * 0.5 - lateral * scale
        canvas_y = canvas_size * 0.60 - longitudinal * scale
        return np.column_stack([canvas_x, canvas_y]).astype(np.float32)

    def _extract_real_road_patch(self, sample_token: str, patch_size: int = 232, context: int = 268) -> np.ndarray:
        from pyquaternion import Quaternion

        sample = self.nusc.get("sample", sample_token)
        lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])

        map_mask = self._get_mapmask(sample_token)
        mask = map_mask.mask()

        px, py = map_mask.to_pixel_coords(ego_pose["translation"][0], ego_pose["translation"][1])
        px, py = int(px[0]), int(py[0])
        yaw_deg = float(np.degrees(Quaternion(ego_pose["rotation"]).yaw_pitch_roll[0]))

        radius = context
        padded = cv2.copyMakeBorder(mask, radius, radius, radius, radius, cv2.BORDER_CONSTANT, value=255)
        cx, cy = px + radius, py + radius
        pre = padded[cy - radius : cy + radius, cx - radius : cx + radius]

        rot_m = cv2.getRotationMatrix2D((radius, radius), yaw_deg - 90.0, 1.0)
        rotated = cv2.warpAffine(pre, rot_m, (2 * radius, 2 * radius), flags=cv2.INTER_NEAREST, borderValue=255)

        half = patch_size // 2
        crop_center_y = radius + 22
        crop = rotated[crop_center_y - half : crop_center_y + half, radius - half : radius + half]
        road = (crop == 0).astype(np.uint8) * 255
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        road = cv2.morphologyEx(road, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        return self._focus_main_geometry(road)

    def _focus_main_geometry(self, road: np.ndarray) -> np.ndarray:
        """Keep the ego-related main road structure and suppress detached clutter."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((road > 0).astype(np.uint8), connectivity=8)
        if num_labels <= 1:
            return road

        h, w = road.shape
        center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        chosen_label = 0
        chosen_score = None
        component_info = []

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 120:
                continue
            ys, xs = np.where(labels == label)
            if len(xs) == 0:
                continue
            centroid = np.array([xs.mean(), ys.mean()], dtype=np.float32)
            dist = float(np.linalg.norm(centroid - center))
            score = dist - area * 0.015
            component_info.append((label, area, centroid, dist, score))
            if chosen_score is None or score < chosen_score:
                chosen_score = score
                chosen_label = label

        if chosen_label == 0:
            return road

        main_mask = (labels == chosen_label).astype(np.uint8) * 255
        support = cv2.dilate(main_mask, np.ones((61, 61), np.uint8), iterations=1)
        refined = np.zeros_like(road)
        for label, area, centroid, dist, score in component_info:
            component_mask = (labels == label).astype(np.uint8) * 255
            intersects_support = cv2.countNonZero(cv2.bitwise_and(component_mask, support)) > 0
            is_center_relevant = dist <= 78.0
            is_large = area >= 260
            if label == chosen_label or intersects_support or (is_center_relevant and is_large):
                refined = cv2.bitwise_or(refined, component_mask)

        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        return refined

    def _skeletonize(self, mask: np.ndarray) -> np.ndarray:
        try:
            from skimage.morphology import skeletonize

            return skeletonize(mask > 0).astype(np.uint8) * 255
        except Exception:
            skeleton = np.zeros_like(mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            current = mask.copy()
            while cv2.countNonZero(current) > 0:
                eroded = cv2.erode(current, kernel)
                opened = cv2.dilate(eroded, kernel)
                skeleton = cv2.bitwise_or(skeleton, cv2.subtract(current, opened))
                current = eroded
            return skeleton

    def _smooth_mask(self, mask: np.ndarray, scale: int = 4) -> np.ndarray:
        up = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        up = cv2.GaussianBlur(up, (0, 0), sigmaX=2.0, sigmaY=2.0)
        return (up > 100).astype(np.uint8) * 255

    def _edge_layer(self, mask: np.ndarray, erode_size: int, out_shape: Tuple[int, int]) -> np.ndarray:
        work = mask.copy()
        if erode_size > 0:
            work = cv2.erode(work, np.ones((erode_size, erode_size), np.uint8), iterations=1)
        if cv2.countNonZero(work) == 0:
            return np.zeros(out_shape, dtype=np.uint8)

        smooth = self._smooth_mask(work, scale=4)
        edge = cv2.Canny(smooth, 80, 160)
        edge[:10, :] = 0
        edge[-10:, :] = 0
        edge[:, :10] = 0
        edge[:, -10:] = 0
        edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=1)
        edge = cv2.resize(edge, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_AREA)
        return (edge > 30).astype(np.uint8) * 255

    def _contours_to_polylines(
        self,
        mask: np.ndarray,
        closed: bool,
        min_area: float = 80.0,
        min_length: float = 24.0,
        simplify: float = 0.006,
    ) -> List[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        polylines: List[np.ndarray] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            arc = float(cv2.arcLength(contour, closed))
            if area < min_area and arc < min_length:
                continue
            epsilon = max(0.8, simplify * arc)
            approx = cv2.approxPolyDP(contour, epsilon, closed).reshape(-1, 2)
            if len(approx) < 2:
                continue
            if closed:
                approx = np.vstack([approx, approx[:1]])
            dense = self._densify_polyline(approx, samples_per_seg=18)
            polylines.append(self._smooth_polyline(dense, window=9))
        return polylines

    def _distance_band_masks(self, road: np.ndarray) -> List[np.ndarray]:
        binary = (road > 0).astype(np.uint8)
        if cv2.countNonZero(binary) == 0:
            return []

        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        max_dist = float(dist.max())
        if max_dist < 2.5:
            return []

        bands: List[np.ndarray] = []
        for ratio in (0.26, 0.46):
            threshold = max(2.5, max_dist * ratio)
            band = (dist >= threshold).astype(np.uint8) * 255
            band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
            band = cv2.morphologyEx(band, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            if cv2.countNonZero(band) >= 110:
                bands.append(band)
        return bands

    def _neighbor_count(self, skeleton: np.ndarray) -> np.ndarray:
        binary = (skeleton > 0).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        counts = cv2.filter2D(binary, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        return counts - binary

    def _ordered_component_points(self, component_mask: np.ndarray) -> Optional[np.ndarray]:
        ys, xs = np.where(component_mask > 0)
        if len(xs) < 2:
            return None

        points = {(int(x), int(y)) for x, y in zip(xs, ys)}
        neighbors: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for x, y in points:
            linked: List[Tuple[int, int]] = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    q = (x + dx, y + dy)
                    if q in points:
                        linked.append(q)
            neighbors[(x, y)] = linked

        endpoints = [pt for pt, linked in neighbors.items() if len(linked) == 1]
        start = endpoints[0] if endpoints else next(iter(points))

        ordered: List[Tuple[int, int]] = [start]
        prev: Optional[Tuple[int, int]] = None
        current = start
        visited = {start}
        while True:
            next_candidates = [pt for pt in neighbors[current] if pt != prev]
            unvisited = [pt for pt in next_candidates if pt not in visited]
            if unvisited:
                nxt = unvisited[0]
            elif next_candidates:
                nxt = next_candidates[0]
            else:
                break
            if nxt in visited and len(endpoints) > 0:
                break
            ordered.append(nxt)
            visited.add(nxt)
            prev, current = current, nxt

        if len(ordered) < 2:
            return None
        return np.asarray(ordered, dtype=np.float32)

    def _skeleton_segments(self, road: np.ndarray) -> List[np.ndarray]:
        inner = cv2.erode(road, np.ones((5, 5), np.uint8), iterations=1)
        if cv2.countNonZero(inner) == 0:
            inner = road

        skeleton = self._skeletonize((inner > 0).astype(np.uint8) * 255)
        skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        neighbor_count = self._neighbor_count(skeleton)
        branch_mask = np.where((skeleton > 0) & (neighbor_count > 2), 255, 0).astype(np.uint8)
        segment_mask = cv2.bitwise_and(skeleton, cv2.bitwise_not(branch_mask))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((segment_mask > 0).astype(np.uint8), connectivity=8)
        center = np.array([road.shape[1] / 2.0, road.shape[0] / 2.0], dtype=np.float32)
        scored: List[Tuple[float, np.ndarray]] = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 10:
                continue
            component = np.where(labels == label, 255, 0).astype(np.uint8)
            ordered = self._ordered_component_points(component)
            if ordered is None or len(ordered) < 8:
                continue
            dense = self._densify_polyline(ordered, samples_per_seg=6)
            smooth = self._smooth_polyline(dense, window=7)
            length = self._polyline_length(smooth)
            if length < 28.0:
                continue
            centroid = smooth.mean(axis=0)
            score = length - float(np.linalg.norm(centroid - center)) * 0.55
            scored.append((score, smooth))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [polyline for _, polyline in scored[:8]]

    def _polyline_tangents(self, polyline: np.ndarray) -> np.ndarray:
        pts = polyline.astype(np.float32)
        if len(pts) < 2:
            return np.zeros_like(pts)
        prev_pts = np.vstack([pts[:1], pts[:-1]])
        next_pts = np.vstack([pts[1:], pts[-1:]])
        tangents = next_pts - prev_pts
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-6, None)
        return tangents / norms

    def _estimate_half_width(self, polyline: np.ndarray, dist_map: np.ndarray) -> float:
        pts = np.round(polyline).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, dist_map.shape[1] - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, dist_map.shape[0] - 1)
        values = dist_map[pts[:, 1], pts[:, 0]]
        if len(values) == 0:
            return 6.0
        return float(np.clip(np.median(values) * 0.95, 4.0, 18.0))

    def _offset_polyline(self, polyline: np.ndarray, offset: float, out_shape: Tuple[int, int]) -> np.ndarray:
        tangents = self._polyline_tangents(polyline)
        normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
        offset_pts = polyline.astype(np.float32) + normals * float(offset)
        offset_pts[:, 0] = np.clip(offset_pts[:, 0], 0.0, out_shape[1] - 1)
        offset_pts[:, 1] = np.clip(offset_pts[:, 1], 0.0, out_shape[0] - 1)
        return self._smooth_polyline(offset_pts, window=7)

    def _clip_point(self, pt: np.ndarray, shape: Tuple[int, int]) -> Tuple[float, float]:
        x = float(np.clip(pt[0], 0.0, shape[1] - 1))
        y = float(np.clip(pt[1], 0.0, shape[0] - 1))
        return x, y

    def _geometry_to_polylines(self, geometry, closed: bool = False) -> List[np.ndarray]:
        from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon

        polylines: List[np.ndarray] = []
        if geometry.is_empty:
            return polylines

        if isinstance(geometry, LineString):
            coords = np.asarray(geometry.coords, dtype=np.float32)
            if len(coords) >= 2:
                polylines.append(coords)
            return polylines

        if isinstance(geometry, Polygon):
            coords = np.asarray(geometry.exterior.coords, dtype=np.float32)
            if not closed and len(coords) > 1:
                coords = coords[:-1]
            if len(coords) >= 2:
                polylines.append(coords)
            for interior in geometry.interiors:
                hole = np.asarray(interior.coords, dtype=np.float32)
                if not closed and len(hole) > 1:
                    hole = hole[:-1]
                if len(hole) >= 2:
                    polylines.append(hole)
            return polylines

        if isinstance(geometry, (MultiLineString, MultiPolygon, GeometryCollection)):
            for item in geometry.geoms:
                polylines.extend(self._geometry_to_polylines(item, closed=closed))
        return polylines

    def _normalize_polylines(
        self,
        polylines: List[np.ndarray],
        closed: bool,
        simplify_px: float = 1.0,
        min_arc: float = 12.0,
    ) -> List[np.ndarray]:
        normalized: List[np.ndarray] = []
        for polyline in polylines:
            pts = polyline.astype(np.float32)
            if len(pts) < 2:
                continue
            if closed and np.linalg.norm(pts[0] - pts[-1]) > 1e-3:
                pts = np.vstack([pts, pts[:1]])
            arc = self._polyline_length(pts)
            if arc < min_arc * 0.8:
                continue
            epsilon = max(0.6, simplify_px)
            approx = cv2.approxPolyDP(np.round(pts).astype(np.float32), epsilon, closed).reshape(-1, 2).astype(np.float32)
            if closed and len(approx) >= 2:
                approx = np.vstack([approx, approx[:1]])
            dense = self._densify_polyline(approx, samples_per_seg=12 if closed else 8)
            smooth = self._smooth_polyline(dense, window=5)
            if self._polyline_length(smooth) >= min_arc:
                normalized.append(smooth)
        return normalized

    def _polyline_center_score(self, polyline: np.ndarray, canvas_size: int = 236) -> float:
        center = np.array([canvas_size * 0.5, canvas_size * 0.58], dtype=np.float32)
        centroid = polyline.astype(np.float32).mean(axis=0)
        length = self._polyline_length(polyline)
        dist = float(np.linalg.norm(centroid - center))
        return length - dist * 0.75

    def _polyline_accent_score(self, polyline: np.ndarray, canvas_size: int = 236) -> float:
        center = np.array([canvas_size * 0.5, canvas_size * 0.56], dtype=np.float32)
        centroid = polyline.astype(np.float32).mean(axis=0)
        length = self._polyline_length(polyline)
        dist = float(np.linalg.norm(centroid - center))
        return length * 0.38 - dist * 1.25

    def _select_diverse_polylines(
        self,
        polylines: Sequence[np.ndarray],
        limit: int,
        thickness: int,
        score_fn,
        min_new_pixels: int,
        canvas_size: int = 236,
    ) -> List[np.ndarray]:
        if limit <= 0:
            return []
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        selected: List[np.ndarray] = []
        for polyline in sorted(polylines, key=score_fn, reverse=True):
            pts = np.round(polyline.astype(np.float32)).astype(np.int32).reshape(-1, 1, 2)
            if len(pts) < 2:
                continue
            layer = np.zeros_like(canvas)
            cv2.polylines(layer, [pts], False, 255, thickness, lineType=cv2.LINE_AA)
            new_pixels = cv2.countNonZero(cv2.bitwise_and(layer, cv2.bitwise_not(canvas)))
            if selected and new_pixels < min_new_pixels:
                continue
            selected.append(polyline)
            canvas = cv2.bitwise_or(canvas, layer)
            if len(selected) >= limit:
                break
        return selected

    def _polygon_to_stripes(self, polygon, stripe_count: int = 4) -> List[np.ndarray]:
        from shapely.geometry import LineString

        if polygon.is_empty or polygon.area < 18.0:
            return []

        rect = polygon.minimum_rotated_rectangle
        rect_pts = np.asarray(rect.exterior.coords, dtype=np.float32)[:4]
        if len(rect_pts) < 4:
            return []

        edges = [rect_pts[(idx + 1) % 4] - rect_pts[idx] for idx in range(4)]
        lengths = [float(np.linalg.norm(edge)) for edge in edges]
        long_idx = int(np.argmax(lengths))
        short_idx = (long_idx + 1) % 4
        long_vec = edges[long_idx]
        short_vec = edges[short_idx]
        long_len = max(1e-6, float(np.linalg.norm(long_vec)))
        short_len = max(1e-6, float(np.linalg.norm(short_vec)))
        long_unit = long_vec / long_len
        short_unit = short_vec / short_len
        center = np.asarray(polygon.centroid.coords[0], dtype=np.float32)

        stripes: List[np.ndarray] = []
        for offset in np.linspace(-0.36, 0.36, stripe_count):
            c = center + short_unit * (offset * short_len * 0.65)
            line = LineString([tuple(c - long_unit * long_len * 0.55), tuple(c + long_unit * long_len * 0.55)])
            clipped = polygon.intersection(line)
            stripes.extend(self._geometry_to_polylines(clipped, closed=False))
        return stripes

    def _vector_record_to_geometry(self, nusc_map, layer_name: str, token: str):
        from shapely.ops import unary_union

        record = nusc_map.get(layer_name, token)
        if "polygon_tokens" in record and record["polygon_tokens"]:
            polygons = [nusc_map.extract_polygon(poly_token) for poly_token in record["polygon_tokens"] if poly_token]
            polygons = [polygon for polygon in polygons if polygon is not None]
            if polygons:
                return unary_union(polygons)
        if "line_token" in record and record["line_token"]:
            return nusc_map.extract_line(record["line_token"])
        if "polygon_token" in record and record["polygon_token"]:
            return nusc_map.extract_polygon(record["polygon_token"])
        return None

    def _safe_geometry(self, geometry):
        if geometry is None:
            return None
        try:
            if geometry.is_empty:
                return geometry
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
        except Exception:
            try:
                geometry = geometry.buffer(0)
            except Exception:
                return None
        return geometry

    def _safe_intersection(self, geometry, patch_box):
        geometry = self._safe_geometry(geometry)
        if geometry is None:
            return None
        try:
            clipped = geometry.intersection(patch_box)
        except Exception:
            geometry = self._safe_geometry(geometry)
            if geometry is None:
                return None
            try:
                clipped = geometry.intersection(patch_box)
            except Exception:
                return None
        return self._safe_geometry(clipped)

    def _accent_anchor(self, road: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        ys, xs = np.where(road > 0)
        if len(xs) < 40:
            return None

        preferred = np.array([road.shape[1] * 0.5, road.shape[0] * 0.46], dtype=np.float32)
        points = np.column_stack([xs, ys]).astype(np.float32)
        dists = np.sum((points - preferred[None, :]) ** 2, axis=1)
        center = points[int(np.argmin(dists))]

        local = points[np.sum((points - center[None, :]) ** 2, axis=1) <= 32.0**2]
        if len(local) < 30:
            local = points

        centered = local - local.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(1, len(centered) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        tangent = eigvecs[:, int(np.argmax(eigvals))].astype(np.float32)
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-6:
            tangent = np.array([1.0, 0.0], dtype=np.float32)
        else:
            tangent /= norm
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)

        dist = cv2.distanceTransform((road > 0).astype(np.uint8), cv2.DIST_L2, 5)
        width = float(np.clip(dist[int(center[1]), int(center[0])] * 1.75, 10.0, 26.0))
        return center, tangent, normal, width

    def _build_accent_polylines(self, road: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        anchor = self._accent_anchor(road)
        if anchor is None:
            return [], []

        center, tangent, normal, width = anchor
        red_segments: List[np.ndarray] = []
        green_segments: List[np.ndarray] = []

        red_half = width * 0.92
        for offset in (-0.8, 0.0, 0.8):
            c = center + tangent * (offset * width * 0.55)
            p0 = np.array(self._clip_point(c - normal * red_half, road.shape), dtype=np.float32)
            p1 = np.array(self._clip_point(c + normal * red_half, road.shape), dtype=np.float32)
            red_segments.append(self._densify_polyline([tuple(p0), tuple(p1)], samples_per_seg=10))

        green_half = width * 0.78
        for offset in (-0.7, 0.7):
            c = center + normal * (offset * width * 0.9)
            p0 = np.array(self._clip_point(c - tangent * green_half, road.shape), dtype=np.float32)
            p1 = np.array(self._clip_point(c + tangent * green_half, road.shape), dtype=np.float32)
            green_segments.append(self._densify_polyline([tuple(p0), tuple(p1)], samples_per_seg=10))

        return red_segments, green_segments

    def _mask_band_polylines(
        self,
        mask: np.ndarray,
        erode_size: int = 0,
        min_area: float = 70.0,
        min_length: float = 26.0,
        simplify: float = 0.007,
    ) -> List[np.ndarray]:
        band = mask.copy()
        if erode_size > 0:
            band = cv2.erode(band, np.ones((erode_size, erode_size), np.uint8), iterations=1)
        if cv2.countNonZero(band) == 0:
            return []
        edge = self._edge_layer(band, erode_size=0, out_shape=band.shape)
        return self._contours_to_polylines(edge, closed=False, min_area=min_area, min_length=min_length, simplify=simplify)

    def _localize_polylines(
        self,
        polylines: Sequence[np.ndarray],
        center: np.ndarray,
        radius: float,
        min_length: float = 10.0,
    ) -> List[np.ndarray]:
        localized: List[np.ndarray] = []
        radius_sq = float(radius * radius)
        center = center.astype(np.float32)
        for polyline in polylines:
            pts = polyline.astype(np.float32)
            inside = np.sum((pts - center[None, :]) ** 2, axis=1) <= radius_sq
            current: List[np.ndarray] = []
            for keep, pt in zip(inside, pts):
                if keep:
                    current.append(pt)
                elif len(current) >= 2:
                    segment = np.asarray(current, dtype=np.float32)
                    if self._polyline_length(segment) >= min_length:
                        localized.append(segment)
                    current = []
                else:
                    current = []
            if len(current) >= 2:
                segment = np.asarray(current, dtype=np.float32)
                if self._polyline_length(segment) >= min_length:
                    localized.append(segment)
        return localized

    def _fragment_mask(self, mask: np.ndarray, keep: float, jitter: float, rng: np.random.Generator) -> np.ndarray:
        if keep >= 0.995:
            return mask.copy()

        field = rng.random(mask.shape).astype(np.float32)
        field = cv2.GaussianBlur(field, (0, 0), sigmaX=max(0.7, 1.0 + jitter), sigmaY=max(0.7, 1.0 + jitter))
        threshold = float(np.quantile(field, min(max(keep, 0.03), 0.98)))
        kept = (field <= threshold).astype(np.uint8) * 255
        fragmented = cv2.bitwise_and(mask, kept)
        return cv2.morphologyEx(fragmented, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    def _build_scene_polylines(self, sample_token: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        from shapely.geometry import box

        center_x, center_y, yaw, _ = self._sample_pose(sample_token)
        nusc_map = self._get_vector_map(sample_token)
        patch_radius_m = 30.0
        patch_box = box(center_x - patch_radius_m, center_y - patch_radius_m, center_x + patch_radius_m, center_y + patch_radius_m)
        records = nusc_map.get_records_in_patch(
            (center_x - patch_radius_m, center_y - patch_radius_m, center_x + patch_radius_m, center_y + patch_radius_m),
            layer_names=[
                "road_segment",
                "lane",
                "lane_divider",
                "road_divider",
                "ped_crossing",
            ],
            mode="intersect",
        )

        road = self._extract_real_road_patch(sample_token, patch_size=320, context=392)
        road = cv2.resize(road, (236, 236), interpolation=cv2.INTER_NEAREST)
        boundary_polylines: List[np.ndarray] = []
        lane_polylines: List[np.ndarray] = []
        red_segments: List[np.ndarray] = []
        green_segments: List[np.ndarray] = []
        to_canvas = lambda poly: self._world_to_canvas(poly, (center_x, center_y), yaw, patch_radius_m=patch_radius_m)

        lane_boundary_candidates: List[np.ndarray] = []
        for token in records["lane"]:
            geometry = self._vector_record_to_geometry(nusc_map, "lane", token)
            if geometry is None:
                continue
            clipped = self._safe_intersection(geometry, patch_box)
            if clipped is None or clipped.is_empty:
                continue
            for polyline in self._geometry_to_polylines(clipped.boundary, closed=False):
                lane_boundary_candidates.append(to_canvas(polyline))

        road_segment_candidates: List[np.ndarray] = []
        for token in records["road_segment"]:
            geometry = self._vector_record_to_geometry(nusc_map, "road_segment", token)
            if geometry is None:
                continue
            clipped = self._safe_intersection(geometry, patch_box)
            if clipped is None or clipped.is_empty:
                continue
            for polyline in self._geometry_to_polylines(clipped.boundary, closed=False):
                road_segment_candidates.append(to_canvas(polyline))

        boundary_polylines.extend(lane_boundary_candidates)
        lane_boundary_length = sum(self._polyline_length(poly) for poly in lane_boundary_candidates)
        if len(boundary_polylines) < 4 or lane_boundary_length < 360.0:
            boundary_polylines.extend(road_segment_candidates)
        if len(boundary_polylines) < 4 or lane_boundary_length < 260.0:
            boundary_polylines.extend(self._mask_band_polylines(road, erode_size=0, min_area=140.0, min_length=72.0, simplify=0.008))

        divider_segments: List[np.ndarray] = []
        for layer_name in ("lane_divider", "road_divider"):
            for token in records[layer_name]:
                geometry = self._vector_record_to_geometry(nusc_map, layer_name, token)
                if geometry is None:
                    continue
                clipped = self._safe_intersection(geometry, patch_box)
                if clipped is None or clipped.is_empty:
                    continue
                for polyline in self._geometry_to_polylines(clipped, closed=False):
                    canvas_pts = to_canvas(polyline)
                    divider_segments.append(canvas_pts)
        green_segments.extend(divider_segments)

        for token in records["ped_crossing"]:
            geometry = self._vector_record_to_geometry(nusc_map, "ped_crossing", token)
            if geometry is None:
                continue
            clipped = self._safe_intersection(geometry, patch_box)
            if clipped is None or clipped.is_empty:
                continue
            for polygon_pts in self._geometry_to_polylines(clipped, closed=True):
                if len(polygon_pts) < 4:
                    continue
                canvas_polygon = to_canvas(polygon_pts)
                try:
                    from shapely.geometry import Polygon

                    polygon = Polygon(canvas_polygon)
                except Exception:
                    continue
                red_segments.extend(self._polygon_to_stripes(polygon, stripe_count=3))

        if len(green_segments) < 1:
            fallback_center = np.array([118.0, 124.0], dtype=np.float32)
            fallback_dividers = self._localize_polylines(self._skeleton_segments(road), fallback_center, radius=86.0, min_length=18.0)
            green_segments.extend(fallback_dividers[:2])

        boundary_polylines = self._normalize_polylines(boundary_polylines, closed=False, simplify_px=1.6, min_arc=26.0)
        lane_polylines = self._normalize_polylines(lane_polylines, closed=False, simplify_px=1.0, min_arc=20.0)
        red_segments = self._normalize_polylines(red_segments, closed=False, simplify_px=0.9, min_arc=12.0)
        green_segments = self._normalize_polylines(green_segments, closed=False, simplify_px=0.9, min_arc=12.0)

        boundary_polylines = self._select_diverse_polylines(
            boundary_polylines,
            limit=8,
            thickness=2,
            score_fn=self._polyline_center_score,
            min_new_pixels=120,
        )
        lane_polylines = self._select_diverse_polylines(
            lane_polylines,
            limit=0,
            thickness=1,
            score_fn=self._polyline_center_score,
            min_new_pixels=40,
        )
        red_segments = self._select_diverse_polylines(
            red_segments,
            limit=4,
            thickness=1,
            score_fn=self._polyline_accent_score,
            min_new_pixels=28,
        )
        green_segments = self._select_diverse_polylines(
            green_segments,
            limit=4,
            thickness=1,
            score_fn=self._polyline_accent_score,
            min_new_pixels=28,
        )
        return boundary_polylines, lane_polylines, red_segments, green_segments

    def _tight_crop_and_pad(self, rgb: np.ndarray, target_size: Tuple[int, int], mode: str = "contain") -> np.ndarray:
        active = np.any(rgb < 250, axis=2)
        ys, xs = np.where(active)
        if len(xs) > 0:
            margin = 8
            x0 = max(0, xs.min() - margin)
            y0 = max(0, ys.min() - margin)
            x1 = min(rgb.shape[1], xs.max() + margin + 1)
            y1 = min(rgb.shape[0], ys.max() + margin + 1)
            rgb = rgb[y0:y1, x0:x1]

        target_w, target_h = target_size
        if mode == "cover":
            scale = max(target_w / max(1, rgb.shape[1]), target_h / max(1, rgb.shape[0]))
        else:
            scale = min(target_w / max(1, rgb.shape[1]), target_h / max(1, rgb.shape[0]))
        resized = cv2.resize(
            rgb,
            (max(1, int(rgb.shape[1] * scale)), max(1, int(rgb.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
        if mode == "cover" and (resized.shape[1] > target_w or resized.shape[0] > target_h):
            start_x = max(0, (resized.shape[1] - target_w) // 2)
            start_y = max(0, (resized.shape[0] - target_h) // 2)
            resized = resized[start_y : start_y + target_h, start_x : start_x + target_w]
        off_x = max(0, (target_w - resized.shape[1]) // 2)
        off_y = max(0, (target_h - resized.shape[0]) // 2)
        canvas[off_y : off_y + resized.shape[0], off_x : off_x + resized.shape[1]] = resized
        return canvas

    def _fragment_polyline(self, polyline: np.ndarray, keep: float, jitter: float, rng: np.random.Generator) -> List[np.ndarray]:
        pts = polyline.astype(np.float32).copy()
        pts += rng.normal(0.0, jitter, size=pts.shape).astype(np.float32)
        if keep >= 0.995:
            return [pts]

        keep_mask = rng.random(len(pts)) <= keep
        keep_mask = np.convolve(keep_mask.astype(np.int16), np.ones(7, dtype=np.int16), mode="same") >= 4
        segments: List[np.ndarray] = []
        current: List[np.ndarray] = []
        for idx, pt in enumerate(pts):
            if keep_mask[idx]:
                current.append(pt)
            elif len(current) >= 2:
                segments.append(np.array(current, dtype=np.float32))
                current = []
            else:
                current = []
        if len(current) >= 2:
            segments.append(np.array(current, dtype=np.float32))
        return segments

    def _draw_polyline_group(
        self,
        canvas: np.ndarray,
        polylines: Sequence[np.ndarray],
        color: Tuple[int, int, int],
        linewidth: int,
        keep: float,
        jitter: float,
        rng: np.random.Generator,
    ) -> None:
        for polyline in polylines:
            for segment in self._fragment_polyline(polyline, keep, jitter, rng):
                pts = np.round(segment).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts], False, color, linewidth, lineType=cv2.LINE_AA)

    def _render_panel_rgb(self, sample_token: str, quality: PanelQuality, seed: int, size: Tuple[int, int] = (124, 232)) -> np.ndarray:
        rng = np.random.default_rng(seed)
        if sample_token not in self._shape_cache:
            self._shape_cache[sample_token] = self._build_scene_polylines(sample_token)
        boundaries, lanes, red_segments, green_segments = self._shape_cache[sample_token]

        panel = np.full((236, 236, 3), 255, dtype=np.uint8)
        self._draw_polyline_group(panel, boundaries, self.COLORS["boundary"], 2, quality.boundary_keep, quality.jitter, rng)
        self._draw_polyline_group(panel, lanes, self.COLORS["lane"], 1, quality.lane_keep, quality.jitter * 0.55, rng)
        self._draw_polyline_group(panel, green_segments, self.COLORS["green"], 1, quality.accent_keep, quality.jitter * 0.12, rng)
        self._draw_polyline_group(panel, red_segments, self.COLORS["red"], 1, quality.accent_keep, quality.jitter * 0.12, rng)

        if quality.false_positive > 0:
            count = int(panel.shape[0] * panel.shape[1] * quality.false_positive)
            for _ in range(count):
                x = int(rng.integers(0, panel.shape[1]))
                y = int(rng.integers(0, panel.shape[0]))
                cv2.circle(panel, (x, y), 1, self.COLORS["boundary"], -1, lineType=cv2.LINE_AA)

        return self._tight_crop_and_pad(panel, size, mode="cover")

    def create_single_figure(self, sample_token: str, save_path: str, seed_base: int = 1000) -> str:
        images = self._load_camera_images(sample_token)
        montage = self._camera_montage(images)

        fig = plt.figure(figsize=(10.2, 3.4))
        fig.patch.set_facecolor("white")
        gs = GridSpec(
            1,
            5,
            figure=fig,
            width_ratios=[2.55, 0.88, 0.88, 0.88, 0.88],
            wspace=0.05,
            left=0.03,
            right=0.99,
            top=0.86,
            bottom=0.10,
        )

        ax_img = fig.add_subplot(gs[0, 0])
        ax_img.imshow(montage, aspect="auto")
        ax_img.axis("off")

        panel_axes = []
        for col_idx, quality in enumerate(self.PANEL_CONFIG):
            ax = fig.add_subplot(gs[0, col_idx + 1])
            panel = self._render_panel_rgb(sample_token, quality, seed=seed_base + col_idx)
            ax.imshow(panel, aspect="auto")
            ax.axis("off")
            panel_axes.append(ax)

        fig.canvas.draw()
        img_box = ax_img.get_position()
        fig.text((img_box.x0 + img_box.x1) * 0.5, 0.885, "Images", ha="center", va="bottom", fontsize=12, fontweight="bold")
        for quality, ax in zip(self.PANEL_CONFIG, panel_axes):
            box = ax.get_position()
            fig.text((box.x0 + box.x1) * 0.5, 0.885, quality.name, ha="center", va="bottom", fontsize=12, fontweight="bold")

        fig.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return save_path

    def create_reference_figure(self, save_path: str, sample_tokens: Optional[Sequence[str]] = None) -> str:
        tokens = list(sample_tokens or self.REFERENCE_SAMPLE_TOKENS)
        tokens = [token for token in tokens if token in self.nusc._token2ind["sample"]]
        if len(tokens) < 5:
            raise RuntimeError("Reference comparison figure requires 5 valid sample tokens.")

        fig = plt.figure(figsize=(10.4, 14.6))
        fig.patch.set_facecolor("white")
        gs = GridSpec(
            5,
            5,
            figure=fig,
            width_ratios=[2.55, 0.88, 0.88, 0.88, 0.88],
            hspace=0.07,
            wspace=0.04,
            left=0.03,
            right=0.99,
            top=0.94,
            bottom=0.05,
        )

        panel_axes: List[object] = []
        image_axes: List[object] = []
        for row_idx, sample_token in enumerate(tokens[:5]):
            images = self._load_camera_images(sample_token)
            montage = self._camera_montage(images)

            ax_img = fig.add_subplot(gs[row_idx, 0])
            ax_img.imshow(montage, aspect="auto")
            ax_img.axis("off")
            image_axes.append(ax_img)

            row_seed_base = 1000 + row_idx * 100
            for col_idx, quality in enumerate(self.PANEL_CONFIG):
                ax = fig.add_subplot(gs[row_idx, col_idx + 1])
                panel = self._render_panel_rgb(sample_token, quality, seed=row_seed_base + col_idx, size=(124, 232))
                ax.imshow(panel, aspect="auto")
                ax.axis("off")
                if row_idx == 0:
                    panel_axes.append(ax)

        fig.canvas.draw()
        first_img_box = image_axes[0].get_position()
        fig.text((first_img_box.x0 + first_img_box.x1) * 0.5, 0.955, "Images", ha="center", va="bottom", fontsize=13, fontweight="bold")
        for quality, ax in zip(self.PANEL_CONFIG, panel_axes):
            box = ax.get_position()
            fig.text((box.x0 + box.x1) * 0.5, 0.955, quality.name, ha="center", va="bottom", fontsize=13, fontweight="bold")

        fig.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return save_path

    def create_reference_batch(self, sample_tokens: Optional[Sequence[str]] = None) -> List[str]:
        return self.create_named_batch(
            batch_dir_name="different_scene_batch",
            file_prefix="different_scene",
            sample_tokens=sample_tokens or self.REFERENCE_SAMPLE_TOKENS,
        )

    def create_named_batch(
        self,
        batch_dir_name: str,
        file_prefix: str,
        sample_tokens: Sequence[str],
    ) -> List[str]:
        tokens = [token for token in sample_tokens if token in self.nusc._token2ind["sample"]]
        if len(tokens) < 1:
            raise RuntimeError(f"{batch_dir_name} requires at least 1 valid sample token.")

        batch_dir = os.path.join(self.output_dir, batch_dir_name)
        os.makedirs(batch_dir, exist_ok=True)
        for file_name in os.listdir(batch_dir):
            if file_name.lower().endswith(".png"):
                os.remove(os.path.join(batch_dir, file_name))

        saved_paths: List[str] = []
        preview_images: List[Image.Image] = []
        for idx, token in enumerate(tokens, start=1):
            save_path = os.path.join(batch_dir, f"{file_prefix}_{idx:02d}_{token[:8]}.png")
            self.create_single_figure(token, save_path, seed_base=1000 + idx * 37)
            saved_paths.append(save_path)
            preview_images.append(Image.open(save_path).convert("RGB"))

        if preview_images:
            tile_w, tile_h = preview_images[0].size
            contact = Image.new("RGB", (tile_w, tile_h * len(preview_images)), (255, 255, 255))
            for idx, image in enumerate(preview_images):
                contact.paste(image, (0, idx * tile_h))
            index_path = os.path.join(batch_dir, "index.png")
            contact.save(index_path)
            saved_paths.append(index_path)

        return saved_paths

    def create_same_scene_batch(self, sample_tokens: Optional[Sequence[str]] = None) -> List[str]:
        if sample_tokens is not None:
            tokens = list(sample_tokens)
        elif self.SAME_SCENE_TOKEN in self.nusc._token2ind["sample"]:
            tokens = [self.SAME_SCENE_TOKEN] * 6
        else:
            tokens = [token for token in self.SAMPLE_TOKENS if token in self.nusc._token2ind["sample"]][:1] * 6
        return self.create_named_batch(
            batch_dir_name="same_scene_batch",
            file_prefix="same_scene",
            sample_tokens=tokens,
        )

    def create_batch(self) -> List[str]:
        tokens = self._available_sample_tokens()
        batch_dir = os.path.join(self.output_dir, "reference_style_batch")
        os.makedirs(batch_dir, exist_ok=True)
        legacy_dir = os.path.join(self.output_dir, "bevsegformer_style_comparison")
        os.makedirs(legacy_dir, exist_ok=True)

        saved_paths: List[str] = []
        preview_images: List[Image.Image] = []
        for idx, token in enumerate(tokens, start=1):
            save_path = os.path.join(batch_dir, f"figure_{idx:02d}_{token[:8]}.png")
            self.create_single_figure(token, save_path)
            saved_paths.append(save_path)
            preview_images.append(Image.open(save_path).convert("RGB"))

            if idx == 1:
                legacy_root = os.path.join(self.output_dir, "bevsegformer_style_comparison.png")
                legacy_nested = os.path.join(legacy_dir, "bevsegformer_style_comparison.png")
                self.create_single_figure(token, legacy_root)
                self.create_single_figure(token, legacy_nested)
                saved_paths.extend([legacy_root, legacy_nested])

        if preview_images:
            tile_w, tile_h = preview_images[0].size
            cols = 1
            rows = len(preview_images)
            sheet = Image.new("RGB", (tile_w * cols, tile_h * rows), (255, 255, 255))
            for idx, image in enumerate(preview_images):
                sheet.paste(image, (0, idx * tile_h))
            contact_path = os.path.join(batch_dir, "index.png")
            sheet.save(contact_path)
            saved_paths.append(contact_path)

        return saved_paths


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_root = os.path.join(project_root, "data", "nuscenes")
    output_dir = os.path.join(project_root, "visualization_results")

    visualizer = ReferenceStyleBatchVisualizer(data_root=data_root, output_dir=output_dir)
    reference_path = os.path.join(output_dir, "scene_comparison_hdmap.png")
    visualizer.create_reference_figure(reference_path)
    print(f"Saved: {reference_path}")
    paths = visualizer.create_batch()
    for path in paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    main()
