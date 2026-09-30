"""Compute the frozen 36-setting sensitivity grid without redundant skeletons.

Each kernel is skeletonized once; pruning is advanced in blocks of five
iterations; all three minimum-component thresholds share those intermediates.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize

from fast_extractor import (ExtractorSettings, NEIGHBOR_KERNEL, component_points,
                            dominant_path_fast, prune_skeleton, raw_path)


KERNELS = (3, 5, 7)
PRUNINGS = (5, 10, 15, 20)
MINIMUMS = (5, 10, 20)


def analyze_pruned(pruned: np.ndarray, path_length: float, path_status: str,
                   kernel_size: int, pruning: int) -> list[dict]:
    base = {"scale_status": "unknown", "length_unit": "pixel",
            "closing_kernel": kernel_size, "pruning_iterations": pruning,
            "dominant_path_length": path_length, "path_status": path_status}
    if not pruned.any():
        return [{**base, "min_component_pixels": minimum,
                 "retained_segment_count": 0, "junction_region_count": 0,
                 "mean_local_acute_angle_deg": float("nan"), "valid_angle_n": 0,
                 "angle_failure_reason": "skeleton_empty_after_pruning"}
                for minimum in MINIMUMS]

    neighbor_count = cv2.filter2D(pruned, -1, NEIGHBOR_KERNEL, borderType=cv2.BORDER_CONSTANT) * pruned
    junction_pixels = (neighbor_count >= 3).astype(np.uint8)
    junction_count = max(0, cv2.connectedComponents(junction_pixels, connectivity=8)[0] - 1)

    primary_path, _, primary_component, primary_status = dominant_path_fast(pruned)
    primary_mask = np.zeros_like(pruned)
    if primary_path:
        for y, x in primary_path:
            primary_mask[y, x] = 1
    elif primary_status != "no_skeleton_component":
        primary_mask = primary_component

    lateral = ((pruned == 1) & (primary_mask == 0)).astype(np.uint8)
    lateral[junction_pixels == 1] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(lateral, connectivity=8)
    components = [(int(stats[index, cv2.CC_STAT_AREA]), component_points(labels, stats, index))
                  for index in range(1, count)
                  if int(stats[index, cv2.CC_STAT_AREA]) >= min(MINIMUMS)]

    measured = []
    if primary_path and components:
        path_array = np.asarray(primary_path)
        for size, component in components:
            distances = cdist(component, path_array, metric="euclidean")
            component_index, path_index = np.unravel_index(np.argmin(distances), distances.shape)
            base_point = component[component_index]
            radial = np.linalg.norm(component - base_point, axis=1)
            candidates = np.where((radial >= 8) & (radial <= 15))[0]
            lateral_vector = component[candidates[0]] - base_point if len(candidates) else component[-1] - base_point
            downstream = min(len(primary_path) - 1, path_index + 12)
            primary_vector = path_array[downstream] - path_array[path_index]
            norm_primary = float(np.linalg.norm(primary_vector))
            norm_lateral = float(np.linalg.norm(lateral_vector))
            if norm_primary == 0 or norm_lateral == 0:
                measured.append((size, float("nan"), "zero_length_angle_vector"))
                continue
            cosine = float(np.dot(primary_vector, lateral_vector) / (norm_primary * norm_lateral))
            angle = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
            measured.append((size, min(angle, 180.0 - angle), ""))

    rows = []
    for minimum in MINIMUMS:
        qualifying = [(size, angle, reason) for size, angle, reason in measured if size >= minimum]
        n_components = sum(size >= minimum for size, _ in components)
        reasons: set[str] = set()
        if not primary_path:
            reasons.add("no_valid_primary_path_after_pruning")
        elif n_components == 0:
            reasons.add("no_retained_lateral_components")
        else:
            reasons.update(reason for _, _, reason in qualifying if reason)
        angles = [angle for _, angle, _ in qualifying if math.isfinite(angle)]
        if not angles and not reasons:
            reasons.add("no_valid_angle_vectors")
        rows.append({**base, "min_component_pixels": minimum,
                     "retained_segment_count": n_components,
                     "junction_region_count": junction_count,
                     "mean_local_acute_angle_deg": float(np.mean(angles)) if angles else float("nan"),
                     "valid_angle_n": len(angles),
                     "angle_failure_reason": ";".join(sorted(reasons))})
    return rows


def extract_grid(mask: np.ndarray) -> list[dict]:
    binary, path_length, path_status = raw_path(mask)
    if not binary.any():
        return [{"scale_status": "unknown", "length_unit": "pixel",
                 "closing_kernel": kernel, "pruning_iterations": pruning,
                 "min_component_pixels": minimum, "dominant_path_length": float("nan"),
                 "retained_segment_count": 0, "junction_region_count": 0,
                 "mean_local_acute_angle_deg": float("nan"), "valid_angle_n": 0,
                 "angle_failure_reason": "empty_mask", "path_status": "empty_mask"}
                for kernel in KERNELS for pruning in PRUNINGS for minimum in MINIMUMS]
    rows = []
    for kernel_size in KERNELS:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        current = skeletonize(closed).astype(np.uint8)
        previous_pruning = 0
        for pruning in PRUNINGS:
            current = prune_skeleton(current, pruning - previous_pruning)
            rows.extend(analyze_pruned(current, path_length, path_status, kernel_size, pruning))
            previous_pruning = pruning
    assert len(rows) == 36
    return rows
