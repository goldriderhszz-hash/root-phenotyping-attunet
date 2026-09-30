"""Equivalent, crop-aware implementation of the frozen phenotype extractor.

This module deliberately keeps the descriptor definitions in the original
19_五折受控交叉验证_2026-09-20/code/phenotype_extractor.py. The only algorithmic
change is enumerating connected-component pixels within their bounding boxes,
instead of scanning the full 7.4-megapixel label image for every component.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
import os

import cv2
import numpy as np
from skimage.morphology import skeletonize

ORIGINAL_CODE = Path(__file__).resolve().parents[1] / "research"
sys.path.insert(0, str(ORIGINAL_CODE))
from phenotype_extractor import ExtractorSettings, prune_skeleton, weighted_shortest_path  # noqa: E402


NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


def component_points(labels: np.ndarray, stats: np.ndarray, index: int) -> np.ndarray:
    x, y, width, height, _ = (int(v) for v in stats[index])
    points = np.argwhere(labels[y:y + height, x:x + width] == index)
    points[:, 0] += y
    points[:, 1] += x
    return points


def dominant_path_fast(skeleton: np.ndarray, minimum_pixels: int = 10):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton.astype(np.uint8), connectivity=8)
    if count <= 1:
        return None, float("nan"), np.zeros_like(skeleton, dtype=np.uint8), "no_skeleton_component"
    candidates = []
    for index in range(1, count):
        span = int(stats[index, cv2.CC_STAT_HEIGHT] - 1)
        candidates.append((span, int(stats[index, cv2.CC_STAT_AREA]), -index, index))
    _, n_pixels, _, selected = max(candidates, key=lambda item: item[:3])
    component = (labels == selected).astype(np.uint8)
    if n_pixels < minimum_pixels:
        return None, float("nan"), component, "dominant_component_below_minimum"
    points = component_points(labels, stats, selected)
    ordered = points[np.lexsort((points[:, 1], points[:, 0]))]
    top, bottom = tuple(ordered[0]), tuple(ordered[-1])
    path, length = weighted_shortest_path(component, top, bottom)
    if path is None:
        return None, float("nan"), component, "weighted_path_not_found"
    return path, length, component, "ok"


def raw_path(mask: np.ndarray):
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        return binary, float("nan"), "empty_mask"
    raw_skeleton = skeletonize(binary).astype(np.uint8)
    _, length, _, status = dominant_path_fast(raw_skeleton)
    return binary, length, status


def extract_with_raw(binary: np.ndarray, path_length: float, path_status: str,
                     settings: ExtractorSettings = ExtractorSettings()) -> dict:
    if settings.closing_kernel not in {3, 5, 7} or settings.pruning_iterations < 0 or settings.min_component_pixels < 1:
        raise ValueError(settings)
    base = {
        "scale_status": "unknown", "length_unit": "pixel",
        "closing_kernel": settings.closing_kernel,
        "pruning_iterations": settings.pruning_iterations,
        "min_component_pixels": settings.min_component_pixels,
    }
    if not binary.any():
        return {**base, "dominant_path_length": float("nan"), "retained_segment_count": 0,
                "junction_region_count": 0, "mean_local_acute_angle_deg": float("nan"),
                "valid_angle_n": 0, "angle_failure_reason": "empty_mask", "path_status": "empty_mask"}

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (settings.closing_kernel, settings.closing_kernel))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    pruned = prune_skeleton(skeletonize(closed).astype(np.uint8), settings.pruning_iterations)
    if not pruned.any():
        return {**base, "dominant_path_length": path_length, "retained_segment_count": 0,
                "junction_region_count": 0, "mean_local_acute_angle_deg": float("nan"),
                "valid_angle_n": 0, "angle_failure_reason": "skeleton_empty_after_pruning",
                "path_status": path_status}

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
    components = [component_points(labels, stats, index) for index in range(1, count)
                  if int(stats[index, cv2.CC_STAT_AREA]) >= settings.min_component_pixels]

    angles: list[float] = []
    failure_reasons: set[str] = set()
    if not primary_path:
        failure_reasons.add("no_valid_primary_path_after_pruning")
    elif not components:
        failure_reasons.add("no_retained_lateral_components")
    else:
        path_array = np.asarray(primary_path)
        for component in components:
            distances = np.linalg.norm(component[:, None, :] - path_array[None, :, :], axis=2)
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
                failure_reasons.add("zero_length_angle_vector")
                continue
            cosine = float(np.dot(primary_vector, lateral_vector) / (norm_primary * norm_lateral))
            angle = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
            angles.append(min(angle, 180.0 - angle))
    if not angles and not failure_reasons:
        failure_reasons.add("no_valid_angle_vectors")
    return {**base, "dominant_path_length": path_length,
            "retained_segment_count": len(components), "junction_region_count": junction_count,
            "mean_local_acute_angle_deg": float(np.mean(angles)) if angles else float("nan"),
            "valid_angle_n": len(angles), "angle_failure_reason": ";".join(sorted(failure_reasons)),
            "path_status": path_status}


def extract_descriptors_fast(mask: np.ndarray,
                             settings: ExtractorSettings = ExtractorSettings()) -> dict:
    binary, length, status = raw_path(mask)
    return extract_with_raw(binary, length, status, settings)
