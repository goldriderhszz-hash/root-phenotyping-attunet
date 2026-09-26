from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class ExtractorSettings:
    closing_kernel: int = 5
    pruning_iterations: int = 15
    min_component_pixels: int = 10


NEIGHBORS = [
    (-1, -1, math.sqrt(2.0)),
    (-1, 0, 1.0),
    (-1, 1, math.sqrt(2.0)),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, -1, math.sqrt(2.0)),
    (1, 0, 1.0),
    (1, 1, math.sqrt(2.0)),
]


def prune_skeleton(skeleton: np.ndarray, iterations: int) -> np.ndarray:
    result = skeleton.astype(np.uint8).copy()
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    for _ in range(iterations):
        neighbors = cv2.filter2D(result, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        result[(result == 1) & (neighbors == 1)] = 0
    return result


def weighted_shortest_path(skeleton: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> tuple[list[tuple[int, int]] | None, float]:
    height, width = skeleton.shape
    queue: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    distances = {start: 0.0}
    previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        distance, current = heapq.heappop(queue)
        if distance != distances.get(current):
            continue
        if current == end:
            break
        y, x = current
        for dy, dx, edge in NEIGHBORS:
            neighbor = (y + dy, x + dx)
            ny, nx = neighbor
            if not (0 <= ny < height and 0 <= nx < width) or skeleton[ny, nx] != 1:
                continue
            candidate = distance + edge
            if candidate < distances.get(neighbor, math.inf):
                distances[neighbor] = candidate
                previous[neighbor] = current
                heapq.heappush(queue, (candidate, neighbor))
    if end not in previous:
        return None, float("nan")
    path = []
    current: tuple[int, int] | None = end
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    return path, float(distances[end])


def dominant_path(skeleton: np.ndarray, minimum_pixels: int = 10) -> tuple[list[tuple[int, int]] | None, float, np.ndarray, str]:
    count, labels = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)
    if count <= 1:
        return None, float("nan"), np.zeros_like(skeleton, dtype=np.uint8), "no_skeleton_component"
    candidates = []
    for label_index in range(1, count):
        points = np.argwhere(labels == label_index)
        span = int(points[:, 0].max() - points[:, 0].min())
        candidates.append((span, len(points), -label_index, label_index, points))
    _, n_pixels, _, selected, points = max(candidates, key=lambda item: item[:3])
    component = (labels == selected).astype(np.uint8)
    if n_pixels < minimum_pixels:
        return None, float("nan"), component, "dominant_component_below_minimum"
    top = tuple(points[np.lexsort((points[:, 1], points[:, 0]))][0])
    bottom = tuple(points[np.lexsort((points[:, 1], points[:, 0]))][-1])
    path, length = weighted_shortest_path(component, top, bottom)
    if path is None:
        return None, float("nan"), component, "weighted_path_not_found"
    return path, length, component, "ok"


def extract_descriptors(mask: np.ndarray, settings: ExtractorSettings = ExtractorSettings()) -> dict:
    if settings.closing_kernel not in {3, 5, 7} or settings.pruning_iterations < 0 or settings.min_component_pixels < 1:
        raise ValueError(settings)
    binary = (mask > 0).astype(np.uint8)
    base = {
        "scale_status": "unknown",
        "length_unit": "pixel",
        "closing_kernel": settings.closing_kernel,
        "pruning_iterations": settings.pruning_iterations,
        "min_component_pixels": settings.min_component_pixels,
    }
    if not binary.any():
        return {**base, "dominant_path_length": float("nan"), "retained_segment_count": 0, "junction_region_count": 0, "mean_local_acute_angle_deg": float("nan"), "valid_angle_n": 0, "angle_failure_reason": "empty_mask", "path_status": "empty_mask"}

    raw_skeleton = skeletonize(binary).astype(np.uint8)
    _, path_length, _, path_status = dominant_path(raw_skeleton)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (settings.closing_kernel, settings.closing_kernel))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    pruned = prune_skeleton(skeletonize(closed).astype(np.uint8), settings.pruning_iterations)
    if not pruned.any():
        return {**base, "dominant_path_length": path_length, "retained_segment_count": 0, "junction_region_count": 0, "mean_local_acute_angle_deg": float("nan"), "valid_angle_n": 0, "angle_failure_reason": "skeleton_empty_after_pruning", "path_status": path_status}

    neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(pruned, -1, neighbor_kernel, borderType=cv2.BORDER_CONSTANT) * pruned
    junction_pixels = (neighbor_count >= 3).astype(np.uint8)
    junction_count = max(0, cv2.connectedComponents(junction_pixels, connectivity=8)[0] - 1)

    primary_path, _, primary_component, primary_status = dominant_path(pruned)
    primary_mask = np.zeros_like(pruned)
    if primary_path:
        for y, x in primary_path:
            primary_mask[y, x] = 1
    elif primary_status != "no_skeleton_component":
        primary_mask = primary_component

    lateral = ((pruned == 1) & (primary_mask == 0)).astype(np.uint8)
    lateral[junction_pixels == 1] = 0
    component_count, component_labels = cv2.connectedComponents(lateral, connectivity=8)
    components = [np.argwhere(component_labels == index) for index in range(1, component_count)]
    components = [points for points in components if len(points) >= settings.min_component_pixels]

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
    return {
        **base,
        "dominant_path_length": path_length,
        "retained_segment_count": len(components),
        "junction_region_count": junction_count,
        "mean_local_acute_angle_deg": float(np.mean(angles)) if angles else float("nan"),
        "valid_angle_n": len(angles),
        "angle_failure_reason": ";".join(sorted(failure_reasons)),
        "path_status": path_status,
    }
