"""Portable inference for the frozen paper pipeline; no reference labels at inference."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import base64
import hashlib
import io
import json
import cv2
import numpy as np
from PIL import Image
from ..engine import InferenceEngine, read_gray, write_image, preview, evaluate, sha256
from ..models import MODEL_DIR
from . import ALGORITHM_ID
from .axis import select_path
from .candidates import generate
from .decoder import select
from .geometry import describe, vector_curve
from .terminal import local_extend


@lru_cache(maxsize=1)
def catalog():
    return json.loads((MODEL_DIR / 'paper/catalog.json').read_text(encoding='utf-8'))


def model_spec(fold):
    return catalog()['segmentation_models'][int(fold)]


@lru_cache(maxsize=20)
def artifact(fold, kind):
    record = next(r for r in catalog()['reconstruction_models'] if r['fold'] == fold and r['kind'] == kind)
    path = MODEL_DIR.parent / record['path']
    if sha256(path) != record['sha256']:
        raise ValueError(f'Paper model checksum mismatch: {kind}, fold {fold}')
    if kind == 'axis_policy':
        return json.loads(path.read_text(encoding='utf-8'))
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def forest_predict(fold, kind, inputs):
    """Exact ExtraTrees regression from numeric arrays (float32 input as sklearn)."""
    forest = artifact(fold, kind)
    x = np.asarray(inputs, dtype=np.float32)
    expected = next(r['features'] for r in catalog()['reconstruction_models'] if r['fold'] == fold and r['kind'] == kind)
    if x.ndim != 2 or x.shape[1] != expected or not np.isfinite(x).all():
        raise ValueError('Invalid forest feature matrix')
    result = np.zeros((len(x), forest['value'].shape[1]), dtype=float)
    for root in forest['roots']:
        node = np.full(len(x), root, dtype=np.int64)
        for _ in range(128):
            active = np.flatnonzero(forest['left'][node] >= 0)
            if not len(active):
                break
            current = node[active]
            left = x[active, forest['feature'][current]] <= forest['threshold'][current]
            node[active] = np.where(left, forest['left'][current], forest['right'][current])
        else:
            raise ValueError('Forest depth exceeds supported bound')
        result += forest['value'][node]
    return result / len(forest['roots'])


def hysteresis(probability, hard, threshold):
    weak = np.uint8(probability > max(.08, threshold * .35))
    n, labels, _, _ = cv2.connectedComponentsWithStats(weak, 8)
    count = np.bincount(labels[hard.astype(bool)], minlength=n)
    keep = count >= max(8, round(int(hard.sum()) * .0002))
    keep[0] = False
    return keep[labels].astype(np.uint8)


def reconstruct(photograph, probability, threshold, fold):
    """D3 mask -> protected axis -> full candidate pool -> primary E decoder."""
    hard = np.uint8(probability > threshold)
    if not hard.any():
        return dict(algorithm=ALGORITHM_ID, primary_arm='whole_RGB_joint', fold=fold,
                    geometry=describe(hard, []), paths=[], selected_count=0,
                    candidate_count=0, qc_flags=['empty_prediction', 'research_prototype_requires_review'])
    policy = artifact(fold, 'axis_policy')
    baseline = select_path(hard, hard, image=photograph, policy=policy)
    weak = hysteresis(probability, hard, threshold)
    alternative = select_path(weak, weak, image=photograph, policy=policy)
    anchor, extension, analysis_mask = local_extend(baseline['path'], alternative.get('alternatives', []), hard, weak)
    if len(anchor) < 2:
        raise ValueError('Protected axis unavailable; paper decoder cannot proceed')
    pool, feature = generate(hard, photograph, anchor)
    x = np.asarray(feature['x'])
    estimated = max(1., float(forest_predict(fold, 'count', [feature['count_features']])[0, 0]))
    errors = forest_predict(fold, 'RGB', x)
    chosen, solver = select(pool, errors, estimated, True)
    paths = [dict(candidate_id=int(i), path_yx=np.asarray(pool['candidates'][i]['path']).tolist(),
                  curve_yx=vector_curve(pool['candidates'][i]['path'])[0].tolist(),
                  curve_length_px=pool['candidates'][i]['curve_length'],
                  edge_ids=pool['candidates'][i]['edge_ids'],
                  predicted_support=float(errors[i, 0]), predicted_fragment=float(errors[i, 1]),
                  predicted_mixing=float(errors[i, 2]), protected=(i == 0)) for i in chosen]
    geometry = describe(analysis_mask, anchor)
    flags = sorted(set(baseline.get('flags', []) + geometry['flags'] + ['research_prototype_requires_review']))
    if solver['solver_status'] != 0:
        flags.append('decoder_feasible_incumbent_not_proven_optimal')
    return dict(algorithm=ALGORITHM_ID, primary_arm='whole_RGB_joint', fold=fold,
                candidate_count=len(pool['candidates']), selected_count=len(chosen),
                paths=paths, geometry=geometry, extension=extension, solver=solver,
                qc_flags=flags, reference_labels_read_during_inference=False,
                biological_identity_validated=False)


class PaperEngine(InferenceEngine):
    def __init__(self, fold=0):
        self.fold = int(fold)
        self.spec = model_spec(fold)
        super().__init__(MODEL_DIR / self.spec['filename'], self.spec['sha256'])

    def predict_photo(self, gray, photograph, batch_size=None):
        from .preprocessing import prepare
        normalized, flipped = prepare(gray, photograph)
        if batch_size is None:
            batch_size = 8 if self.device == 'CUDAExecutionProvider' else 2
        h, w = gray.shape
        padded = np.pad(normalized, ((0, -h % 256), (0, -w % 256), (0, 0)), mode='reflect')
        coords = [(y, x) for y in range(0, padded.shape[0]-255, 128) for x in range(0, padded.shape[1]-255, 128)]
        total = np.zeros(padded.shape[:2], np.float32)
        weights = np.zeros_like(total)
        for start in range(0, len(coords), batch_size):
            positions = coords[start:start+batch_size]
            tiles = np.stack([padded[y:y+256, x:x+256].transpose(2, 0, 1) for y, x in positions])
            outputs = self.session.run(['probability'], {'image': tiles})[0][:, 0]
            for result, (y, x) in zip(outputs, positions):
                total[y:y+256, x:x+256] += result * self.weight
                weights[y:y+256, x:x+256] += self.weight
        p = (total / (weights + 1e-8))[:h, :w]
        return p[:, ::-1].copy() if flipped else p


def export_figure(output, photograph, reconstruction):
    """Original image plus editable vector curves; no invented physical scale."""
    h, w = photograph.shape[:2]
    overlay = photograph.copy()
    skeleton = np.full_like(photograph, 255)
    buffer = io.BytesIO()
    source_image = Image.fromarray(cv2.cvtColor(photograph, cv2.COLOR_BGR2RGB))
    source_image.save(output / 'source_decoded.png', format='PNG')
    # Photographs may use JPEG for presentation; quantitative inference always uses the input file.
    source_image.save(buffer, format='JPEG', quality=95, subsampling=0)
    scale = min(180 / w, 170 / h)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w*scale:.3f}mm" height="{h*scale:.3f}mm" viewBox="0 0 {w} {h}">',
           '<title>RootScope contour-guided reconstruction</title>',
           '<desc>Orange: protected operational axis; blue: other retained paths. Coordinates in pixels. Biological identities unvalidated. Photograph uses JPEG quality 95 for presentation; lossless decoded source saved separately.</desc>',
           f'<image id="photograph" width="{w}" height="{h}" href="data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"/>',
           '<g id="editable-reconstruction" fill="none" stroke-linecap="round" stroke-linejoin="round">']
    for item in reconstruction['paths']:
        points = np.asarray(item['curve_yx'])
        color = '#d55e00' if item['protected'] else '#0072b2'
        coordinates = ' '.join(f'{x:.2f},{y:.2f}' for y, x in points)
        svg.append(f'<polyline id="candidate-{item["candidate_id"]}" stroke="{color}" stroke-width="2" points="{coordinates}"/>')
        pixels = np.rint(points[:, ::-1]).astype(np.int32)
        bgr = (0, 94, 213) if item['protected'] else (178, 114, 0)
        cv2.polylines(overlay, [pixels], False, bgr, 2)
        cv2.polylines(skeleton, [pixels], False, bgr, 2)
    svg.append('</g></svg>')
    (output / 'reconstruction.svg').write_text('\n'.join(svg), encoding='utf-8')
    write_image(output / 'overlay.png', overlay)
    write_image(output / 'skeleton.png', skeleton)
    # DPI metadata does not create additional information; the SVG retains the original pixels.
    Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)).save(output / 'figure_preview.png', dpi=(300, 300))


def analyze(image_path, output, engine, threshold, reference_path=None, save_probability=False):
    gray = read_gray(image_path)
    photo = cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_COLOR)
    reference = read_gray(reference_path) > 127 if reference_path else None
    if reference is not None and reference.shape != gray.shape:
        raise ValueError('Reference and photograph dimensions differ')
    probability = engine.predict_photo(gray, photo)
    reconstruction = reconstruct(photo, probability, threshold, engine.fold)
    geometry = reconstruction['geometry']
    descriptors = {key: geometry.get(key) for key in ('axis_curve_length_px', 'first_order_lateral_count',
                   'branch_junction_count', 'mean_parent_child_angle_deg', 'ambiguous_contact_count', 'valid_angle_n', 'missing_angle_n')}
    descriptors['reconstructed_path_count'] = reconstruction['selected_count']
    descriptors['path_status'] = 'candidate' if reconstruction['paths'] else 'unavailable'
    output.mkdir(parents=True, exist_ok=True)
    export_figure(output, photo, reconstruction)
    write_image(output / 'prediction_mask.png', np.uint8(probability > threshold) * 255)
    write_image(output / 'source_preview.jpg', preview(photo))
    if save_probability:
        np.savez_compressed(output / 'probability_float32.npz', probability=probability)
    result = dict(image_name=image_path.name, image_sha256=sha256(image_path), width=gray.shape[1], height=gray.shape[0],
                  foreground_fraction=float((probability > threshold).mean()), threshold=float(threshold),
                  pipeline='paper', model_id=engine.spec['id'], fold=engine.fold,
                  onnx_sha256=engine.spec['sha256'], descriptors=descriptors,
                  reconstruction=reconstruction, qc_flags=reconstruction['qc_flags'])
    if reference is not None:
        result['reference_sha256'] = sha256(reference_path)
        result['evaluation'] = evaluate(probability > threshold, reference)
    return result
