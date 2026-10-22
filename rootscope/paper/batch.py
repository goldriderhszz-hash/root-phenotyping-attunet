"""Paper batch analysis with per-image folds and reproducible reporting."""
from pathlib import Path
from datetime import datetime
import json
import platform
import sys
import uuid
import zipfile
import cv2
import numpy as np
import scipy
import networkx
import onnxruntime
from .. import APPLICATION_NAME
from ..batch import IMAGE_SUFFIXES, finite, match_reference, row_for, write_csv, utc_now
from ..engine import sha256
from ..models import MODEL_DIR
from .runtime import PaperEngine, analyze, catalog


def resolve_fold(image, model_id):
    if model_id != 'auto':
        fold = int(model_id.rsplit('_', 1)[-1])
        if fold not in range(5):
            raise ValueError('Fold must be between 0 and 4')
        return fold, 'operator_selected_fold'
    study = json.loads((MODEL_DIR / 'paper/study_folds.json').read_text(encoding='utf-8'))
    entry = study.get(sha256(image))
    return (int(entry['fold']), 'verified_study_image_hash') if entry else (0, 'new_image_uses_fold_0_not_external_validation')


def run_paper_batch(images, references, output_root, threshold=None, save_probability=False,
                    cancel_event=None, progress=None, make_zip=True, model_id='auto'):
    if not images:
        raise ValueError('No input images')
    if threshold is not None and not .1 <= threshold <= .9:
        raise ValueError('Threshold must be between 0.10 and 0.90')
    for image in images + references:
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f'Invalid input image: {image}')
    folder = Path(output_root).expanduser().resolve() / ('RootScope_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
    (folder / 'images').mkdir(parents=True)
    started = utc_now()
    results, errors, engines = [], [], {}
    stopped = False
    for index, image in enumerate(images, 1):
        if cancel_event is not None and cancel_event.is_set():
            stopped = True
            break
        if progress:
            progress(index - 1, len(images), image.name)
        try:
            fold, assignment = resolve_fold(image, model_id)
            # Keep at most one ONNX session: five simultaneous sessions can exceed desktop memory.
            if fold not in engines:
                engines.clear()
                engines[fold] = PaperEngine(fold)
            engine = engines[fold]
            used = float(engine.spec['threshold']) if threshold is None else float(threshold)
            image_id = uuid.uuid4().hex[:10]
            item = analyze(image, folder / 'images' / image_id, engine, used, match_reference(image, references), save_probability)
            item['fold_assignment'] = assignment
            if assignment != 'verified_study_image_hash':
                item['qc_flags'].append(assignment)
            if threshold is not None and used != float(engine.spec['threshold']):
                item['qc_flags'].append('threshold_override_changes_paper_protocol')
            item.update(image_id=image_id, _source_path=str(image))
            results.append(finite(item) | {'_source_path': str(image)})
        except Exception as error:
            errors.append(dict(image_name=image.name, reason=f'{type(error).__name__}: {error}'))
        if progress:
            progress(index, len(images), image.name)
    write_csv(folder / 'results.csv', [row_for(r) for r in results], ['image_name', 'axis_curve_length_px', 'reconstructed_path_count', 'qc_flags'])
    write_csv(folder / 'errors.csv', errors, ['image_name', 'reason'])
    (folder / 'results.json').write_text(json.dumps(finite(results), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    report = dict(application=APPLICATION_NAME, pipeline='paper', algorithm=catalog()['algorithm'],
                  primary_arm=catalog()['primary_arm'], model_catalog_sha256=sha256(MODEL_DIR / 'paper/catalog.json'),
                  created_utc=started, completed_utc=utc_now(), status='cancelled' if stopped else 'complete',
                  requested_images=len(images), successful_images=len(results), failed_images=len(errors), errors=errors,
                  measurement_scale='pixels; physical calibration unavailable', biological_identity_validated=False,
                  all_data_production_model=False, validation_scope='conditional internal development evidence on reused images',
                  preprocessing='canonical reflection; CLAHE gray + Lab-b + absolute local contrast; 256 tiles; stride 128; Gaussian sigma 64',
                  models=catalog(), images=[{k:r.get(k) for k in ('image_name', 'image_sha256', 'model_id', 'fold', 'fold_assignment', 'threshold', 'onnx_sha256', 'reference_sha256')} for r in results],
                  software=dict(python=sys.version.split()[0],platform=platform.platform(),numpy=np.__version__,opencv=cv2.__version__,scipy=scipy.__version__,networkx=networkx.__version__,onnxruntime=onnxruntime.__version__))
    build_info = Path(__file__).resolve().parents[1] / 'build_info.json'
    if build_info.is_file():
        report['source_identity'] = json.loads(build_info.read_text(encoding='utf-8'))
    (folder / 'provenance.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# RootScope analysis report', '', f"Completed images: {len(results)} / {len(images)}; failures: {len(errors)}.", '',
             'Pipeline: D3 segmentation, protected operational axis, whole-root RGB/geometry quality forest, count forest and edge-exclusive MILP (paper primary arm E).', '',
             'Lengths are in pixels. Root identities, anatomical lateral counts and emergence angles require independent review. Automatic flags are not a measured human-correction rate.', '',
             '| Image | Axis length (px) | Retained paths | Y attachments | Degree-3 junctions | Parent-child angle (deg) |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for item in results:
        d = item['descriptors']
        values=[d.get(k) for k in ('axis_curve_length_px','reconstructed_path_count','first_order_lateral_count','branch_junction_count','mean_parent_child_angle_deg')]
        safe_name=item['image_name'].replace('|','\\|').replace('\n',' ')
        lines.append('| '+safe_name+' | '+' | '.join('NA' if v is None else f'{v:.3f}' if isinstance(v,float) else str(v) for v in values)+' |')
    lines += ['', 'Machine-readable observations: results.csv and results.json. Reproducibility: provenance.json. Each image includes reconstruction.svg with editable curve overlays and the embedded original photograph. No physical scale is inferred.', '',
              'Figure caption draft: Orange identifies the protected operational axis; blue identifies other retained candidates. Curves are model-derived and may include incomplete or mixed paths. Source coordinates are pixels. Report the actual sample selection and validation unit when reusing this figure.']
    (folder / 'Report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive = None
    if make_zip:
        archive = folder / (folder.name + '_results.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as output:
            for path in folder.rglob('*'):
                if path.is_file() and path != archive:
                    output.write(path,path.relative_to(folder))
    return dict(folder=str(folder),archive=str(archive) if archive else None,results=results,errors=errors,report=report)
