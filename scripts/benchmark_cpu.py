"""Time frozen ONNX CPU inference and descriptor extraction on own-fold images.

Excludes loading, warm-up, evaluation and export. Three actual repeated passes.
"""
from pathlib import Path
import argparse
import csv
import json
import platform
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rootscope.engine import InferenceEngine,read_gray
from rootscope.models import model_path,model_spec
from rootscope.phenotype_extractor import extract_descriptors

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('data',type=Path);p.add_argument('--output',type=Path,default=ROOT/'validation'/'cpu_timing');p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--resume',action='store_true',help='Continue an interrupted run with unchanged inputs, models and software')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    folds=json.loads((ROOT/'research'/'fold_assignments.json').read_text(encoding='utf-8'))['folds']
    rows=[]
    path=a.output/'cpu_image_runs.csv'
    if a.resume and path.exists():
        rows=list(csv.DictReader(path.open(encoding='utf-8')))
        for row in rows:
            for key in ['fold','repeat']:row[key]=int(row[key])
            for key in ['read_seconds','clahe_inference_fusion_seconds','threshold_seconds','descriptor_seconds','total_seconds']:row[key]=float(row[key])
    complete={(r['image'],r['fold'],r['repeat']) for r in rows}
    for fold in range(5):
        spec=model_spec(f'fold_{fold}');engine=InferenceEngine(model_path(spec['id']),spec['sha256'])
        assert engine.device=='CPUExecutionProvider'
        engine.predict(np.zeros((256,256),dtype=np.uint8))
        for repeat in range(a.repeats):
            for name in folds[str(fold)]:
                if (name,fold,repeat+1) in complete:continue
                while (ROOT / '.pause-cpu-benchmark').exists():
                    time.sleep(1)
                start=time.perf_counter();image=read_gray(a.data/f'{name}.tif');read=time.perf_counter()
                probability=engine.predict(image);infer=time.perf_counter()
                mask=probability>float(spec['threshold']);threshold=time.perf_counter()
                desc=extract_descriptors(mask);end=time.perf_counter()
                row={'image':name,'fold':fold,'repeat':repeat+1,'provider':engine.device,'read_seconds':read-start,'clahe_inference_fusion_seconds':infer-read,'threshold_seconds':threshold-infer,'descriptor_seconds':end-threshold,'total_seconds':end-start,'dominant_path_length':desc['dominant_path_length'],'retained_segment_count':desc['retained_segment_count'],'junction_region_count':desc['junction_region_count'],'mean_local_acute_angle_deg':desc['mean_local_acute_angle_deg']}
                rows.append(row)
                with path.open('w',encoding='utf-8',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(rows)
                print(f"{len(rows)}/{50*a.repeats}: fold {fold} repeat {repeat+1} {name} {row['total_seconds']:.2f} s",flush=True)
    summary={'status':'complete','n_distinct_images':50,'repeats':a.repeats,'n_image_runs':len(rows),'platform':platform.platform(),'python':sys.version,'processor':platform.processor(),'provider':'CPUExecutionProvider','timing_scope':'source ONNX CPU numerical pipeline; model load, tile warm-up, reference evaluation and disk output excluded','stages':{}}
    for key in ['read_seconds','clahe_inference_fusion_seconds','threshold_seconds','descriptor_seconds','total_seconds']:
        values=np.array([r[key] for r in rows]);summary['stages'][key]={'median':float(np.median(values)),'q25':float(np.quantile(values,.25)),'q75':float(np.quantile(values,.75)),'p90':float(np.quantile(values,.9))}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
