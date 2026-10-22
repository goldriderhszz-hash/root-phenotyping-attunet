"""Convert the catalog-verified frozen sklearn forests to portable numeric arrays.

Requires scikit-learn/joblib in the research environment, not the application.
Input layout: fold_0/count.joblib, fold_0/geometry.joblib, fold_0/RGB.joblib, etc.
No model is fitted. Reject source files that do not match the recorded study hashes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import joblib
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--catalog',type=Path,default=Path(__file__).resolve().parents[1]/'model/paper/catalog.json')
    args=parser.parse_args()
    records=json.loads(args.catalog.read_text(encoding='utf-8'))['reconstruction_models']
    for record in records:
        if record['kind']=='axis_policy':continue
        fold,kind=record['fold'],record['kind']
        source=args.input/f'fold_{fold}/{kind}.joblib'
        if hashlib.sha256(source.read_bytes()).hexdigest()!=record['source_sha256']:
            raise ValueError(f'Source model hash mismatch: fold {fold}, {kind}')
        forest=joblib.load(source)
        arrays={key:[] for key in ('left','right','feature','threshold','value')}
        roots=[];offset=0
        for tree in forest.estimators_:
            t=tree.tree_;roots.append(offset)
            arrays['left'].append(np.where(t.children_left>=0,t.children_left+offset,-1))
            arrays['right'].append(np.where(t.children_right>=0,t.children_right+offset,-1))
            arrays['feature'].append(t.feature);arrays['threshold'].append(t.threshold);arrays['value'].append(t.value[:,:,0])
            offset+=t.node_count
        destination=args.output/f'fold_{fold}/{kind}.npz'
        destination.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(destination,roots=np.array(roots),**{key:np.concatenate(values) for key,values in arrays.items()})
        print(json.dumps(dict(fold=fold,kind=kind,trees=len(roots),source_sha256=record['source_sha256'])))

if __name__=='__main__':main()
