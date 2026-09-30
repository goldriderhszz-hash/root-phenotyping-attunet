"""Replay the recorded segmentation bootstrap without training or significance claims."""
from pathlib import Path
import csv,json
import numpy as np

ROOT=Path(__file__).resolve().parent
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    rows=read(ROOT/'tables'/'all_oof_image_metrics.csv')
    names=sorted({r['name'] for r in rows});seeds=[42,3407,2026]
    lookup={(r['config'],int(r['seed']),r['name']):float(r['dice']) for r in rows}
    comparisons=[('attunet_pixel','unet_pixel','attention_architecture_effect'),('attunet_cl_fixed','attunet_pixel','fixed_skeleton_loss_effect'),('attunet_cl_schedule','attunet_cl_fixed','two_stage_schedule_effect')]
    old={r['comparison']:r for r in read(ROOT/'tables'/'paired_config_effects.csv')}
    rng=np.random.default_rng(20260920);report=[]
    for treatment,control,label in comparisons:
        difference=np.array([[lookup[treatment,s,n]-lookup[control,s,n] for n in names] for s in seeds])
        boot=np.empty(10000)
        for i in range(len(boot)):
            seed_indices=rng.integers(0,len(seeds),len(seeds))
            image_indices=rng.integers(0,len(names),len(names))
            boot[i]=np.nanmean(difference[np.ix_(seed_indices,image_indices)])
        low,high=np.quantile(boot,[.025,.975]);mean=np.nanmean(difference)
        assert np.allclose([mean,low,high],[float(old[label][k]) for k in ['mean_paired_difference','bootstrap_ci95_low','bootstrap_ci95_high']],atol=1e-12,rtol=0), label
        report.append({'comparison':label,'mean_difference':mean,'lower_descriptive_percentile':low,'upper_descriptive_percentile':high,'random_seed':20260920,'draws':10000,'resampling':'independent draws of 3 seed indices and 50 image indices; paired treatment/control retained on Cartesian index product; not fold resampling'})
    target=ROOT/'generated';target.mkdir(exist_ok=True)
    output={'status':'passed','matches_recorded_intervals':True,'definition':'Segmentation intervals use seed-and-image resampling with fixed folds. These descriptive intervals do not model unknown biological dependence or support calibrated p-values.','comparisons':report}
    (target/'uncertainty_audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    with (target/'paired_segmentation_descriptive.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(report[0]));w.writeheader();w.writerows(report)
    print(json.dumps(output,indent=2))
if __name__=='__main__':main()
