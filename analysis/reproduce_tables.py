"""Audit frozen results and regenerate manuscript summaries without training.

Run from any directory: python analysis/reproduce_tables.py --output <directory>
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
CONFIGS=['unet_pixel','attunet_pixel','attunet_cl_fixed','attunet_cl_schedule']
SEEDS={42,3407,2026}
TRAITS=['dominant_path_length','retained_segment_count','junction_region_count','mean_local_acute_angle_deg']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'generated')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'tables'
    metrics=pd.read_csv(source/'all_oof_image_metrics.csv')
    assert len(metrics)==600 and metrics.name.nunique()==50
    assert set(metrics.config)==set(CONFIGS) and set(metrics.seed)==SEEDS
    assert not metrics.duplicated(['name','config','seed']).any()
    assert (metrics.groupby('name').fold.nunique()==1).all()
    assert (metrics.groupby(['config','seed','fold']).size()==10).all()
    pairs=pd.read_csv(source/'descriptor_comparison_long.csv')
    assert len(pairs)==2400 and not pairs.duplicated(['sample_name','config','seed','trait']).any()
    assert set(pairs.trait)==set(TRAITS) and (pairs.groupby(['config','seed','trait']).size()==50).all()
    assert (pairs.groupby(['sample_name','trait']).reference_value.nunique()==1).all()
    assert set(pairs.status)=={'valid'}
    summary=[]
    for config in CONFIGS:
        for endpoint in ['dice','iou','hard_cldice','width_le3_recall']:
            values=metrics[metrics.config==config].groupby('seed')[endpoint].mean()
            summary.append({'config':config,'endpoint':endpoint,'mean':values.mean(),'sd_across_seeds':values.std(ddof=1),'n_distinct_images':50,'n_training_seeds':3})
    pd.DataFrame(summary).to_csv(out/'table1_segmentation.csv',index=False)
    original=pd.read_csv(source/'run_summary.csv')
    endpoint_map={'dice':'OOF macro dice','iou':'OOF macro iou','hard_cldice':'OOF macro hard_cldice','width_le3_recall':'OOF macro width_le3_recall'}
    for row in summary:
        match=original[(original.config==row['config'])&(original.endpoint==endpoint_map[row['endpoint']])]
        if len(match)!=1:
            # Width endpoint spelling is checked explicitly rather than silently skipped.
            match=original[(original.config==row['config'])&original.endpoint.str.contains(row['endpoint'],regex=False)]
        assert len(match)==1, row
        assert abs(row['mean']-match.iloc[0]['mean'])<1e-12
        assert abs(row['sd_across_seeds']-match.iloc[0]['sd_across_seeds'])<1e-12
    descriptor=[]
    for (config,seed,trait),part in pairs.groupby(['config','seed','trait']):
        x=part.reference_value.to_numpy();y=part.estimate_value.to_numpy();d=y-x
        ccc=2*np.mean((x-x.mean())*(y-y.mean()))/(x.var()+y.var()+(x.mean()-y.mean())**2)
        nonzero=x!=0
        mape=100*np.mean(np.abs(d[nonzero]/x[nonzero])) if nonzero.any() else np.nan
        descriptor.append({'config':config,'seed':int(seed),'trait':trait,'mae':abs(d).mean(),'bias':d.mean(),'rmse':np.sqrt(np.mean(d*d)),'mape_percent_nonzero_reference':mape,'ccc':ccc,'n':len(d)})
    desc=pd.DataFrame(descriptor)
    old=pd.read_csv(source/'descriptor_summary_by_seed.csv')
    merged=desc.merge(old,on=['config','seed','trait'],suffixes=['_new','_old'],validate='one_to_one')
    assert np.allclose(merged.mae_new,merged.mae_old,rtol=0,atol=1e-10)
    assert np.allclose(merged.ccc_new,merged.ccc_old,rtol=0,atol=1e-12)
    assert np.allclose(merged.bias,merged.bias_estimate_minus_reference,rtol=0,atol=1e-10)
    assert np.allclose(merged.rmse_new,merged.rmse_old,rtol=0,atol=1e-10)
    assert np.allclose(merged.mape_percent_nonzero_reference_new,merged.mape_percent_nonzero_reference_old,rtol=0,atol=1e-10,equal_nan=True)
    assert (merged.n==merged.n_valid_pairs).all()
    desc.to_csv(out/'table2_3_descriptor_seed_statistics.csv',index=False)
    contrasts=[('attention','attunet_pixel','unet_pixel'),('fixed_loss','attunet_cl_fixed','attunet_pixel'),('two_stage','attunet_cl_schedule','attunet_cl_fixed')]
    effects=[];lofo=[]
    pivot=metrics.pivot(index=['name','seed','fold'],columns='config',values='dice')
    for label,treatment,control in contrasts:
        d=(pivot[treatment]-pivot[control])*100
        for (fold,seed),part in d.groupby(level=['fold','seed']):
            effects.append({'contrast':label,'endpoint':'dice_pp','fold':int(fold),'seed':int(seed),'mean_difference':part.mean()})
        for fold in range(5):
            keep=d[d.index.get_level_values('fold')!=fold]
            lofo.append({'contrast':label,'endpoint':'dice_pp','omitted_fold':fold,'n_distinct_images':40,'mean_difference':keep.mean()})
    for trait in TRAITS:
        p=pairs[pairs.trait==trait].pivot(index=['sample_name','seed','fold'],columns='config',values='absolute_error')
        for label,treatment,control in contrasts:
            d=p[treatment]-p[control]
            for (fold,seed),part in d.groupby(level=['fold','seed']):
                effects.append({'contrast':label,'endpoint':'absolute_error_'+trait,'fold':int(fold),'seed':int(seed),'mean_difference':part.mean()})
            for fold in range(5):
                keep=d[d.index.get_level_values('fold')!=fold]
                lofo.append({'contrast':label,'endpoint':'absolute_error_'+trait,'omitted_fold':fold,'n_distinct_images':40,'mean_difference':keep.mean()})
    pd.DataFrame(effects).to_csv(out/'paired_fold_seed_effects.csv',index=False)
    pd.DataFrame(lofo).to_csv(out/'leave_one_fold_out.csv',index=False)
    # Descriptive limits for the plotted image means, not for pooled 150 repeated pairs.
    fixed=pairs[pairs.config=='attunet_cl_fixed']
    grouped=fixed.groupby(['sample_name','fold','trait'],as_index=False).agg(reference=('reference_value','first'),estimate=('estimate_value','mean'),estimate_min=('estimate_value','min'),estimate_max=('estimate_value','max'))
    grouped['difference']=grouped.estimate-grouped.reference
    grouped['pair_mean']=(grouped.estimate+grouped.reference)/2
    grouped.to_csv(out/'fig7_image_mean_pairs.csv',index=False)
    loa=[]
    for trait,part in grouped.groupby('trait'):
        bias=part.difference.mean();sd=part.difference.std(ddof=1)
        loa.append({'trait':trait,'n_distinct_images':50,'bias_of_seed_mean':bias,'difference_sd_across_images':sd,'lower_descriptive_limit':bias-1.96*sd,'upper_descriptive_limit':bias+1.96*sd,'estimand':'prediction averaged over three training seeds per source image'})
    pd.DataFrame(loa).to_csv(out/'fig7_descriptive_limits.csv',index=False)
    records=json.loads((ROOT/'run_records.json').read_text(encoding='utf-8'))
    scheduled=[r for r in records if r['config']=='attunet_cl_schedule']
    report={'status':'passed','distinct_images':50,'oof_segmentation_rows':600,'descriptor_rows':2400,'fits':len(records),'schedule_selected_before_or_at_transition':sum(r['best_epoch']<=25 for r in scheduled),'schedule_selected_after_transition':sum(r['best_epoch']>25 for r in scheduled),'leave_one_fold_out':'descriptive omission of existing predictions; no retraining','uncertainty':'conditional on this image collection, split and three seeds; folds share training images; plant independence is unverified'}
    (out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
