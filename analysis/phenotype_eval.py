"""Evaluate the unchanged phenotype extractor against annotation-mask-derived references."""
import argparse,json,os,sys,time,platform,inspect,hashlib
from pathlib import Path
from common import ROOT,OUT,SEED,read_mask,sha256,save_json,load_functions,bootstrap_ci
import cv2,numpy as np,pandas as pd,skimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from skimage.morphology import skeletonize

TRAITS=['length','count','junctions','angle']
SOURCES={'annotation_mask':ROOT/'data/masks','Exp4_Ours':ROOT/'artifacts/ablation/combined/predictions'}
RESULT=OUT/'results/phenotypes';FIG=OUT/'figures/phenotypes'

def observed_call(fn,mask):
    """Observe original return locals; never rewrite source, result, or function locals."""
    diag={}
    def trace(frame,event,arg):
        if frame.f_code is not fn.__code__:return None
        frame.f_trace_lines=False
        if event=='return':
            a=frame.f_locals.get('angles',[])
            diag['valid_angle_components']=len(a)
            diag['angle_is_measured']=len(a)>0
            diag['angle_is_default_60']=len(a)==0 and arg is not None and float(arg[3])==60.0
            diag['angle_no_measurement_zero']=len(a)==0 and arg is not None and float(arg[3])==0.0
            diag['raw_primary_path_found']=frame.f_locals.get('primary_path_raw') is not None
        return trace
    previous=sys.gettrace();start=time.perf_counter()
    try:
        sys.settrace(trace);values=fn(mask)
    finally:sys.settrace(previous)
    return list(map(float,values)),diag,time.perf_counter()-start

def cache_context():
    return {'numpy':np.__version__,'skimage':skimage.__version__,'cv2':cv2.__version__,'python':sys.version,
            'observer_sha256':hashlib.sha256(inspect.getsource(observed_call).encode()).hexdigest(),
            'mask_reader_sha256':hashlib.sha256(inspect.getsource(read_mask).encode()).hexdigest()}

def metrics(ref,pred):
    ref=np.asarray(ref,float);pred=np.asarray(pred,float);d=pred-ref;n=len(d)
    if not n:return {'n':0}
    cov=np.mean((ref-ref.mean())*(pred-pred.mean()))
    den=np.var(ref)+np.var(pred)+(ref.mean()-pred.mean())**2
    nz=ref!=0
    lo,hi=bootstrap_ci(d)
    return {'n':n,'reference_zero_n':int((~nz).sum()),'mae':np.mean(abs(d)),'rmse':np.sqrt(np.mean(d*d)),
            'bias_pred_minus_reference':d.mean(),'bias_CI_low':lo,'bias_CI_high':hi,
            'LoA_low':d.mean()-1.96*np.std(d,ddof=1) if n>1 else np.nan,
            'LoA_high':d.mean()+1.96*np.std(d,ddof=1) if n>1 else np.nan,
            'pearson_r2':np.corrcoef(ref,pred)[0,1]**2 if n>1 and np.std(ref)>0 and np.std(pred)>0 else np.nan,
            'predictive_R2':1-np.sum(d*d)/np.sum((ref-ref.mean())**2) if np.var(ref)>0 else np.nan,
            'CCC':2*cov/den if den>0 else np.nan,
            'MAPE_nonzero_reference_pct':np.mean(abs(d[nz]/ref[nz]))*100 if nz.any() else np.nan,
            'MAPE_effective_n':int(nz.sum())}

def plot_panel(pairs):
    plt.rcParams.update({'font.family':'Arial','font.size':9,
                         'axes.labelsize':9,'axes.titlesize':9.5,
                         'xtick.labelsize':8.5,'ytick.labelsize':8.5,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(4,2,figsize=(6.0,7.4),constrained_layout=True)
    titles=['Primary-path length','Lateral-fragment count','Junction-region count','Mean branching angle']
    for i,t in enumerate(TRAITS):
        d=pairs[pairs['trait']==t]
        if t=='angle':d=d[d['both_angles_measured']]
        x=d['reference'].to_numpy();y=d['prediction'].to_numpy();a=axs[i,0];b=axs[i,1]
        a.set_title(f'({chr(97+2*i)}) {titles[i]}',loc='left',fontweight='bold',fontsize=9.5)
        b.set_title(f'({chr(98+2*i)}) Bland–Altman',loc='left',fontweight='bold',fontsize=9.5)
        a.scatter(x,y,s=25,facecolor='white',edgecolor='#24746C',linewidth=1.1)
        if len(x):
            limits=[min(x.min(),y.min()),max(x.max(),y.max())];pad=max((limits[1]-limits[0])*.1,1)
            limits=[limits[0]-pad,limits[1]+pad];a.plot(limits,limits,'--',c='#73818D',lw=1,label='1:1')
            a.set_xlim(limits);a.set_ylim(limits);a.set_aspect('equal',adjustable='box')
            mean=(x+y)/2;diff=y-x;m=metrics(x,y)
            b.scatter(mean,diff,s=23,facecolor='white',edgecolor='#24746C',lw=1.1)
            b.axhline(0,c='#AAB3BC',lw=.8);b.axhline(m['bias_pred_minus_reference'],c='#24746C',lw=1.2)
            if len(x)>1:
                b.axhline(m['LoA_low'],c='#9A652A',ls='--',lw=1);b.axhline(m['LoA_high'],c='#9A652A',ls='--',lw=1)
            a.text(.03,.96,f'n = {len(x)}\nCCC = {m["CCC"]:.3f}',transform=a.transAxes,va='top',fontsize=8.5)
        a.set_xlabel('Annotation-mask reference');a.set_ylabel('Prediction-mask estimate')
        b.set_xlabel('Mean of paired values');b.set_ylabel('Prediction − reference')
        if t=='angle':
            a.set_xlabel('Annotation-mask reference (°)');a.set_ylabel('Prediction-mask estimate (°)')
            b.set_xlabel('Mean of paired values (°)');b.set_ylabel('Prediction − reference (°)')
        a.grid(alpha=.12);b.grid(alpha=.12)
    fig.suptitle('Propagation of segmentation differences through the unchanged extractor (Exp4)',fontsize=10.5)
    for ext in ['png','tiff','svg','pdf']:
        kwargs={'dpi':600,'facecolor':'white'}
        if ext=='tiff':kwargs['pil_kwargs']={'compression':'tiff_lzw'}
        fig.savefig(FIG/f'phenotype_agreement.{ext}',**kwargs)
    plt.close(fig)

def main():
    global RESULT,FIG
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'runs/evaluation/phenotypes')
    args=parser.parse_args()
    RESULT=args.output_dir.resolve();FIG=RESULT/'figures'
    if not RESULT.is_relative_to(ROOT):raise ValueError(f'Unsafe output location: {RESULT}')
    RESULT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    context_path=RESULT/'cache_context.json';context=cache_context()
    if context_path.exists():
        if json.loads(context_path.read_text(encoding='utf-8'))['context']!=context:
            if not args.refresh:raise RuntimeError('Observer or environment changed; explicit --refresh required')
            save_json(context_path,{'context':context,'note':'Context recorded before refreshing cases.'})
    elif any((RESULT/'cases').rglob('*.json')) and not args.refresh:
        raise RuntimeError('Existing cases require their original audited cache context; do not silently infer it.')
    else:save_json(context_path,{'context':context,'note':'Context recorded before computing cases.'})
    original=ROOT/'src/root_phenotyping/phenotypes.py'
    env=load_functions(original,['find_shortest_path','prune_skeleton','extract_phenotypes_decoupled'],{'os':os,'cv2':cv2,'np':np,'plt':plt,'stats':stats,'skeletonize':skeletonize})
    fn=env['extract_phenotypes_decoupled'];code_hash=sha256(original)
    split=json.loads((ROOT/'data/splits/seed42.json').read_text(encoding='utf-8'))
    names=sorted(split['test'])
    assert len(names)==10,'Expected complete 10-image supplied test set'
    allrows=[]
    for source,folder in SOURCES.items():
        for name in names:
            path=folder/(name+'_mask.png' if source=='annotation_mask' else name+'.tif')
            cache=RESULT/'cases'/source/(name+'.json');fingerprint={'input':sha256(path),'source_code':code_hash,'cv2':cv2.__version__}
            if cache.exists() and not args.refresh:
                try:record=json.loads(cache.read_text(encoding='utf-8'))
                except (json.JSONDecodeError,UnicodeError):record=None
                if record is not None and record['fingerprint']!=fingerprint:raise RuntimeError('Input or code changed; explicit --refresh required')
            else:record=None
            if record is not None:print('Resume:',source,name,flush=True)
            else:
                print('Computing:',source,name,flush=True)
                mask=read_mask(path);assert mask.shape==(3209,2311)
                values,diag,seconds=observed_call(fn,mask)
                record={'source':source,'image':name,'fingerprint':fingerprint,'values':dict(zip(TRAITS,values)),'diagnostics':diag,'seconds_with_state_observation':seconds}
                save_json(cache,record)
                print('Completed:',source,name,record['values'],f'{seconds:.2f}s',diag,flush=True)
            allrows.append({'source':source,'image':name,**record['values'],**record['diagnostics'],'seconds_with_state_observation':record['seconds_with_state_observation']})
    allrows=pd.DataFrame(allrows);allrows.to_csv(RESULT/'all_original_function_outputs.csv',index=False,encoding='utf-8-sig')
    ref=allrows[allrows.source=='annotation_mask'].set_index('image');pairs=[];summaries=[]
    for source in [s for s in SOURCES if s!='annotation_mask']:
        pred=allrows[allrows.source==source].set_index('image')
        for trait in TRAITS:
            both=(ref.angle_is_measured & pred.angle_is_measured).to_numpy()
            for j,name in enumerate(names):pairs.append({'source':source,'trait':trait,'image':name,'reference':ref.loc[name,trait],'prediction':pred.loc[name,trait],'both_angles_measured':bool(both[j])})
            for subset in (['all_original_outputs','valid_angles_only'] if trait=='angle' else ['all_original_outputs']):
                choose=both if subset=='valid_angles_only' else np.ones(len(names),bool)
                summaries.append({'source':source,'trait':trait,'analysis_set':subset,**metrics(ref[trait].to_numpy()[choose],pred[trait].to_numpy()[choose])})
    pairs=pd.DataFrame(pairs);pairs.to_csv(RESULT/'paired_phenotypes.csv',index=False,encoding='utf-8-sig')
    summary=pd.DataFrame(summaries);summary.to_csv(RESULT/'agreement_summary.csv',index=False,encoding='utf-8-sig')
    plot_panel(pairs[pairs.source=='Exp4_Ours'])
    save_json(RESULT/'run_metadata.json',{'original_code_sha256':code_hash,'python':sys.version,'platform':platform.platform(),'cv2':cv2.__version__,'seed':SEED,'bootstrap_repetitions':10000,'reference_type':'same unchanged algorithm applied to annotation mask; not independent manual measurement','input_sources':{s:str(p) for s,p in SOURCES.items()},'timing_note':'Contains state-observer overhead; not a pure extractor benchmark or validation of manuscript timing.'})
    print(summary.to_string(index=False),flush=True)
if __name__=='__main__':main()
