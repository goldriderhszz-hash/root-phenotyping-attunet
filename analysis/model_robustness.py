"""Frozen existing weights and unchanged forward/sliding-window source; no retraining."""
import argparse,os,json,time,random,platform,sys,subprocess
from pathlib import Path
from common import ROOT,OUT,SEED,read_mask,sha256,save_json,load_functions,bootstrap_ci
import cv2,numpy as np,pandas as pd,skimage
from skimage.morphology import skeletonize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONDITIONS=['original','gamma_0.8','gamma_1.2','blur_sigma1.0']
RESULT=OUT/'results/robustness';FIG=OUT/'figures/robustness'
CONFIG={'patch_size':256,'overlap_ratio':0.5,'binary_threshold':0.53,
        'pad_to_multiple':256,'padding':'numpy reflect','CLAHE_clipLimit':2.0,'CLAHE_tileGridSize':[16,16],
        'gamma_order':'before padding and CLAHE','gamma_LUT':'round((v/255)^gamma*255)',
        'blur_kernel':[3,3],'blur_sigma':1.0,'blur_border':'BORDER_REFLECT_101',
        'GPU_warmup_patches':10,'evaluation_seed':SEED}

def plot_panel(table):
    """Create the publication figure from the saved per-image metrics."""
    plt.rcParams.update({'font.family':'Arial','font.size':9.5,
                         'axes.labelsize':9.5,'axes.titlesize':10,
                         'xtick.labelsize':8.5,'ytick.labelsize':8.5,
                         'legend.fontsize':8.5,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(1,2,figsize=(6.0,3.1),constrained_layout=True)
    labels=['Original','Gamma\n0.8','Gamma\n1.2','Gaussian blur\nσ = 1 px']
    panel_titles=['Pixel Dice','Hard-skeleton clDice']
    for panel,ax,metric,title in zip(('a','b'),axs,['dice','cldice'],panel_titles):
        p=table.pivot(index='image',columns='condition',values=metric)[CONDITIONS]*100
        for _,row in p.iterrows():
            ax.plot(range(4),row,marker='o',ms=3.5,lw=.8,alpha=.45,color='#8B9FA8')
        ax.plot(range(4),p.mean(),marker='o',ms=5.5,lw=2.2,color='#24746C',label='10-image mean')
        ax.set_xticks(range(4),labels)
        ax.tick_params(axis='x',pad=4)
        ax.set_ylabel(title+' (%)')
        ax.set_title(f'({panel}) {title}',loc='left',fontweight='bold',fontsize=10)
        ax.grid(axis='y',alpha=.15)
        ax.legend(frameon=False,loc='lower left')
    fig.suptitle('Frozen-model sensitivity to prespecified synthetic image changes',fontsize=10)
    for ext in ['png','tiff','pdf','svg']:
        kwargs={'dpi':600,'facecolor':'white'}
        if ext=='tiff':kwargs['pil_kwargs']={'compression':'tiff_lzw'}
        fig.savefig(FIG/f'fixed_model_input_sensitivity.{ext}',**kwargs)
    plt.close(fig)

def compare(g,p):
    g=np.asarray(g,bool);p=np.asarray(p,bool)
    tp=int((g&p).sum());fp=int((~g&p).sum());fn=int((g&~p).sum())
    sg=skeletonize(g);sp=skeletonize(p)
    tr=float((sg&p).sum()/sg.sum()) if sg.any() else np.nan
    tpr=float((sp&g).sum()/sp.sum()) if sp.any() else 0
    return {'tp':tp,'fp':fp,'fn':fn,'dice':2*tp/(2*tp+fp+fn),'iou':tp/(tp+fp+fn),
            'precision':tp/(tp+fp) if tp+fp else 0,'recall':tp/(tp+fn),
            'cldice':2*tr*tpr/(tr+tpr) if tr+tpr else 0}

def change_image(image,condition):
    if condition=='original':return image.copy()
    if condition.startswith('gamma_'):
        gamma=float(condition.split('_')[1]);lut=np.clip(np.round((np.arange(256)/255.0)**gamma*255),0,255).astype(np.uint8)
        return cv2.LUT(image,lut)
    if condition=='blur_sigma1.0':return cv2.GaussianBlur(image,(3,3),1.0,borderType=cv2.BORDER_REFLECT_101)
    raise ValueError(condition)

def main():
    global RESULT,FIG
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'runs/evaluation/robustness')
    args=parser.parse_args()
    RESULT=args.output_dir.resolve();FIG=RESULT/'figures'
    if not RESULT.is_relative_to(ROOT):raise ValueError(f'Unsafe output location: {RESULT}')
    try:import torch,torch.nn as nn
    except ImportError as e:raise RuntimeError('PyTorch is required; install listed CUDA dependency. No artificial results are generated.') from e
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; this run is not silently replaced by CPU benchmarking.')
    RESULT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda');torch.manual_seed(42)
    source=ROOT/'src/root_phenotyping/pipeline.py'
    definitions=['DoubleConv','AttentionGate','AttentionUNet','get_gaussian_window','gaussian_sliding_window_predict']
    env=load_functions(source,definitions,{'torch':torch,'nn':nn,'np':np,'DEVICE':device})
    checkpoint=ROOT/'models/combined_attention_unet_cldice.pth'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=env['AttentionUNet']().to(device)
    model.load_state_dict(state,strict=True);model.eval()
    predict=env['gaussian_sliding_window_predict']
    from scipy import stats
    phsrc=ROOT/'src/root_phenotyping/phenotypes.py'
    ph=load_functions(phsrc,['find_shortest_path','prune_skeleton','extract_phenotypes_decoupled'],{'os':os,'cv2':cv2,'np':np,'plt':plt,'stats':stats,'skeletonize':skeletonize})['extract_phenotypes_decoupled']
    core_hash=sha256(source);weight_hash=sha256(checkpoint)
    split=json.loads((ROOT/'data/splits/seed42.json').read_text(encoding='utf-8'))
    names=sorted(split['test'])
    if len(names)!=10:raise ValueError('Requires complete existing 10-image test set')
    try:gpu_info=subprocess.check_output(['C:/Windows/system32/nvidia-smi.exe','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    except Exception:gpu_info=torch.cuda.get_device_name(0)
    metadata={'status':'running','checkpoint':str(checkpoint),'checkpoint_sha256':weight_hash,'source_sha256':core_hash,
              'source_definitions_executed_without_changes':definitions,'python':sys.version,'torch':torch.__version__,'CUDA':torch.version.cuda,
              'gpu':gpu_info,'cpu':platform.processor(),'platform':platform.platform(),'cv2':cv2.__version__,'cudnn':torch.backends.cudnn.version(),
              'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
              'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'seed':SEED,'conditions':CONDITIONS,'binary_threshold':.53,
              'numpy':np.__version__,'skimage':skimage.__version__,'configuration':CONFIG,
              'wrapper_sha256':sha256(__file__),'common_sha256':sha256(Path(__file__).with_name('common.py')),
              'phenotype_source_sha256':sha256(phsrc),
              'notes':'Synthetic input sensitivity on the already-used 10-image set, not new biological replicates or external validation. No parameter/threshold search. All source functions unchanged. Timings exclude startup, metric computation and disk output; perturbation and phenotype timings separate.'}
    execution_fingerprint={k:v for k,v in metadata.items() if k not in ['status','notes']}
    save_json(RESULT/'run_metadata.json',metadata)
    print('Hardware:',gpu_info,'Torch:',torch.__version__,'strict weight loading PASS',flush=True)
    with torch.no_grad():
        for _ in range(10):model(torch.zeros(1,1,256,256,device=device))
    torch.cuda.synchronize()
    cases=[(name,c) for name in names for c in CONDITIONS];random.Random(SEED).shuffle(cases)
    rows=[]
    for n,(name,condition) in enumerate(cases,1):
        inp=ROOT/'data/images'/f'{name}.tif';gtp=ROOT/'data/masks'/f'{name}_mask.png'
        oldpred=ROOT/'artifacts/ablation/combined/predictions'/f'{name}.tif'
        file=RESULT/'cases'/condition/(name+'.json');mp=RESULT/'predictions'/condition/(name+'.png')
        fingerprint={'image_sha256':sha256(inp),'gt_sha256':sha256(gtp),'saved_Exp4_sha256':sha256(oldpred),
                     'condition':condition,'execution':execution_fingerprint}
        if file.exists() and mp.exists() and not args.refresh:
            try:row=json.loads(file.read_text(encoding='utf-8'))
            except (json.JSONDecodeError,UnicodeError):row=None
            if row is not None:
                if row.get('fingerprint')!=fingerprint:raise RuntimeError('Cache mismatch; explicit refresh required')
                if sha256(mp)==row.get('prediction_sha256'):
                    rows.append(row);print(f'Resume {n}/40:',name,condition,flush=True);continue
                print('Recompute case with incomplete prediction:',name,condition,flush=True)
        print(f'Inference {n}/40:',name,condition,flush=True)
        t=time.perf_counter();image=read_mask(inp);io_seconds=time.perf_counter()-t
        gt=read_mask(gtp)>127
        assert image.shape==gt.shape==(3209,2311)
        t=time.perf_counter();image=change_image(image,condition);perturbation_seconds=time.perf_counter()-t
        t=time.perf_counter();h,w=image.shape
        pad_h=(256-h%256)%256;pad_w=(256-w%256)%256
        padded=np.pad(image,((0,pad_h),(0,pad_w)),mode='reflect')
        prep=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(16,16)).apply(padded).astype(np.float32)/255.0
        preprocessing_seconds=time.perf_counter()-t
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        probability=predict(model,prep,patch_size=256,overlap_ratio=.5)
        torch.cuda.synchronize();inference_seconds=time.perf_counter()-t
        peak_bytes=torch.cuda.max_memory_allocated()
        t=time.perf_counter();pred=(probability[:h,:w]>.53).astype(np.uint8)*255;postprocessing_seconds=time.perf_counter()-t
        score=compare(gt,pred>127)
        row={'image':name,'condition':condition,'fingerprint':fingerprint,**score,'image_width':w,'image_height':h,
             'io_seconds':io_seconds,'external_perturbation_seconds':perturbation_seconds,'preprocessing_seconds':preprocessing_seconds,
             'inference_seconds':inference_seconds,'postprocessing_seconds':postprocessing_seconds,'peak_GPU_allocated_bytes':peak_bytes}
        if condition=='original':
            old=read_mask(oldpred)>127;row['different_pixels_from_saved_Exp4']=int(np.count_nonzero(old!=(pred>127)))
            row['dice_against_saved_Exp4']=compare(old,pred>127)['dice']
            t=time.perf_counter();values=ph(pred);row['phenotype_seconds_uninstrumented']=time.perf_counter()-t
            row['phenotype_original_outputs']=list(map(float,values))
            row['original_pipeline_seconds']=sum([io_seconds,preprocessing_seconds,inference_seconds,postprocessing_seconds,row['phenotype_seconds_uninstrumented']])
        mp.parent.mkdir(parents=True,exist_ok=True);cv2.imencode('.png',pred)[1].tofile(str(mp));row['prediction_sha256']=sha256(mp)
        save_json(file,row);rows.append(row)
        print('Done:',name,condition,f'Dice={score["dice"]:.5f}, inference={inference_seconds:.2f}s',flush=True)
        del probability,prep,padded,pred
    table=pd.DataFrame([{k:v for k,v in r.items() if k not in ['fingerprint','phenotype_original_outputs']} for r in rows]).sort_values(['image','condition'])
    table.to_csv(RESULT/'all40_inference_metrics_and_times.csv',index=False,encoding='utf-8-sig')
    baseline=table[table.condition=='original'].set_index('image');summary=[]
    for condition in CONDITIONS:
        d=table[table.condition==condition].set_index('image').loc[baseline.index]
        for metric in ['dice','iou','cldice','precision','recall']:
            change=d[metric].to_numpy()-baseline[metric].to_numpy();ci=bootstrap_ci(change)
            summary.append({'condition':condition,'metric':metric,'n':len(d),'mean':d[metric].mean(),'change_from_original':change.mean(),'change_CI_low':ci[0],'change_CI_high':ci[1],
                            'minimum_image_value':d[metric].min(),'maximum_image_value':d[metric].max()})
    pd.DataFrame(summary).to_csv(RESULT/'condition_paired_summary.csv',index=False,encoding='utf-8-sig')
    plot_panel(table)
    metadata['status']='complete';metadata['completed_cases']=len(table)
    metadata['saved_Exp4_exact_reproduction_images']=int((baseline.different_pixels_from_saved_Exp4==0).sum())
    metadata['source_weight_hash_after']=sha256(checkpoint);assert metadata['source_weight_hash_after']==weight_hash
    save_json(RESULT/'run_metadata.json',metadata)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print('Original-case listed processing-stage timing:',baseline.original_pipeline_seconds.describe().to_dict(),flush=True)
if __name__=='__main__':main()
