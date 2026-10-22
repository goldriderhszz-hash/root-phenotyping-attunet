"""Reflection-consistent incoming-direction hypotheses and learned whole-path ranking."""
from pathlib import Path
from functools import lru_cache
import hashlib,json
import numpy as np
from .hypothesis_graph import select_path as propose,RouteSettings as ProposalSettings,arc
from .policy import rank,features
VERSION='mainroot-direction-20261016.15-deep-proposals'
DEFAULT_POLICY=Path(__file__).resolve().parent/'global_policies/all_references.json'
def canonical_flip(image):
    a=np.ascontiguousarray(image);b=np.ascontiguousarray(a[:,::-1])
    da=hashlib.sha256(a.tobytes()).digest();db=hashlib.sha256(b.tobytes()).digest()
    return db<da,da==db
@lru_cache(maxsize=12)
def load_policy(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def select_path(skeleton,mask,minimum_pixels=10,settings=None,base_hint=None,image=None,policy=None,extend=True):
    mask=np.uint8(np.asarray(mask)>0);h,w=mask.shape
    if not mask.any():return dict(path=[],length=None,status='empty_mask',flags=['empty_mask'],alternatives=[],length_range=[None,None],algorithm_version=VERSION,biological_identity_validated=False)
    if image is not None and image.shape[:2]!=mask.shape:raise ValueError('Photograph/mask shape mismatch')
    yy,xx=np.where(mask);y0,y1=yy.min(),yy.max()+1;x0,x1=xx.min(),xx.max()+1
    source=image[y0:y1,x0:x1] if image is not None else mask[y0:y1,x0:x1]
    flipped,symmetric=canonical_flip(source)
    m=mask[:,::-1].copy() if flipped else mask
    photo=image[:,::-1].copy() if flipped and image is not None else image
    hint=[base_hint[0],w-1-base_hint[1]] if flipped and base_hint is not None else base_hint
    sk=np.asarray(skeleton)[:,::-1].copy() if flipped else skeleton
    ps=ProposalSettings(crown_contacts=96,crown_trace_budget=768)
    raw=propose(sk,m,minimum_pixels,ps,hint,photo);pool=raw.get('all_crown_candidates') or raw.get('alternatives',[])
    def supported_origins(candidates,basal):
        # A disconnected one-pixel strand must not win solely by reaching deeper.
        # Retain thin entries in the same root component and strong stem evidence.
        if not basal:return candidates
        wide=max(basal,key=lambda b:b.get('entry_width_px',0.))
        width=wide.get('entry_width_px',0.)
        if width<=2:return candidates
        kept=[]
        for c in candidates:
            p0=np.asarray(c['path'][0]);b=min(basal,key=lambda b:np.linalg.norm(np.asarray(b['point'])-p0))
            same=b.get('connected_pixels')==wide.get('connected_pixels')
            if same or b.get('entry_width_px',0.)>=.35*width or b.get('photograph_support',0.)>.5:kept.append(c)
        return kept or candidates
    pool=supported_origins(pool,raw.get('basal_candidates',[]))
    if not pool:
        raw.update(algorithm_version=VERSION);return raw
    if policy is None:policy=load_policy(str(DEFAULT_POLICY))
    elif isinstance(policy,(str,Path)):policy=load_policy(str(policy))
    top=float(y0);extent=max(1.,float(y1-y0-1));order=rank(pool,policy,h,w,top,extent);chosen=pool[order[0]]
    initial_n=len(pool);expanded=False
    if False and (chosen['length']<.4*extent or (max(p[0] for p in chosen['path'])-top)<.55*extent):
        ps=ProposalSettings(crown_contacts=96,crown_trace_budget=768)
        raw=propose(sk,m,minimum_pixels,ps,hint,photo);pool=raw.get('all_crown_candidates') or raw.get('alternatives',[])
        pool=supported_origins(pool,raw.get('basal_candidates',[]))
        if pool:order=rank(pool,policy,h,w,top,extent);chosen=pool[order[0]];expanded=True
    x=np.asarray([features(pool[i],h,w,top,extent) for i in order[:3]])
    scores=((x-np.array(policy['mean']))/np.array(policy['scale']))@np.array(policy['coef'])
    gap=float(scores[0]-scores[1]) if len(scores)>1 else None
    flags=list(chosen.get('flags',[]))+[f for f in raw.get('flags',[]) if f in ['narrow_enclosed_lumen_filled','photograph_not_supplied','crown_multiple_basal_hypotheses']]
    if gap is not None and gap<.25:flags.append('root_identity_ambiguous')
    if chosen['length']<.2*extent:flags.append('short_route_requires_review')
    if raw.get('candidate_enumeration',{}).get('truncated'):flags.append('hypothesis_budget_reached')
    alternatives=[]
    def map_points(points):return [[int(y),int(w-1-x if flipped else x)] for y,x in points]
    for i in order[:6]:
        c=pool[i];alternatives.append(dict(path=map_points(c['path']),base=map_points([c['path'][0]])[0],tip=map_points([c['path'][-1]])[0],length=c['length'],flags=c.get('flags',[])))
    path=map_points(chosen['path']);length=float(arc(path)[-1]);status='candidate_uncertain' if flags else 'candidate'
    if symmetric and any(abs(p[1]-(w-1)/2)>.5 for p in path):
        flags.append('reflection_symmetric_identity_unresolvable');path=[];length=None;status='needs_review'
    return dict(path=path,length=length,status=status,flags=sorted(set(flags)),alternatives=alternatives,
        length_range=[None,None],algorithm_version=VERSION,biological_identity_validated=False,
        endpoint_status=chosen.get('endpoint_status'),candidate_count=len(pool),initial_candidate_count=initial_n,
        expanded_search=expanded,score_gap=gap,canonical_horizontal_reflection=flipped,
        selection_rule='Incoming-direction hypotheses, local edge non-repetition and whole-route pairwise preference scoring',
        policy_training_images=policy.get('training_images',[]),held_out_fold=policy.get('held_out_fold'),
        candidate_enumeration=raw.get('candidate_enumeration'),
        suppressed_burr_pixels=raw.get('suppressed_burr_pixels',0),filled_lumen_pixels=raw.get('filled_lumen_pixels',0))
