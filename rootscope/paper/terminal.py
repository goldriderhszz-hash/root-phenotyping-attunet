import cv2
import numpy as np
from scipy.spatial import cKDTree
from .common import arc
from .geometry import canonical, extend_axis

def local_extend(baseline,alternatives,hard,weak):
    old=np.asarray(baseline,int).reshape(-1,2);width=hard.shape[1]
    if len(old)<2:return old,dict(extended=False,status='no_original_axis',baseline_prefix_exact=True),hard.copy()
    flip=canonical(hard)
    h=hard[:,::-1].copy() if flip else hard.copy();w=weak[:,::-1].copy() if flip else weak.copy()
    p=old.copy()
    if flip:p[:,1]=width-1-p[:,1]
    radius=cv2.distanceTransform(h,cv2.DIST_L2,cv2.DIST_MASK_PRECISE);r=max(1.,float(np.median(radius[p[-min(24,len(p)):,0],p[-min(24,len(p)):,1]])))
    s=arc(p);index=int(np.searchsorted(s,max(0.,s[-1]-max(64.,12*r))))
    suffix=p[index:];alts=[]
    for a in alternatives:
        q=np.asarray(a.get('path',[]),int).reshape(-1,2)
        if len(q)<2:continue
        if flip:q[:,1]=width-1-q[:,1]
        j=int(np.argmin(np.linalg.norm(q-suffix[0],axis=1)));alts.append(dict(path=q[j:].tolist()))
    extended,meta,mask=extend_axis(suffix,alts,h,w)
    if not meta.get('extended'):
        return old,dict(**meta,terminal_context_px=max(64.,12*r)),hard.copy()
    extra=extended[len(suffix):]
    past=p[:index];dist=cKDTree(past).query(extra)[0] if len(past) and len(extra) else np.array([])
    if len(dist) and np.any(dist<=max(1.,r)):
        return old,dict(extended=False,status='extension_revisits_older_axis',baseline_prefix_exact=True),hard.copy()
    if any(tuple(q) in set(map(tuple,p)) for q in extra):
        return old,dict(extended=False,status='extension_repeats_original_pixel',baseline_prefix_exact=True),hard.copy()
    result=np.vstack([p[:index],extended]);assert np.array_equal(result[:len(p)],p)
    if flip:result[:,1]=width-1-result[:,1];mask=mask[:,::-1].copy()
    assert np.array_equal(result[:len(old)],old)
    return result,dict(**meta,terminal_context_px=max(64.,12*r),full_original_axis_preserved=True),mask
