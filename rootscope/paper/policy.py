import math
import numpy as np

def features(c,height,width,top,extent):
    p=np.asarray(c['path'],float);ds=np.linalg.norm(np.diff(p,axis=0),axis=1);s=np.r_[0,np.cumsum(ds)];length=max(1,float(s[-1]));extent=max(1.,extent)
    samples=np.linspace(0,length,65);q=np.c_[np.interp(samples,s,p[:,0]),np.interp(samples,s,p[:,1])];v=np.diff(q,axis=0);vn=v/np.maximum(1e-6,np.linalg.norm(v,axis=1))[:,None]
    turns=np.arccos(np.clip(np.sum(vn[1:]*vn[:-1],axis=1),-1,1))/np.pi
    contacts=c.get('contacts',[]);angles=np.array([r.get('turn_angle_deg',math.degrees(math.sqrt(max(0,r.get('continuation_cost',0)))*math.pi/2)) for r in contacts])/180
    iw=np.array([r.get('incoming_width_px',1.) for r in contacts]);ow=np.array([r.get('outgoing_width_px',r.get('incoming_width_px',1.)) for r in contacts])
    wr=np.clip(np.log(np.maximum(1,ow)/np.maximum(1,iw)),-3,3) if len(iw) else np.array([0.])
    first=min(length,.12*extent);fi=np.searchsorted(s,first);initial=p[min(fi,len(p)-1)]-p[0]
    medw=float(np.median(iw)) if len(iw) else max(1.,extent*.002)
    return [math.log1p(length/extent),float(np.ptp(p[:,0])/extent),float((p[-1,0]-top)/extent),float((p[0,0]-top)/extent),
        float(np.linalg.norm(p[-1]-p[0])/length),len(contacts)*medw/length,
        float(angles.mean()) if len(angles) else 0.,float(angles.max()) if len(angles) else 0.,float(np.abs(wr).mean()),float(np.abs(wr).max()),
        math.log1p(float(np.median(ow))/extent*100) if len(ow) else 0.,float(np.median(wr)),
        float(initial[0]/max(1.,np.linalg.norm(initial))),float(np.ptp(p[:fi+1,0])/max(1,first)),
        float(np.mean(vn[:,0]>0)),float(np.mean(turns>.25)),float(turns.mean()),
        float(np.clip(np.log(max(1,ow[-1])/max(1,iw[0])),-3,3)) if len(iw) else 0.,
        float(min(p[-1,0],p[-1,1],height-1-p[-1,0],width-1-p[-1,1])<=3),
        float((p[-1,0]-p[0,0])<.2*extent and length<.4*extent)]

def rank(pool,model,height,width,top,extent):
    if not pool:return []
    x=np.asarray([features(c,height,width,top,extent) for c in pool]);values=((x-np.array(model['mean']))/np.array(model['scale']))@np.array(model['coef'])
    return sorted(range(len(pool)),key=lambda i:-round(float(values[i]),9))
