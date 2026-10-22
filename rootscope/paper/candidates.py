import hashlib, itertools, math
import cv2, networkx as nx, numpy as np
from scipy.spatial import cKDTree
from .common import arc
from .geometry import graph_for, canonical, vector_curve
from . import hypothesis_graph as hg

GEOM=['log_length','straightness','vertical_extent_per_length','horizontal_extent_per_length',
 'log_median_radius','radius_p10_ratio','radius_p90_ratio','radius_cv','end_radius_min_ratio','end_radius_max_ratio',
 'turn_median','turn_p90','turn_sum_per100px','log_edge_count','max_join_per_radius','foreground_support','covered_graph_arc_per_length','log_length_per_radius']
VISUAL=['root_gray_mean','root_gray_std','root_gray_p10','root_gray_p90','side_contrast_mean','side_contrast_p10',
 'side_contrast_p90','positive_contrast_fraction','lab_b_mean','lab_b_std','lab_b_side_difference','root_gray_change_p90']


def sample_arc(points):
    """Include contact-joining line interiors, not just skeleton vertices."""
    p=np.asarray(points,float).reshape(-1,2)
    if len(p)<2:return p
    s=arc(p);keep=np.r_[True,np.diff(s)>1e-12];p=p[keep];s=s[keep]
    if len(p)<2:return p
    u=np.r_[np.arange(0,s[-1],1.),s[-1]]
    return np.column_stack([np.interp(u,s,p[:,k]) for k in range(2)])

def follow_all(graph,offset,G,pairs,flipped,width):
    available={k for _,_,k in G.edges(keys=True)};used=set();traces=[]
    def follow(port):
        points=[];eids=[];joins=[]
        while port[0] not in used:
            eid,end=port;used.add(eid);eids.append(eid);e=graph['edges'][eid];p=(e['points'][::-1] if end else e['points'])+offset
            if points:joins.append(float(np.linalg.norm(np.asarray(points[-1])-p[0])))
            points.extend(p[1 if points and np.array_equal(points[-1],p[0]) else 0:].tolist())
            port=pairs.get((eid,1-end))
            if port is None or port[0] in used:break
        p=np.array(points,int).reshape(-1,2)
        if len(p)<2 or arc(p)[-1]<10:return
        if flipped:p[:,1]=width-1-p[:,1]
        traces.append(dict(path=p,edge_ids=eids,max_join_gap_px=max(joins,default=0.)))
    ports=[]
    for eid in sorted(available):
        for end in [0,1]:
            if (eid,end) not in pairs:ports.append((eid,end))
    for p in ports:
        if p[0] not in used:follow(p)
    for eid in sorted(available-used):
        if eid not in used:follow((eid,0))
    assert used==available
    return traces

def features(t,gray,lab,hard,radii,length_by_edge):
    p=sample_arc(t['path']);q=p[::4]
    if len(q)<2:q=p
    pi=np.rint(q).astype(int);pi[:,0]=np.clip(pi[:,0],0,gray.shape[0]-1);pi[:,1]=np.clip(pi[:,1],0,gray.shape[1]-1)
    r=radii[pi[:,0],pi[:,1]];median=max(1.,float(np.median(r)));length=float(arc(p)[-1]);disp=float(np.linalg.norm(p[-1]-p[0]))
    v=np.diff(q,axis=0);vn=np.linalg.norm(v,axis=1);v=v/np.maximum(vn[:,None],1e-9);turn=np.arccos(np.clip(np.sum(v[1:]*v[:-1],axis=1),-1,1))
    if not len(turn):turn=np.array([0.])
    geom=[math.log1p(length),disp/max(1,length),np.ptp(p[:,0])/max(1,length),np.ptp(p[:,1])/max(1,length),
      math.log1p(median),np.percentile(r,10)/median,np.percentile(r,90)/median,np.std(r)/median,
      min(float(r[0]),float(r[-1]))/median,max(float(r[0]),float(r[-1]))/median,
      np.median(turn),np.percentile(turn,90),turn.sum()/max(1,length)*100,math.log1p(len(t['edge_ids'])),t['max_join_gap_px']/median,
      np.mean(hard[pi[:,0],pi[:,1]]>0),sum(length_by_edge.get(e,0.) for e in t['edge_ids'])/max(1,length),math.log1p(length/median)]
    tangent=np.gradient(q,axis=0);norm=np.linalg.norm(tangent,axis=1);normal=np.c_[-tangent[:,1],tangent[:,0]]/np.maximum(norm[:,None],1e-9)
    side=[];sideb=[]
    for sign in [-1,1]:
        s=np.rint(q+sign*normal*(r+3)[:,None]).astype(int);s[:,0]=np.clip(s[:,0],0,gray.shape[0]-1);s[:,1]=np.clip(s[:,1],0,gray.shape[1]-1)
        side.append(gray[s[:,0],s[:,1]].astype(float));sideb.append(lab[s[:,0],s[:,1],2].astype(float))
    g=gray[pi[:,0],pi[:,1]].astype(float);b=lab[pi[:,0],pi[:,1],2].astype(float);contrast=(np.mean(side,axis=0)-g)/255
    visual=[g.mean()/255,g.std()/255,np.percentile(g,10)/255,np.percentile(g,90)/255,
       contrast.mean(),np.percentile(contrast,10),np.percentile(contrast,90),np.mean(contrast>0),
       b.mean()/255,b.std()/255,np.mean(b-np.mean(sideb,axis=0))/255,np.percentile(abs(np.diff(g)),90)/255 if len(g)>1 else 0.]
    assert len(geom)==len(GEOM) and len(visual)==len(VISUAL)
    return list(map(float,geom+visual))

def generate(hard, photograph, anchor):
    flip=canonical(hard);m=hard[:,::-1].copy() if flip else hard
    graph,offset,G=graph_for(m);width=hard.shape[1];edge_ids={k for _,_,k in G.edges(keys=True)};lengths={eid:float(graph['edges'][eid]['length']) for eid in edge_ids}
    ports={n:[] for n in G}
    for eid in sorted(edge_ids):
        e=graph['edges'][eid];ports[e['a']].append((eid,0));ports[e['b']].append((eid,1))
    alltraces=[];settings=hg.RouteSettings()
    for mode in ['single_edges','original','direction','width','optional']:
        pairs={}
        for node,pp in ports.items():
            if mode=='single_edges':continue
            if len(pp)==2:pairs[pp[0]]=pp[1];pairs[pp[1]]=pp[0];continue
            if not 3<=len(pp)<=8:continue
            arms={p:hg._arm(graph,p,settings) for p in pp};valid=[p for p in pp if arms[p] is not None];C=nx.Graph();C.add_nodes_from(valid)
            for a,b in itertools.combinations(valid,2):
                cosine=float(np.clip(-arms[a]['direction']@arms[b]['direction'],-1,1));turn=math.acos(cosine)/(math.pi/2)
                ratio=abs(math.log(max(1.,arms[a]['width'])/max(1.,arms[b]['width'])))
                cost=float(hg._pair_cost(arms[a],arms[b])) if mode in ['original','optional'] else turn**2+(2. if mode=='width' else 0.)*ratio
                if mode=='optional':
                    if cost<1:C.add_edge(a,b,weight=1-cost)
                else:C.add_edge(a,b,weight=cost)
            chosen=nx.max_weight_matching(C,maxcardinality=False) if mode=='optional' else nx.min_weight_matching(C)
            for a,b in chosen:pairs[a]=b;pairs[b]=a
        for t in follow_all(graph,offset,G,pairs,flip,width):t['source']=mode;alltraces.append(t)
    unique={}
    for t in alltraces:
        k=tuple(sorted(t['edge_ids']))
        if k not in unique:unique[k]=t
    traces=list(unique.values());anchor=np.asarray(anchor,int).reshape(-1,2)
    tree=cKDTree(anchor);covered=[]
    for eid in sorted(edge_ids):
        p=graph['edges'][eid]['points']+offset
        if flip:p=p.copy();p[:,1]=width-1-p[:,1]
        if len(p) and np.mean(tree.query(p)[0]<=2.)>=.5:covered.append(eid)
    protected=dict(path=anchor,edge_ids=covered,max_join_gap_px=float(np.max(np.linalg.norm(np.diff(anchor,axis=0),axis=1))) if len(anchor)>1 else 0.,source='protected_v18_axis',protected=True)
    traces=[protected]+[dict(t,protected=False) for t in traces if not set(t['edge_ids'])&set(covered)]
    assert len(traces)<=2000,'Candidate cap exceeded: report failure, do not drop paths'
    gray=cv2.cvtColor(photograph,cv2.COLOR_BGR2GRAY);lab=cv2.cvtColor(photograph,cv2.COLOR_BGR2LAB);radii=cv2.distanceTransform(hard,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    for i,t in enumerate(traces):
        t['id']=i;t['features']=features(t,gray,lab,hard,radii,lengths)
        _,t['curve_length']=vector_curve(t['path']);t['raster_length']=float(arc(t['path'])[-1])
    lens=np.array(list(lengths.values()));rr=radii[hard>0];ncc=cv2.connectedComponents(hard,8)[0]-1
    counts=[sum(t['source']==mode for t in traces) for mode in ['single_edges','original','direction','width','optional']]
    cf=[math.log1p(hard.sum()),math.log1p(lens.sum()),math.log1p(len(lens)),math.log1p(ncc),
        math.log1p(np.median(rr)),float(np.std(rr)/max(1,np.mean(rr))),math.log1p(np.percentile(lens,50)),math.log1p(np.percentile(lens,90)),
        *[math.log1p(v) for v in counts],math.log1p(len(traces))]

    return dict(candidates=traces,edge_lengths=lengths), dict(x=[t['features'] for t in traces],count_features=cf,features=GEOM+VISUAL)
