"""Experimental ordered-root geometry. No anatomical identity is inferred.

Continuation preserves the selected baseline exactly; counts and angles refer
to explicit graph attachments rather than disconnected leftover blobs.
"""
from pathlib import Path
import hashlib,heapq,math,sys
import cv2,networkx as nx,numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from .common import arc
from . import hypothesis_graph as hg
VERSION='curve-topology-20261018.18-experimental'

def canonical(mask):
    m=np.ascontiguousarray(mask,dtype=np.uint8)
    return hashlib.sha256(m[:,::-1].copy().tobytes()).digest()<hashlib.sha256(m.tobytes()).digest()

def vector_curve(points):
    p=np.asarray(points,float).reshape(-1,2)
    if len(p)<2:return p,0.
    s=arc(p);keep=np.r_[True,np.diff(s)>1e-10];p=p[keep];s=s[keep]
    if len(p)<2:return p,0.
    t=np.linspace(0,s[-1],max(2,int(math.ceil(s[-1]))+1))
    q=np.c_[np.interp(t,s,p[:,0]),np.interp(t,s,p[:,1])]
    if len(q)>=7:q=savgol_filter(q,7,2,axis=0,mode='interp')
    q[0]=p[0];q[-1]=p[-1]
    return q,float(arc(q)[-1])

def local_vector(points,start=0.,skip=8.,span=16.):
    p=np.asarray(points,float).reshape(-1,2)
    if len(p)<2:return None
    s=arc(p);a=start+skip;b=a+span
    if s[-1]+1e-8<b:return None
    qv=np.linspace(a,b,17);q=np.c_[np.interp(qv,s,p[:,0]),np.interp(qv,s,p[:,1])]
    if np.linalg.norm(q[-1]-q[0])<1e-8:return None
    _,_,v=np.linalg.svd(q-q.mean(axis=0),full_matrices=False);u=v[0]
    if u@(q[-1]-q[0])<0:u=-u
    return u

def connector(mask,a,b,maximum):
    a=tuple(map(int,a));b=tuple(map(int,b))
    if a==b:return [a]
    if np.linalg.norm(np.asarray(a)-b)>maximum:return None
    margin=int(math.ceil(maximum))+2
    y0=max(0,min(a[0],b[0])-margin);y1=min(mask.shape[0],max(a[0],b[0])+margin+1)
    x0=max(0,min(a[1],b[1])-margin);x1=min(mask.shape[1],max(a[1],b[1])+margin+1)
    if not(mask[a] and mask[b]):return None
    queue=[(0.,a)];dist={a:0.};parent={a:None}
    while queue:
        d,p=heapq.heappop(queue)
        if d!=dist[p] or d>maximum:continue
        if p==b:
            out=[p]
            while parent[out[-1]] is not None:out.append(parent[out[-1]])
            return out[::-1]
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                q=(p[0]+dy,p[1]+dx)
                if not(dy or dx) or not(y0<=q[0]<y1 and x0<=q[1]<x1) or not mask[q]:continue
                nd=d+math.hypot(dy,dx)
                if nd<dist.get(q,math.inf):dist[q]=nd;parent[q]=p;heapq.heappush(queue,(nd,q))
    return None

def extend_axis(baseline,alternatives,hard,weak):
    """Choose a longer tail only; never replace the existing root/crown path."""
    original=np.asarray(baseline,int).reshape(-1,2)
    if len(original)<2:return original,dict(status='no_baseline_axis',extended=False,baseline_prefix_exact=True),hard.copy()
    flip=canonical(hard);width=hard.shape[1]
    h=hard[:,::-1].copy() if flip else hard.copy();w=weak[:,::-1].copy() if flip else weak.copy()
    old=original.copy()
    if flip:old[:,1]=width-1-old[:,1]
    radius=cv2.distanceTransform(h,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    rr=radius[old[:,0],old[:,1]];tol=np.maximum(3.,rr+2.)
    r=max(1.,float(np.median(rr[-min(24,len(rr)):])))
    s=arc(old);context=max(24.,6*r)
    start=int(np.searchsorted(s,max(0.,s[-1]-context)))
    incoming=old[-1].astype(float)-old[start]
    if np.linalg.norm(incoming)<1e-8:return original,dict(status='unresolved_terminal_direction',extended=False,baseline_prefix_exact=True),hard.copy()
    incoming/=np.linalg.norm(incoming);past=old[:max(0,start)]
    past_tree=cKDTree(past) if len(past) else None
    proposals=[];audit=[]
    oldtree=cKDTree(old)
    for i,item in enumerate(alternatives):
        c=np.asarray(item.get('path',[]),int).reshape(-1,2)
        if len(c)<2:continue
        if flip:c[:,1]=width-1-c[:,1]
        dc,ix=cKDTree(c).query(old);j=int(ix[-1])
        rec=dict(index=i,status='not_ordered_full_prefix',tail_arc_px=0.)
        audit.append(rec)
        if np.mean(dc<=tol)<.98 or np.mean(np.diff(ix)>=-2)<.98 or j<1:continue
        cp=c[:j+1];dback,oldix=oldtree.query(cp)
        if np.mean(dback<=tol[oldix])<.98:continue
        tail=c[j:]
        if len(tail)<2:rec['status']='no_tail';continue
        length=float(arc(tail)[-1]);rec['tail_arc_px']=length
        if length<max(8.,2*r):rec['status']='tail_below_resolution_rule';continue
        q=float(min(arc(tail)[-1],context));t=arc(tail);end=np.array([np.interp(q,t,tail[:,0]),np.interp(q,t,tail[:,1])]);v=end-tail[0]
        if np.linalg.norm(v)<1e-8 or incoming@v/np.linalg.norm(v)<.5:rec['status']='terminal_direction_not_supported';continue
        con=connector(w,old[-1],tail[0],max(6.,3*r))
        if con is None:rec['status']='no_short_foreground_connector';continue
        con=np.asarray(con,int)
        extension=np.vstack([con[1:],tail[1:]])
        if len(extension)==0:rec['status']='empty_extension';continue
        # Keep a natural U return, but do not revisit the old root/crown.
        te=arc(np.vstack([old[-1],extension]));far=extension[te[1:]>2*r]
        if past_tree is not None and len(far) and np.any(past_tree.query(far)[0]<=max(1.,r)):
            rec['status']='returns_to_old_axis';continue
        old_pixels=set(map(tuple,old))
        if any(tuple(p) in old_pixels for p in extension) or len(set(map(tuple,extension)))!=len(extension):
            rec['status']='repeated_root_pixels';continue
        result=np.vstack([old,extension]);step=np.linalg.norm(np.diff(result,axis=0),axis=1)
        if np.any(step>math.sqrt(2)+1e-8):rec['status']='nonlocal_path_step';continue
        rec['status']='admissible_low_confidence_extension'
        added=float(arc(result)[-1]-s[-1]);proposals.append((added,i,result,extension))
    if not proposals:return original,dict(status='unchanged_no_supported_tail',extended=False,baseline_prefix_exact=True,proposals=audit),hard.copy()
    added,index,result,extension=max(proposals,key=lambda v:(v[0],-v[1]))
    line=np.zeros_like(h);cv2.polylines(line,[extension[:,::-1].astype(np.int32)],False,1,1)
    wr=cv2.distanceTransform(w,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    band=max(3.,float(np.median(wr[extension[:,0],extension[:,1]]))+1.)
    near=cv2.distanceTransform(1-line,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)<=band
    analysis=np.uint8((h>0)|((w>0)&near))
    if flip:result[:,1]=width-1-result[:,1];analysis=analysis[:,::-1].copy()
    assert np.array_equal(result[:len(original)],original)
    return result,dict(status='tentative_continuation_requires_review',extended=True,added_raster_arc_px=added,
        alternative_index=index,baseline_prefix_exact=True,proposals=audit),analysis

def graph_for(mask):
    m=np.uint8(mask>0);ys,xs=np.where(m)
    if not len(ys):return None,np.array([0,0]),nx.MultiGraph()
    y0,y1=ys.min(),ys.max()+1;x0,x1=xs.min(),xs.max()+1;offset=np.array([y0-16,x0-16])
    local=np.pad(m[y0:y1,x0:x1],16);sk=skeletonize(local>0).astype(np.uint8)
    graph=hg._graph(sk,local,suppress_burrs=False,bounded_contacts=True)
    G=nx.MultiGraph()
    for i,node in enumerate(graph['nodes']):G.add_node(i,point=np.asarray(node,int)+offset,radius=float(graph['radius'][tuple(node)]))
    for i,edge in enumerate(graph['edges']):
        G.add_edge(edge['a'],edge['b'],key=i,eid=i,length=edge['length'])
    # Raster X contacts can appear as two nearby Y nodes. Keep these as one
    # unresolved multi-arm contact, rather than counting two biological forks.
    # The complete cluster is bounded by its local diameter to avoid chaining
    # together many distinct attachments along an entire root.
    parent={n:n for n in G};members={n:{n} for n in G}
    def find(n):
        while parent[n]!=n:parent[n]=parent[parent[n]];n=parent[n]
        return n
    for a,b,eid,data in sorted(G.edges(keys=True,data=True),key=lambda e:(e[3]['length'],e[2])):
        if a==b or G.degree(a)<3 or G.degree(b)<3:continue
        x,y=find(a),find(b)
        if x==y:continue
        group=members[x]|members[y];points=np.array([G.nodes[n]['point'] for n in group])
        limit=2*max(G.nodes[n]['radius'] for n in group)+1.
        if G.degree(a)==3 and G.degree(b)==3:
            # At an acute X the merged shaft spans more than one diameter.
            # Its extent follows the local arm angle, not a fixed pixel cutoff.
            factors=[]
            for node in [a,b]:
                directions=[]
                for _,_,k in G.edges(node,keys=True):
                    if k==eid:continue
                    e=graph['edges'][k];q=e['points'] if e['a']==node else e['points'][::-1]
                    s=arc(q);j=min(len(q)-1,max(1,int(np.searchsorted(s,12.))))
                    v=q[j].astype(float)-q[0]
                    if np.linalg.norm(v)>0:directions.append(v/np.linalg.norm(v))
                if len(directions)==2:
                    theta=math.acos(float(np.clip(directions[0]@directions[1],-1,1)))
                    factors.append(1/max(.1,math.sin(theta/2)))
            if factors:limit=(limit-1)*max(factors)+1.
        diameter=float(np.linalg.norm(points[:,None,:]-points[None,:,:],axis=2).max())
        if data['length']<=limit and diameter<=limit:
            parent[y]=x;members[x]=group;del members[y]
    if any(find(n)!=n for n in G):
        groups={}
        for n in G:groups.setdefault(find(n),[]).append(n)
        new=nx.MultiGraph();representatives={}
        for root,group in groups.items():
            center=np.mean([G.nodes[n]['point'] for n in group],axis=0)
            rep=min(group,key=lambda n:(float(np.sum((G.nodes[n]['point']-center)**2)),n))
            for n in group:representatives[n]=rep
            new.add_node(rep,**G.nodes[rep])
        for a,b,eid,data in G.edges(keys=True,data=True):
            aa,bb=representatives[a],representatives[b]
            graph['edges'][eid]['a']=aa;graph['edges'][eid]['b']=bb
            if aa==bb and a!=b:continue
            new.add_edge(aa,bb,key=eid,**data)
        G=new
    return graph,offset,G

def longest_observed_axis(mask):
    """Fallback longest crown-to-terminal geodesic; ambiguous contacts flagged.

    For single-root masks this reconstructs a full U, irrespective of endpoint y.
    It is not the anatomical definition of a primary root.
    """
    flip=canonical(mask);m=mask[:,::-1].copy() if flip else mask
    graph,offset,G=graph_for(m)
    if graph is None or not G.number_of_edges():return np.empty((0,2),int),['no_connected_root_axis']
    ends=[n for n in G if G.degree(n)==1]
    if not ends:return np.empty((0,2),int),['closed_network_requires_review']
    start=min(ends,key=lambda n:tuple(G.nodes[n]['point']))
    distances,paths=nx.single_source_dijkstra(G,start,weight='length')
    reachable=[n for n in ends if n!=start and n in distances]
    if not reachable:return np.empty((0,2),int),['no_reachable_terminal']
    end=max(reachable,key=lambda n:(distances[n],-n));nodes=paths[end];out=[]
    for a,b in zip(nodes,nodes[1:]):
        eid=min(G[a][b],key=lambda k:G[a][b][k]['length']);e=graph['edges'][eid]
        p=e['points'] if e['a']==a else e['points'][::-1]
        out.extend((p+(offset))[1 if out else 0:].tolist())
    out=np.asarray(out,int)
    flags=[]
    if any(G.degree(n)>2 for n in nodes):flags.append('longest_geodesic_is_operational_not_anatomical_identity')
    if G.number_of_edges()-G.number_of_nodes()+nx.number_connected_components(G)>0:flags.append('cycle_shortcuts_unresolved')
    if flip:out[:,1]=mask.shape[1]-1-out[:,1]
    return out,flags

def describe(mask,axis,scale_mm_per_pixel=None):
    m=np.uint8(mask>0);p=np.asarray(axis,float).reshape(-1,2);width=m.shape[1]
    flip=canonical(m)
    if flip:m=m[:,::-1].copy();p=p.copy();p[:,1]=width-1-p[:,1]
    curve,length=vector_curve(p);raster=float(arc(p)[-1]) if len(p) else None
    graph,offset,G=graph_for(m)
    def mapping(a):
        q=np.asarray(a,float).reshape(-1,2).copy()
        if flip:q[:,1]=width-1-q[:,1]
        return q.tolist()
    flags=[];attachments=[];angle_records=[]
    if graph is None or len(p)<2:
        return dict(version=VERSION,axis_path_yx=mapping(p),axis_curve_yx=mapping(curve),axis_curve_length_px=length if len(p)>1 else None,
            axis_raster_length_px=raster,first_order_lateral_count=None,branch_junction_count=None,ambiguous_contact_count=None,
            mean_parent_child_angle_deg=None,mean_acute_parent_child_angle_deg=None,valid_angle_n=0,missing_angle_n=0,
            angles=[],attachments=[],flags=['empty_or_unavailable_axis'],scale_status='unknown',biological_identity_validated=False)
    tree=cKDTree(p);pa=arc(p);eid_axis={};edge_global={}
    for eid,e in enumerate(graph['edges']):
        q=e['points']+offset;edge_global[eid]=q
        d,_=tree.query(q);rad=np.asarray(e['widths'])/2
        eid_axis[eid]=float(np.mean(d<=np.maximum(1.5,.35*rad)))>=.8
    removed=[]
    while True:
        remove=[]
        for a,b,eid,data in list(G.edges(keys=True,data=True)):
            if a==b or eid_axis[eid]:continue
            terminal=a if G.degree(a)==1 else b if G.degree(b)==1 else None
            other=b if terminal==a else a
            if terminal is not None and G.degree(other)>=3 and data['length']<=G.nodes[other]['radius']+1:
                remove.append((a,b,eid));removed.append(eid)
        if not remove:break
        G.remove_edges_from(remove);G.remove_nodes_from(list(nx.isolates(G)))
    degree3=[n for n in G if G.degree(n)==3];complex_nodes=[n for n in G if G.degree(n)>=4]
    cycle_rank=G.number_of_edges()-G.number_of_nodes()+nx.number_connected_components(G) if len(G) else 0
    if complex_nodes:flags.append('multibranch_contacts_not_resolved_as_anatomical_branches')
    if cycle_rank:flags.append('cycles_or_overlapping_roots_require_review')
    def outpath(eid,node):
        e=graph['edges'][eid];return edge_global[eid] if e['a']==node else edge_global[eid][::-1]
    def trace(first,node):
        eid=first;current=node;visited=set();result=[];secondary=0
        while eid not in visited:
            visited.add(eid);e=graph['edges'][eid];q=outpath(eid,current)
            result.extend(q[1 if result else 0:].tolist());other=e['b'] if e['a']==current else e['a']
            choices=[k for _,_,k in G.edges(other,keys=True) if k not in visited and not eid_axis[k]]
            if not choices:break
            if len(choices)>1:secondary+=len(choices)-1
            incoming=q[-1].astype(float)-q[max(0,len(q)-min(16,len(q)))]
            def turn(k):
                a=outpath(k,other);s=arc(a);j=min(len(a)-1,max(1,int(np.searchsorted(s,12.))))
                v=a[j].astype(float)-a[0]
                return float(incoming@v/max(1e-9,np.linalg.norm(incoming)*np.linalg.norm(v)))
            eid=max(choices,key=lambda k:(turn(k),-k));current=other
        return np.asarray(result,float),secondary
    for n in G:
        point=G.nodes[n]['point'];distance,index=tree.query(point);radius=max(1.,G.nodes[n]['radius'])
        if distance>max(2.,radius+1.):continue
        ports=list(G.edges(n,keys=True));ax=[k for _,_,k in ports if eid_axis[k]];side=[k for _,_,k in ports if not eid_axis[k]]
        if G.degree(n)>=3 and (len(ax)!=2 or G.degree(n)>=4):
            attachments.append(dict(point_yx=mapping([point])[0],status='ambiguous_contact_or_axis_endpoint',arms=G.degree(n),axis_arms=len(ax)))
            continue
        if G.degree(n)!=3 or len(ax)!=2 or len(side)!=1:continue
        branch,secondary=trace(side[0],n);skip=max(2.,2*radius);span=max(8.,4*radius)
        parent=local_vector(p,float(pa[int(index)]),skip,span);child=local_vector(branch,0.,skip,span)
        rec=dict(attachment_yx=mapping([point])[0],axis_arc_px=float(pa[int(index)]),radius_px=radius,skip_arc_px=skip,span_arc_px=span,
          parent_child_angle_deg=None,acute_parent_child_angle_deg=None,vertical_orientation_deg=None,
          status='insufficient_parent_arc' if parent is None else 'insufficient_child_arc' if child is None else 'valid_projected_angle')
        if parent is not None and child is not None:
            angle=math.degrees(math.acos(float(np.clip(parent@child,-1,1))))
            vertical=math.degrees(math.acos(float(np.clip(child[0],-1,1))))
            rec.update(parent_child_angle_deg=angle,acute_parent_child_angle_deg=min(angle,180-angle),vertical_orientation_deg=min(vertical,180-vertical))
        angle_records.append(rec)
        attachments.append(dict(point_yx=mapping([point])[0],status='geometric_Y_attachment',secondary_forks_on_trace=secondary,
          branch_path_yx=mapping(branch),branch_curve_length_px=vector_curve(branch)[1]))
    valid=[r for r in angle_records if r['parent_child_angle_deg'] is not None]
    radius_path=cv2.distanceTransform(m,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[np.clip(np.rint(p[:,0]).astype(int),0,m.shape[0]-1),np.clip(np.rint(p[:,1]).astype(int),0,m.shape[1]-1)]
    result=dict(version=VERSION,axis_path_yx=mapping(p),axis_curve_yx=mapping(curve),axis_curve_length_px=length,axis_raster_length_px=raster,
      smoothing_length_change_px=length-raster,first_order_lateral_count=sum(r['status']=='geometric_Y_attachment' for r in attachments),
      branch_junction_count=len(degree3),ambiguous_contact_count=len(complex_nodes),all_junction_region_count=len(degree3)+len(complex_nodes),
      mean_parent_child_angle_deg=float(np.mean([r['parent_child_angle_deg'] for r in valid])) if valid else None,
      mean_acute_parent_child_angle_deg=float(np.mean([r['acute_parent_child_angle_deg'] for r in valid])) if valid else None,
      valid_angle_n=len(valid),missing_angle_n=len(angle_records)-len(valid),angles=angle_records,attachments=attachments,
      graph_nodes=G.number_of_nodes(),graph_edges=G.number_of_edges(),cycle_rank=cycle_rank,
      detached_graph_components=nx.number_connected_components(G) if len(G) else 0,removed_short_tip_spurs=len(removed),
      median_axis_radius_px=float(np.median(radius_path)),flags=flags,
      base_status='input_axis_start_not_independently_localized_root_collar',tip_status='visible_input_path_terminal_not_independently_verified',
      scale_status='calibrated_2D' if scale_mm_per_pixel is not None else 'unknown',
      axis_curve_length_mm=None if scale_mm_per_pixel is None else length*scale_mm_per_pixel,
      biological_identity_validated=False)
    return result
