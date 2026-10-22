import time
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix

def select(pool,predicted_errors,estimated_count,use_count_penalty=True):
    candidates=pool['candidates'];n=len(candidates);lengths={int(k):v for k,v in pool['edge_lengths'].items()};eids=sorted(lengths);pos={eid:i for i,eid in enumerate(eids)}
    rr=[];cc=[];vv=[]
    for j,c in enumerate(candidates):
        for eid in c['edge_ids']:rr.append(pos[eid]);cc.append(j);vv.append(1.)
    q,frag,mix=np.asarray(predicted_errors).T;meanroot=sum(lengths.values())/max(1.,estimated_count)
    coverage=np.array([sum(lengths[e] for e in c['edge_ids'])/max(1,meanroot) for c in candidates])
    utility=2*q-1-frag-mix+coverage
    # Two nonnegative slack variables implement absolute count deviation.
    countrow=len(eids);rr.extend([countrow]*(n+2));cc.extend(range(n+2));vv.extend([1.]*n+[-1.,1.])
    A=coo_matrix((vv,(rr,cc)),shape=(len(eids)+1,n+2)).tocsr()
    lo=np.r_[np.full(len(eids),-np.inf),estimated_count];hi=np.r_[np.ones(len(eids)),estimated_count]
    lower=np.zeros(n+2);upper=np.r_[np.ones(n),np.inf,np.inf];lower[0]=upper[0]=1.
    cost=np.r_[-utility,[1.,1.] if use_count_penalty else [0.,0.]]
    start=time.perf_counter();result=milp(cost,integrality=np.r_[np.ones(n),[0,0]],bounds=Bounds(lower,upper),constraints=LinearConstraint(A,lo,hi),
        options={'time_limit':8.,'mip_rel_gap':1e-6,'presolve':True})
    if result.x is None:raise RuntimeError(f'No feasible decoder result: {result.message}')
    chosen=np.flatnonzero(result.x[:n]>.5).tolist();assert 0 in chosen
    used=[eid for j in chosen for eid in candidates[j]['edge_ids']];assert len(used)==len(set(used))
    return chosen,dict(solver_status=int(result.status),solver_message=result.message,mip_gap=float(getattr(result,'mip_gap',float('nan'))),seconds=time.perf_counter()-start,
        estimated_count=float(estimated_count),selected_count=len(chosen),graph_edge_exclusive=True,protected_axis_retained=True,
        use_count_penalty=use_count_penalty,reference_count_passed_to_decoder=False)
