import numpy as np
def arc(points):
    p=np.asarray(points,float).reshape(-1,2)
    return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))] if len(p) else np.array([])
