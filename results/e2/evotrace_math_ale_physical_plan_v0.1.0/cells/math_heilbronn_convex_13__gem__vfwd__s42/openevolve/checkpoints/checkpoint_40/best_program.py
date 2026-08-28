# EVOLVE-BLOCK-START
import numpy as np

def _min_area(pts):
    n = len(pts)
    m = 1e9
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = 0.5*abs((pts[j,0]-pts[i,0])*(pts[k,1]-pts[i,1])-(pts[k,0]-pts[i,0])*(pts[j,1]-pts[i,1]))
                if a < m: m = a
    return m

def _min_area_with_idx(pts, idx):
    """Min triangle area involving point idx (faster for single-point moves)."""
    n = len(pts)
    m = 1e9
    for j in range(n):
        if j == idx: continue
        for k in range(j+1, n):
            if k == idx: continue
            i = idx
            a = 0.5*abs((pts[j,0]-pts[i,0])*(pts[k,1]-pts[i,1])-(pts[k,0]-pts[i,0])*(pts[j,1]-pts[i,1]))
            if a < m: m = a
    return m

def heilbronn_convex13():
    n = 13
    # Two concentric hexagonal rings + center
    ai = np.linspace(0, 2*np.pi, 6, endpoint=False)
    ao = np.linspace(np.pi/6, 2*np.pi+np.pi/6, 6, endpoint=False)
    pts = np.vstack([
        np.column_stack([0.5+0.35*np.cos(ai), 0.5+0.35*np.sin(ai)]),
        np.column_stack([0.5+0.48*np.cos(ao), 0.5+0.48*np.sin(ao)]),
        np.array([[0.5, 0.5]])
    ])
    # Gradient-free local search: for each point, try 8 directions
    rng = np.random.default_rng(42)
    for it in range(200):
        step = 0.08*(1-it/200)+0.003
        idx = rng.integers(0, n)
        cur = _min_area(pts)
        best = pts[idx].copy()
        best_v = cur
        for d in range(8):
            ang = d*np.pi/4
            np_ = pts.copy()
            np_[idx] = np.clip(pts[idx]+step*np.array([np.cos(ang),np.sin(ang)]),0,1)
            v = _min_area(np_)
            if v > best_v:
                best_v = v
                best = np_[idx]
        pts[idx] = best
    # Scale to fill square
    mn, mx = pts.min(0), pts.max(0)
    c = (mn+mx)/2
    s = 0.95/np.max(mx-mn) if np.max(mx-mn)>0 else 1
    return (pts-c)*s+0.5

# EVOLVE-BLOCK-END
