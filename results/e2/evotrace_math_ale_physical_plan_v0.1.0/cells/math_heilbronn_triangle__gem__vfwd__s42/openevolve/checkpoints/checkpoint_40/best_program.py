# EVOLVE-BLOCK-START
import numpy as np

def heilbronn_triangle11() -> np.ndarray:
    s3 = np.sqrt(3)
    h = s3 / 2.0
    # Equilateral triangle: (0,0), (1,0), (0.5,h)
    # Initial well-spread config inside triangle
    p = np.array([
        [0.15, 0.05], [0.45, 0.05], [0.75, 0.05],
        [0.25, 0.22], [0.55, 0.22], [0.80, 0.15],
        [0.35, 0.40], [0.60, 0.40],
        [0.45, 0.55], [0.55, 0.55],
        [0.50, 0.15]
    ])
    # Verify all inside: y >= 0, x >= 0, y <= s3*(1-x)
    for i in range(11):
        x, y = p[i]
        if y > s3*(1-x) - 1e-9:
            p[i,1] = s3*(1-x) - 1e-6
        if x < 0: p[i,0] = 1e-6
        if y < 0: p[i,1] = 1e-6
    # Simple hill climbing
    np.random.seed(42)
    def marea(pts):
        n=len(pts); m=1e9
        for i in range(n):
            for j in range(i+1,n):
                for k in range(j+1,n):
                    a=0.5*abs((pts[j,0]-pts[i,0])*(pts[k,1]-pts[i,1])-(pts[k,0]-pts[i,0])*(pts[j,1]-pts[i,1]))
                    if a<m: m=a
        return m
    ba=marea(p)
    for _ in range(1500):
        i=np.random.randint(11)
        d=np.random.randn(2)*0.03
        t=p.copy(); t[i]=p[i]+d
        x,y=t[i]
        if x>0 and y>0 and y<s3*(1-x):
            a=marea(t)
            if a>ba: p,ba=t,a
    return p
# EVOLVE-BLOCK-END
