import numpy as np, time
n=6375
A=np.random.rand(n,n).astype(np.float64)
B=np.random.rand(n,n).astype(np.float64)
t=time.time()
for _ in range(5): C=np.dot(A,B)
dt=(time.time()-t)/5
print(f"matmul {dt:.2f}s -> 1000 iters = {dt*1000/60:.0f} min")
