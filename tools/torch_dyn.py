#!/usr/bin/env python3
# PyTorch refs for tensor-list/dynamic-shape ops. dynamic-count outputs: also emit the expected count.
import sys, numpy as np, torch
op, pre = sys.argv[1], sys.argv[2]
def save(t,nm=".out"): np.atleast_1d(np.asarray(t.detach().numpy(),dtype=np.float64).astype(np.float32).reshape(-1)).tofile(pre+nm)
def savec(n): np.array([n],dtype=np.int64).tofile(pre+".cnt")
if op=="scatternd":
    upd=torch.from_numpy(np.fromfile(pre+".u",dtype=np.float32).astype(np.float64)).reshape(2,3)
    out=torch.zeros(4,3,dtype=torch.float64); out[torch.tensor([0,2])]=upd; save(out)
elif op=="scatterndupdate":
    self=torch.from_numpy(np.fromfile(pre+".s",dtype=np.float32).astype(np.float64)).reshape(4,3)
    upd=torch.from_numpy(np.fromfile(pre+".u",dtype=np.float32).astype(np.float64)).reshape(2,3)
    out=self.clone(); out[torch.tensor([0,2])]=upd; save(out)
elif op=="unique":
    x=torch.from_numpy(np.fromfile(pre+".x",dtype=np.float32).astype(np.float64))
    u=torch.unique(x,sorted=True); save(u); savec(u.numel())
elif op=="nonzero":
    x=torch.from_numpy(np.fromfile(pre+".x",dtype=np.float32).astype(np.float64))
    nz=torch.nonzero(x).reshape(-1); save(nz.double()); savec(nz.numel())
else: sys.stderr.write("torch_dyn: no ref %s\n"%op); sys.exit(2)
