#!/usr/bin/env python3
# PyTorch references for the scatter / index_put family (forward), each matching the shim's ACTUAL
# convention (verified against the impl, not assumed from the op name):
#   scatter      = index_copy_(0, idx1d, src)   (1-D index along dim0, replace; unique idx)
#   scatteradd   = index_add_(0, idx1d, src)     (1-D index along dim0, accumulate; dups ok)
#   indexadd     = index_add_(0, idx1d, alpha*src)
#   indexcopy    = index_copy_(0, idx1d, src)
#   indexfill    = index_fill_(0, idx1d, value)
#   scattervalue = scatter_(dim, indexNd, value)  (full-shape index, real dim)
import sys, numpy as np, torch
op, pre = sys.argv[1], sys.argv[2]
def Lf(name, n, shape): return torch.from_numpy(np.fromfile(pre+name,dtype=np.float32).astype(np.float64)).reshape(shape)
def Li(name, shape):    return torch.from_numpy(np.fromfile(pre+name,dtype=np.int64)).reshape(shape)
def save(name, t): t.detach().numpy().astype(np.float32).tofile(pre+name)

SV,SROW,SL = 6,4,5                  # dim0 ops: self[SV,SROW], idx[SL], src[SL,SROW]
FV,FROW,FL = 6,4,3                  # indexfill: self[FV,FROW], idx[FL]
VR,VC,VK   = 4,6,3                  # scattervalue: self[VR,VC], idx[VR,VK], dim=1
ALPHA, FILLV, SVALUE = 2.0, 3.14, 2.71

if op in ("indexput","indexputadd"):     # IndexPutImpl: 1-D index on dim0, in-place replace/accumulate
    self=Lf(".self",SV*SROW,(SV,SROW)).clone(); vals=Lf(".src",SL*SROW,(SL,SROW)); idx=Li(".idx",(SL,))
    self.index_put_((idx,), vals, accumulate=(op=="indexputadd")); save(".out",self)
elif op in ("scatter","scatteradd","indexadd","indexcopy"):
    self=Lf(".self",SV*SROW,(SV,SROW)); src=Lf(".src",SL*SROW,(SL,SROW)); idx=Li(".idx",(SL,))
    out=self.clone()
    if   op=="scatter":   out.index_copy_(0,idx,src)
    elif op=="indexcopy": out.index_copy_(0,idx,src)
    elif op=="scatteradd":out.index_add_(0,idx,src)
    else:                 out.index_add_(0,idx,ALPHA*src)
    save(".out",out)
elif op=="indexfill":
    self=Lf(".self",FV*FROW,(FV,FROW)); idx=Li(".idx",(FL,))
    out=self.clone().index_fill_(0,idx,FILLV); save(".out",out)
elif op=="scattervalue":
    self=Lf(".self",VR*VC,(VR,VC)); idx=Li(".idx",(VR,VK))
    out=self.clone().scatter_(1,idx,SVALUE); save(".out",out)
else:
    sys.stderr.write("torch_grad4: no ref for %s\n"%op); sys.exit(2)
