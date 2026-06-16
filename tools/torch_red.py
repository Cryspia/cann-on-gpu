#!/usr/bin/env python3
# PyTorch references for reduction/scan extras (dim=1 of x[Rr,Cc]). bool/count outputs compared as float.
import sys, numpy as np, torch
op, pre = sys.argv[1], sys.argv[2]
Rr,Cc = 4,6
x = torch.from_numpy(np.fromfile(pre+".x",dtype=np.float32).astype(np.float64)).reshape(Rr,Cc)
def save(t,nm=".out"): np.atleast_1d(np.asarray(t.detach().numpy(),dtype=np.float32)).tofile(pre+nm)
if   op=="all":    save(torch.all(x!=0,dim=1).double())
elif op=="any":    save(torch.any(x!=0,dim=1).double())
elif op=="maxdim": save(torch.max(x,1).values)
elif op=="mindim": save(torch.min(x,1).values)
elif op=="aminmax":
    mn,mx=torch.aminmax(x,dim=1); save(mn,".outmin"); save(mx,".outmax")
elif op=="cummax": save(torch.cummax(x,1).values)
elif op=="cummin": save(torch.cummin(x,1).values)
elif op=="logcumsumexp": save(torch.logcumsumexp(x,1))
elif op=="mode":   save(torch.mode(x,1).values)
elif op=="nansum": save(torch.nansum(x,1))
elif op=="nanmean":save(torch.nanmean(x,1))
elif op=="quantile": save(torch.quantile(x,0.5,dim=1))
elif op=="countnonzero": save(torch.count_nonzero(x,1).double())
else: sys.stderr.write("torch_red: no ref %s\n"%op); sys.exit(2)
