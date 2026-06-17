#!/usr/bin/env python3
# PyTorch refs for tensor-list/dynamic-shape ops. dynamic-count outputs: also emit the expected count.
from torch_common import *
op, pre = sys.argv[1], sys.argv[2]
def savec(n): savei(pre, ".cnt", torch.tensor([n]))
if op=="scatternd":
    upd=loadf(pre,".u",(2,3))
    out=torch.zeros(4,3,dtype=torch.float64); out[torch.tensor([0,2])]=upd; savef(pre,".out",out)
elif op=="scatterndupdate":
    self=loadf(pre,".s",(4,3))
    upd=loadf(pre,".u",(2,3))
    out=self.clone(); out[torch.tensor([0,2])]=upd; savef(pre,".out",out)
elif op=="unique":
    x=loadf(pre,".x")
    u=torch.unique(x,sorted=True); savef(pre,".out",u); savec(u.numel())
elif op=="nonzero":
    x=loadf(pre,".x")
    nz=torch.nonzero(x).reshape(-1); savef(pre,".out",nz.double()); savec(nz.numel())
else: no_ref("torch_dyn", op)
