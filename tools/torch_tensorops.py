#!/usr/bin/env python3
# PyTorch references for elementwise / tensor-manip / sort-select ops. Conventions verified from the
# shim (SWhere cond nonzero->self; MaskedFill mask nonzero->value; Lerp weight scalar; OneHot ids int64).
from torch_common import *
op, pre = sys.argv[1], sys.argv[2]
N, n, Rr, Cc, NC = 64, 5, 4, 6, 5
def Lf(name,shape=None): return loadf(pre, name, shape)

if   op=="clamp":     savef(pre,".out",Lf(".x").clamp(-0.3,0.5))
elif op=="threshold": savef(pre,".out",F.threshold(Lf(".x"),0.0,-1.0))
elif op=="lerp":      savef(pre,".out",torch.lerp(Lf(".x"),Lf(".end"),0.3))
elif op=="addcmul":   savef(pre,".out",torch.addcmul(Lf(".x"),Lf(".t1"),Lf(".t2"),value=0.5))
elif op=="addcdiv":   savef(pre,".out",torch.addcdiv(Lf(".x"),Lf(".t1"),Lf(".t2"),value=0.5))
elif op=="swhere":    savef(pre,".out",torch.where(loadu8(pre,".c").bool(),Lf(".x"),Lf(".y")))
elif op=="maskedfill":savef(pre,".out",Lf(".x").masked_fill(loadu8(pre,".m").bool(),9.0))
elif op=="maskedselect":
    r=torch.masked_select(Lf(".x"),loadu8(pre,".m").bool()); savef(pre,".out",r); savei(pre,".cnt",torch.tensor([r.numel()]))
elif op=="tril":      savef(pre,".out",torch.tril(Lf(".A",(n,n)),0))
elif op=="triu":      savef(pre,".out",torch.triu(Lf(".A",(n,n)),0))
elif op=="flip":      savef(pre,".out",torch.flip(Lf(".x",(Rr,Cc)),[1]))
elif op=="roll":      savef(pre,".out",torch.roll(Lf(".x",(Rr,Cc)),2,1))
elif op=="onehot":    savef(pre,".out",F.one_hot(loadi(pre,".ids"),NC).double())
elif op=="sort":      savef(pre,".out",torch.sort(Lf(".x",(Rr,Cc)),1,descending=False).values)
elif op=="topk":      savef(pre,".out",torch.topk(Lf(".x",(Rr,Cc)),3,1,largest=True).values)
elif op=="argsort":   savef(pre,".out",torch.argsort(Lf(".x",(Rr,Cc)),1,descending=False).double())
elif op=="kthvalue":  savef(pre,".out",torch.kthvalue(Lf(".x",(Rr,Cc)),2,1).values)
else: no_ref("torch_tensorops", op)
