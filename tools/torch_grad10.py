#!/usr/bin/env python3
# PyTorch references for elementwise / tensor-manip / sort-select ops. Conventions verified from the
# shim (SWhere cond nonzero->self; MaskedFill mask nonzero->value; Lerp weight scalar; OneHot ids int64).
import sys, numpy as np, torch, torch.nn.functional as F
op, pre = sys.argv[1], sys.argv[2]
N, n, Rr, Cc, NC = 64, 5, 4, 6, 5
def Lf(name,shape=None):
    a=np.fromfile(pre+name,dtype=np.float32).astype(np.float64); t=torch.from_numpy(a)
    return t.reshape(shape) if shape else t
def Lu(name): return torch.from_numpy(np.fromfile(pre+name,dtype=np.uint8))
def Li(name): return torch.from_numpy(np.fromfile(pre+name,dtype=np.int64))
def save(t): np.atleast_1d(np.asarray(t.detach().numpy(),dtype=np.float32)).tofile(pre+".out")
def savei(name,t): np.atleast_1d(t.detach().numpy().astype(np.int64)).tofile(pre+name)

if   op=="clamp":     save(Lf(".x").clamp(-0.3,0.5))
elif op=="threshold": save(F.threshold(Lf(".x"),0.0,-1.0))
elif op=="lerp":      save(torch.lerp(Lf(".x"),Lf(".end"),0.3))
elif op=="addcmul":   save(torch.addcmul(Lf(".x"),Lf(".t1"),Lf(".t2"),value=0.5))
elif op=="addcdiv":   save(torch.addcdiv(Lf(".x"),Lf(".t1"),Lf(".t2"),value=0.5))
elif op=="swhere":    save(torch.where(Lu(".c").bool(),Lf(".x"),Lf(".y")))
elif op=="maskedfill":save(Lf(".x").masked_fill(Lu(".m").bool(),9.0))
elif op=="maskedselect":
    r=torch.masked_select(Lf(".x"),Lu(".m").bool()); save(r); savei(".cnt",torch.tensor([r.numel()]))
elif op=="tril":      save(torch.tril(Lf(".A",(n,n)),0))
elif op=="triu":      save(torch.triu(Lf(".A",(n,n)),0))
elif op=="flip":      save(torch.flip(Lf(".x",(Rr,Cc)),[1]))
elif op=="roll":      save(torch.roll(Lf(".x",(Rr,Cc)),2,1))
elif op=="onehot":    save(F.one_hot(Li(".ids"),NC).double())
elif op=="sort":      save(torch.sort(Lf(".x",(Rr,Cc)),1,descending=False).values)
elif op=="topk":      save(torch.topk(Lf(".x",(Rr,Cc)),3,1,largest=True).values)
elif op=="argsort":   save(torch.argsort(Lf(".x",(Rr,Cc)),1,descending=False).double())
elif op=="kthvalue":  save(torch.kthvalue(Lf(".x",(Rr,Cc)),2,1).values)
else: sys.stderr.write("torch_grad10: no ref for %s\n"%op); sys.exit(2)
