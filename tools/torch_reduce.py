#!/usr/bin/env python3
# PyTorch references for reduction / softmax FORWARD ops (reduce along dim=1 of x[Rr,Cc]).
# Conventions verified from the shim: Var/Std unbiased (correction=1); Amax/Amin plain max/min;
# Norm = (sum |x|^p)^(1/p). argmax/argmin/median outputs compared as float.
from torch_common import *
op, pre = sys.argv[1], sys.argv[2]
Rr,Cc,P = 6,5,2.0
x = loadf(pre, ".x", (Rr,Cc))
ref = {
  "softmax":    lambda: torch.softmax(x,1),
  "logsoftmax": lambda: torch.log_softmax(x,1),
  "cumsum":     lambda: torch.cumsum(x,1),
  "cumprod":    lambda: torch.cumprod(x,1),
  "logsumexp":  lambda: torch.logsumexp(x,1),
  "var":        lambda: torch.var(x,1,correction=1),
  "std":        lambda: torch.std(x,1,correction=1),
  "normp":      lambda: torch.norm(x,P,dim=1),
  "amax":       lambda: torch.amax(x,1),
  "amin":       lambda: torch.amin(x,1),
  "argmax":     lambda: torch.argmax(x,1).double(),
  "argmin":     lambda: torch.argmin(x,1).double(),
  "median":     lambda: torch.median(x,1).values,
}
if op not in ref: no_ref("torch_reduce", op)
savef(pre, ".out", ref[op]())
