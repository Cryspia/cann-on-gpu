#!/usr/bin/env python3
# PyTorch refs for linalg extras / distance / complex. Non-unique factorizations checked by reconstruction
# inside linalg2_check.cpp; here we emit direct refs (+ eigvalsh for eigh).
from torch_common import *
op, pre = sys.argv[1], sys.argv[2]
n=4
def Lf(name,shape): return loadf(pre, name, shape)
def save(t,nm=".out"): savef(pre, nm, t)
if   op=="eigh":     save(torch.linalg.eigvalsh(Lf(".A",(n,n))),".W")            # sorted ascending; vectors checked by recon
elif op in ("lulusolve","lstsq"): pass                                           # reconstruction-only (A@X==B)
elif op=="matrixexp":save(torch.linalg.matrix_exp(Lf(".A",(n,n))))
elif op=="pinverse": save(torch.linalg.pinv(Lf(".A",(n,n))))
elif op=="logdet":   save(torch.logdet(Lf(".A",(n,n))).reshape(1))
elif op=="cdist":    save(torch.cdist(Lf(".a",(3,4)),Lf(".b",(5,4)),p=2.0))
elif op=="pdist":    save(torch.nn.functional.pdist(Lf(".a",(4,3)),p=2.0))
elif op=="complex":  save(torch.view_as_real(torch.complex(Lf(".a",(5,)),Lf(".b",(5,)))))
elif op=="polar":    save(torch.view_as_real(torch.polar(Lf(".a",(5,)).abs(),Lf(".b",(5,)))))
elif op=="real":     save(torch.view_as_complex(Lf(".a",(5,2)).contiguous()).real)
else: no_ref("torch_linalg2", op)
