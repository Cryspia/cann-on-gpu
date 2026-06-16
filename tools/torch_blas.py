#!/usr/bin/env python3
# PyTorch references for the BLAS/matmul family. beta=0.5, alpha=2.0 for add* ops.
import sys, numpy as np, torch
op, pre = sys.argv[1], sys.argv[2]
M,K,N,Bb = 4,5,6,3
def Lf(name,shape): return torch.from_numpy(np.fromfile(pre+name,dtype=np.float32).astype(np.float64)).reshape(shape)
def save(t): np.atleast_1d(np.asarray(t.detach().numpy(),dtype=np.float32)).tofile(pre+".out")
b_,a_ = 0.5, 2.0
if   op=="addmm":   save(b_*Lf(".c",(M,N)) + a_*(Lf(".a",(M,K))@Lf(".b",(K,N))))
elif op=="addbmm":  save(b_*Lf(".c",(M,N)) + a_*torch.bmm(Lf(".a",(Bb,M,K)),Lf(".b",(Bb,K,N))).sum(0))
elif op=="addmv":   save(b_*Lf(".c",(M,)) + a_*(Lf(".a",(M,K))@Lf(".b",(K,))))
elif op=="addr":    save(b_*Lf(".c",(M,N)) + a_*torch.outer(Lf(".a",(M,)),Lf(".b",(N,))))
elif op=="baddbmm": save(b_*Lf(".c",(Bb,M,N)) + a_*torch.bmm(Lf(".a",(Bb,M,K)),Lf(".b",(Bb,K,N))))
elif op in ("bmm","batchmatmul"): save(torch.bmm(Lf(".a",(Bb,M,K)),Lf(".b",(Bb,K,N))))
elif op=="ger":     save(torch.outer(Lf(".a",(M,)),Lf(".b",(N,))))
elif op=="inner":   save(torch.inner(Lf(".a",(8,)),Lf(".b",(8,))).reshape(1))   # shim Inner = 1-D dot
elif op=="kron":    save(torch.kron(Lf(".a",(2,3)),Lf(".b",(2,2))))
elif op=="mm":      save(Lf(".a",(M,K))@Lf(".b",(K,N)))
elif op=="mv":      save(Lf(".a",(M,K))@Lf(".b",(K,)))
elif op=="tensordot": save(torch.tensordot(Lf(".a",(3,4)),Lf(".b",(4,5)),dims=1))
elif op=="vdot":    save(torch.dot(Lf(".a",(8,)),Lf(".b",(8,))).reshape(1))
else: sys.stderr.write("torch_blas: no ref %s\n"%op); sys.exit(2)
