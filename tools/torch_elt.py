#!/usr/bin/env python3
# PyTorch refs for elementwise long-tail + comparison/bitwise. int/bool outputs saved as float.
import sys, numpy as np, torch, torch.nn.functional as F
op, pre = sys.argv[1], sys.argv[2]
def Lf(name,n): return torch.from_numpy(np.fromfile(pre+name,dtype=np.float32).astype(np.float64))
def Li(name): return torch.from_numpy(np.fromfile(pre+name,dtype=np.int32).astype(np.int64))
def save(t): np.atleast_1d(np.asarray(t.detach().numpy(),dtype=np.float64).astype(np.float32).reshape(-1)).tofile(pre+".out")
if   op=="gcd":      save(torch.gcd(Li(".ai"),Li(".bi")).double())
elif op=="logit":    save(torch.logit(Lf(".a",32),eps=1e-6))
elif op=="nantonum": save(torch.nan_to_num(Lf(".a",32),nan=0.0,posinf=1e4,neginf=-1e4))
elif op=="softsign": save(F.softsign(Lf(".a",32)))
elif op=="rounddecimals": save(torch.round(Lf(".a",32),decimals=2))
elif op=="signbit":  save(torch.signbit(Lf(".a",32)).double())
elif op=="equal":    save(torch.eq(Lf(".a",32),Lf(".b",32)).double())
elif op=="isclose":  save(torch.isclose(Lf(".a",32),Lf(".b",32),rtol=1e-3,atol=1e-3).double())
elif op=="logicalnot": save(torch.logical_not(Lf(".a",32)).double())
elif op=="logicalxor":
    au=torch.from_numpy(np.fromfile(pre+".a",dtype=np.uint8)).bool(); bu=torch.from_numpy(np.fromfile(pre+".b",dtype=np.uint8)).bool()
    save(torch.logical_xor(au,bu).double())
elif op=="bucketize":  save(torch.bucketize(Lf(".a",8),Lf(".bnd",5),right=False).double())
elif op=="searchsorted": save(torch.searchsorted(Lf(".srt",5),Lf(".a",8),right=False).double())
else: sys.stderr.write("torch_elt: no ref %s\n"%op); sys.exit(2)
