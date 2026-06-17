#!/usr/bin/env python3
# PyTorch refs for elementwise long-tail + comparison/bitwise. int/bool outputs saved as float.
from torch_common import *
op, pre = sys.argv[1], sys.argv[2]
if   op=="gcd":      savef(pre,".out",torch.gcd(loadi32(pre,".ai"),loadi32(pre,".bi")).double())
elif op=="logit":    savef(pre,".out",torch.logit(loadf(pre,".a"),eps=1e-6))
elif op=="nantonum": savef(pre,".out",torch.nan_to_num(loadf(pre,".a"),nan=0.0,posinf=1e4,neginf=-1e4))
elif op=="softsign": savef(pre,".out",F.softsign(loadf(pre,".a")))
elif op=="rounddecimals": savef(pre,".out",torch.round(loadf(pre,".a"),decimals=2))
elif op=="signbit":  savef(pre,".out",torch.signbit(loadf(pre,".a")).double())
elif op=="equal":    savef(pre,".out",torch.eq(loadf(pre,".a"),loadf(pre,".b")).double())
elif op=="isclose":  savef(pre,".out",torch.isclose(loadf(pre,".a"),loadf(pre,".b"),rtol=1e-3,atol=1e-3).double())
elif op=="logicalnot": savef(pre,".out",torch.logical_not(loadf(pre,".a")).double())
elif op=="logicalxor":
    au=loadu8(pre,".a").bool(); bu=loadu8(pre,".b").bool()
    savef(pre,".out",torch.logical_xor(au,bu).double())
elif op=="bucketize":  savef(pre,".out",torch.bucketize(loadf(pre,".a"),loadf(pre,".bnd"),right=False).double())
elif op=="searchsorted": savef(pre,".out",torch.searchsorted(loadf(pre,".srt"),loadf(pre,".a"),right=False).double())
else: no_ref("torch_elt", op)
