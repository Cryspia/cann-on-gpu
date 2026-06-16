#!/usr/bin/env python3
# PyTorch references for padding forward. Symmetric pads → pad-order convention is irrelevant.
import sys, numpy as np, torch, torch.nn.functional as F
op, pre = sys.argv[1], sys.argv[2]
def Lf(name,shape): return torch.from_numpy(np.fromfile(pre+name,dtype=np.float32).astype(np.float64)).reshape(shape)
def save(t): np.asarray(t.detach().numpy(),dtype=np.float32).tofile(pre+".out")
if   op=="constantpad":   save(F.pad(Lf(".x",(1,2,4)),(1,1),"constant",0.5))
elif op=="reflect1d":     save(F.pad(Lf(".x",(1,2,5)),(2,2),"reflect"))
elif op=="reflect2d":     save(F.pad(Lf(".x",(1,2,5,5)),(2,2,2,2),"reflect"))
elif op=="reflect3d":     save(F.pad(Lf(".x",(1,1,4,4,4)),(1,1,1,1,1,1),"reflect"))
elif op=="replicate1d":   save(F.pad(Lf(".x",(1,2,5)),(2,2),"replicate"))
elif op=="replicate2d":   save(F.pad(Lf(".x",(1,2,5,5)),(2,2,2,2),"replicate"))
elif op=="replicate3d":   save(F.pad(Lf(".x",(1,1,4,4,4)),(1,1,1,1,1,1),"replicate"))
elif op=="circular2d":    save(F.pad(Lf(".x",(1,2,5,5)),(2,2,2,2),"circular"))
else: sys.stderr.write("torch_pad: no ref %s\n"%op); sys.exit(2)
