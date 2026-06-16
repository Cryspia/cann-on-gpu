#!/usr/bin/env python3
# PyTorch refs for indexing/shape extras. int outputs saved as float.
import sys, numpy as np, torch
op, pre = sys.argv[1], sys.argv[2]
def Lf(name,shape): return torch.from_numpy(np.fromfile(pre+name,dtype=np.float32).astype(np.float64)).reshape(shape)
def Li(name,shape): return torch.from_numpy(np.fromfile(pre+name,dtype=np.int64)).reshape(shape)
def Lii(name,shape): return torch.from_numpy(np.fromfile(pre+name,dtype=np.int32).astype(np.int64)).reshape(shape)
def save(t): np.atleast_1d(np.asarray(t.detach().numpy(),dtype=np.float64).astype(np.float32).reshape(-1)).tofile(pre+".out")
if   op=="take":        save(torch.take(Lf(".a",(3,4)),Li(".idx",(5,))))
elif op=="takealongdim":save(torch.take_along_dim(Lf(".a",(3,4)),Li(".idx",(3,4)),dim=1))
elif op=="indexselect": save(torch.index_select(Lf(".a",(4,5)),0,Li(".idx",(3,))))
elif op=="bincount":    save(torch.bincount(Lii(".ai",(8,)),minlength=5).double())
elif op=="histc":       save(torch.histc(Lf(".a",(16,)),bins=4,min=-1.0,max=1.0))
elif op=="narrow":      save(torch.narrow(Lf(".a",(4,6)),1,1,3))
elif op=="rot90":       save(torch.rot90(Lf(".a",(3,4)),1,[0,1]))
elif op=="flatten":     save(torch.flatten(Lf(".a",(3,4))))
elif op=="diagonal":    save(torch.diagonal(Lf(".a",(4,4)),0))
elif op=="tile":        save(torch.tile(Lf(".a",(2,3)),(2,2)))
elif op=="repeat":      save(Lf(".a",(2,3)).repeat(2,2))
elif op=="repeatinterleave": save(torch.repeat_interleave(Lf(".a",(3,4)),2,dim=1))
else: sys.stderr.write("torch_idx: no ref %s\n"%op); sys.exit(2)
