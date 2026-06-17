#!/usr/bin/env python3
# PyTorch references for the pooling family (forward). pad=0 throughout to avoid count_include_pad ambiguity.
from torch_common import *
op, pre = sys.argv[1], sys.argv[2]
def Lf(name,shape): return loadf(pre, name, shape)
def save(t): savef(pre, ".out", t)
if   op=="avgpool3d":        save(F.avg_pool3d(Lf(".x",(1,2,4,4,4)),2,2,0))
elif op=="avgpool1d":        save(F.avg_pool1d(Lf(".x",(1,2,8)),2,2,0))
elif op=="adaptiveavgpool3d":save(F.adaptive_avg_pool3d(Lf(".x",(1,2,4,4,4)),(2,2,2)))
elif op=="adaptivemaxpool2d":save(F.adaptive_max_pool2d(Lf(".x",(1,2,4,4)),(2,2))[0])
elif op=="maxpool3d":        save(F.max_pool3d(Lf(".x",(1,2,4,4,4)),2,2,0))
elif op=="maxpool3dargmax":  save(F.max_pool3d(Lf(".x",(1,2,4,4,4)),2,2,0))
elif op=="channelshuffle":   save(F.channel_shuffle(Lf(".x",(1,4,2,2)),2))
elif op=="im2col":           save(F.unfold(Lf(".x",(1,2,4,4)),2,1,0,2))
elif op=="col2im":           save(F.fold(Lf(".x",(1,8,4)),(4,4),2,1,0,2))
elif op=="lppool2d":         save(F.lp_pool2d(Lf(".x",(1,2,4,4)),2,2,2))
else: no_ref("torch_pool", op)
