#!/usr/bin/env python3
# PyTorch references for vision/conv/interpolation forward. align_corners=False for linear-family upsample.
import sys, numpy as np, torch, torch.nn.functional as F
op, pre = sys.argv[1], sys.argv[2]
def Lf(name,shape): return torch.from_numpy(np.fromfile(pre+name,dtype=np.float32).astype(np.float64)).reshape(shape)
def save(t): np.asarray(t.detach().numpy(),dtype=np.float32).tofile(pre+".out")
if   op=="convolution":   save(F.conv2d(Lf(".x",(1,3,6,6)),Lf(".w",(4,3,3,3)),Lf(".b",(4,)),1,1,1,1))
elif op=="gridsampler2d": save(F.grid_sample(Lf(".x",(1,2,4,4)),Lf(".g",(1,3,3,2)),mode="bilinear",padding_mode="zeros",align_corners=False))
elif op=="pixelshuffle":  save(F.pixel_shuffle(Lf(".x",(1,8,2,2)),2))
elif op=="pixelunshuffle":save(F.pixel_unshuffle(Lf(".x",(1,2,4,4)),2))
elif op=="upnearest3d":   save(F.interpolate(Lf(".x",(1,2,2,2,2)),size=(4,4,4),mode="nearest"))
elif op=="upbilinear2d":  save(F.interpolate(Lf(".x",(1,2,2,2)),size=(4,4),mode="bilinear",align_corners=False))
elif op=="upbicubic2d":   save(F.interpolate(Lf(".x",(1,2,3,3)),size=(6,6),mode="bicubic",align_corners=False))
elif op=="uptrilinear3d": save(F.interpolate(Lf(".x",(1,2,2,2,2)),size=(4,4,4),mode="trilinear",align_corners=False))
elif op=="upnearest1d":   save(F.interpolate(Lf(".x",(1,2,4)),size=(8,),mode="nearest"))
elif op=="uplinear1d":    save(F.interpolate(Lf(".x",(1,2,4)),size=(8,),mode="linear",align_corners=False))
elif op=="affinegrid":    save(F.affine_grid(Lf(".x",(1,2,3)),(1,1,3,3),align_corners=False))
elif op=="lrn":           save(F.local_response_norm(Lf(".x",(1,4,2,2)),3,alpha=1e-4,beta=0.75,k=1.0))
else: sys.stderr.write("torch_vision: no ref %s\n"%op); sys.exit(2)
