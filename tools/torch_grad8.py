#!/usr/bin/env python3
# PyTorch references for loss FORWARD across reduction modes (0=none,1=mean,2=sum).
#   usage: torch_grad8.py <op> <prefix> <reduction>
import sys, numpy as np, torch, torch.nn.functional as F
op, pre, red = sys.argv[1], sys.argv[2], int(sys.argv[3])
RED = {0:"none",1:"mean",2:"sum"}[red]
N,C,D,M = 64,5,8,0.5
def Lf(name,shape=None):
    a=np.fromfile(pre+name,dtype=np.float32).astype(np.float64); t=torch.from_numpy(a)
    return t.reshape(shape) if shape else t
def Li(name): return torch.from_numpy(np.fromfile(pre+name,dtype=np.int64))
def save(t): np.atleast_1d(t.detach().numpy().astype(np.float32)).tofile(pre+".out")

if op in ("l1","smoothl1","mse","bce","kldiv","softmargin"):
    self,tgt = Lf(".self"), Lf(".tgt")
    if   op=="l1":         r=F.l1_loss(self,tgt,reduction=RED)
    elif op=="smoothl1":   r=F.smooth_l1_loss(self,tgt,reduction=RED,beta=1.0)
    elif op=="mse":        r=F.mse_loss(self,tgt,reduction=RED)
    elif op=="bce":        r=F.binary_cross_entropy(self,tgt,reduction=RED)
    elif op=="kldiv":      r=F.kl_div(self,tgt,reduction=RED,log_target=False)
    else:                  r=F.soft_margin_loss(self,tgt,reduction=RED)
elif op=="nllloss":
    r=F.nll_loss(Lf(".lp",(N,C)), Li(".tgt"), reduction=RED)
elif op=="cosine":
    r=F.cosine_embedding_loss(Lf(".x1",(N,D)),Lf(".x2",(N,D)),Lf(".tgt"),margin=M,reduction=RED)
elif op=="marginranking":
    r=F.margin_ranking_loss(Lf(".x1"),Lf(".x2"),Lf(".y"),margin=M,reduction=RED)
else: sys.stderr.write("torch_grad8: no ref for %s\n"%op); sys.exit(2)
save(r)
