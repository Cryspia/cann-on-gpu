#!/usr/bin/env python3
# PyTorch references for loss FORWARD across reduction modes (0=none,1=mean,2=sum).
#   usage: torch_loss.py <op> <prefix> <reduction>
from torch_common import *
op, pre, red = sys.argv[1], sys.argv[2], int(sys.argv[3])
RED = {0:"none",1:"mean",2:"sum"}[red]
N,C,D,M = 64,5,8,0.5

if op in ("l1","smoothl1","mse","bce","kldiv","softmargin"):
    self,tgt = loadf(pre,".self"), loadf(pre,".tgt")
    if   op=="l1":         r=F.l1_loss(self,tgt,reduction=RED)
    elif op=="smoothl1":   r=F.smooth_l1_loss(self,tgt,reduction=RED,beta=1.0)
    elif op=="mse":        r=F.mse_loss(self,tgt,reduction=RED)
    elif op=="bce":        r=F.binary_cross_entropy(self,tgt,reduction=RED)
    elif op=="kldiv":      r=F.kl_div(self,tgt,reduction=RED,log_target=False)
    else:                  r=F.soft_margin_loss(self,tgt,reduction=RED)
elif op=="nllloss":
    r=F.nll_loss(loadf(pre,".lp",(N,C)), loadi(pre,".tgt"), reduction=RED)
elif op=="cosine":
    r=F.cosine_embedding_loss(loadf(pre,".x1",(N,D)),loadf(pre,".x2",(N,D)),loadf(pre,".tgt"),margin=M,reduction=RED)
elif op=="marginranking":
    r=F.margin_ranking_loss(loadf(pre,".x1"),loadf(pre,".x2"),loadf(pre,".y"),margin=M,reduction=RED)
else: no_ref("torch_loss", op)
savef(pre, ".out", r)
