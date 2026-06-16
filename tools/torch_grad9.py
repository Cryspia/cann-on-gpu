#!/usr/bin/env python3
# PyTorch SDPA reference for FlashAttentionScore (BNSD layout, GQA, tail-aligned causal, bool mask where
# nonzero=masked-out, scale=1/sqrt(D)). Manual softmax(scale*Q@Kᵀ + mask)@V on float64.
import sys, numpy as np, torch
case, pre = sys.argv[1], sys.argv[2]
CFG = {  # B,Nq,Nkv,Sq,Skv,D,causal,hasmask
  "plain":(2,2,2,4,4,8,0,0), "causal":(2,2,2,4,4,8,1,0), "gqa":(2,4,2,4,4,8,0,0),
  "mask":(2,2,2,4,4,8,0,1), "perf":(1,2,2,16,16,16,0,0),
}
B,Nq,Nkv,Sq,Skv,D,causal,hasmask = CFG[case]
def Lf(name,shape): return torch.from_numpy(np.fromfile(pre+name,dtype=np.float32).astype(np.float64)).reshape(shape)
q=Lf(".q",(B,Nq,Sq,D)); k=Lf(".k",(B,Nkv,Skv,D)); v=Lf(".v",(B,Nkv,Skv,D))
rep=Nq//Nkv
ke=k.repeat_interleave(rep,1); ve=v.repeat_interleave(rep,1)
scale=1.0/np.sqrt(D)
scores=torch.matmul(q,ke.transpose(-1,-2))*scale          # [B,Nq,Sq,Skv]
off=Skv-Sq
if causal:
    ii=torch.arange(Sq).view(Sq,1); jj=torch.arange(Skv).view(1,Skv)
    scores=scores.masked_fill(jj>ii+off, float("-inf"))
if hasmask:
    m=torch.from_numpy(np.fromfile(pre+".mask",dtype=np.uint8)).reshape(Sq,Skv).bool()
    scores=scores.masked_fill(m, float("-inf"))
p=torch.softmax(scores,-1)
out=torch.matmul(p,ve)                                     # [B,Nq,Sq,D]
out.numpy().astype(np.float32).tofile(pre+".out")
