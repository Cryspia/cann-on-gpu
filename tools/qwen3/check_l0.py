#!/usr/bin/env python3
# Independently recompute Qwen3 layer 0 in numpy fp64, compare each op against the shim dump to locate where a deviation first appears.
# Run the harness with QWEN_DUMP_L0=1 first to produce the l0_*.bin dumps, then run this script.
import numpy as np, os
D=os.path.dirname(__file__); DAT=os.path.join(D,"data")
def r(n,c): return np.fromfile(os.path.join(DAT,n),dtype=np.float32).astype(np.float64).reshape(c)
cfg=open(os.path.join(DAT,"config.txt")).read().split()
Dm,L,Hq,Hkv,hd,F,V=[int(x) for x in cfg[:7]]; eps=float(cfg[7]); S=int(cfg[9])
Dq,Dkv=Hq*hd,Hkv*hd
hs=r("hidden.bin",(L+1,S,Dm))
x=hs[0]                                  # embedding output
def rms(x,g):
    v=(x*x).mean(-1,keepdims=True); return x/np.sqrt(v+eps)*g
g1=r("l0.g1",(Dm,))
h=rms(x,g1)
# Qwen3 attention: no bias on the projections
qw=r("l0.q.w",(Dq,Dm)); kw=r("l0.k.w",(Dkv,Dm)); vw=r("l0.v.w",(Dkv,Dm))
q=h@qw.T; k=h@kw.T; v=h@vw.T              # [S,Dq], [S,Dkv], [S,Dkv]
# Qwen3 per-head QK-RMSNorm (over head_dim), applied before RoPE
qn_w=r("l0.qn",(hd,)); kn_w=r("l0.kn",(hd,))
qn=rms(q.reshape(S,Hq,hd),qn_w).reshape(S,Dq)
kn=rms(k.reshape(S,Hkv,hd),kn_w).reshape(S,Dkv)
# RoPE half-split, using the real HF-exported inv_freq
half=hd//2
inv=np.fromfile(os.path.join(DAT,"inv_freq.bin"),dtype=np.float32).astype(np.float64)   # [half]
pos=np.arange(S)[:,None]
ang=pos*inv[None,:]                        # [S,half]
cos=np.concatenate([np.cos(ang),np.cos(ang)],-1)  # [S,hd]
sin=np.concatenate([np.sin(ang),np.sin(ang)],-1)
def rope(t,H):
    t=t.reshape(S,H,hd)
    x1=t[...,:half]; x2=t[...,half:]
    rot=np.concatenate([-x2,x1],-1)
    return (t*cos[:,None,:]+rot*sin[:,None,:]).reshape(S,H*hd)
qr=rope(qn,Hq); kr=rope(kn,Hkv)
# GQA causal attention
scale=1/np.sqrt(hd)
qh=qr.reshape(S,Hq,hd).transpose(1,0,2)    # [Hq,S,hd]
kh=kr.reshape(S,Hkv,hd).transpose(1,0,2)
vh=v.reshape(S,Hkv,hd).transpose(1,0,2)
grp=Hq//Hkv
out=np.zeros((Hq,S,hd))
mask=np.triu(np.ones((S,S))*-1e30,1)
for hq in range(Hq):
    kk=kh[hq//grp]; vv=vh[hq//grp]
    sc=qh[hq]@kk.T*scale+mask
    sc=sc-sc.max(-1,keepdims=True); e=np.exp(sc); p=e/e.sum(-1,keepdims=True)
    out[hq]=p@vv
attn=out.transpose(1,0,2).reshape(S,Dq)
ow=r("l0.o.w",(Dm,Dq)); op=attn@ow.T
x2=x+op
g2=r("l0.g2",(Dm,)); h2=rms(x2,g2)
gw=r("l0.gate.w",(F,Dm)); uw=r("l0.up.w",(F,Dm)); dw=r("l0.down.w",(Dm,F))
gate=h2@gw.T; up=h2@uw.T
sg=gate/(1+np.exp(-gate)); mlp=sg*up
down=mlp@dw.T
xn=x2+down

# Compare against HF hs[1] (verifies the numpy reference is correct)
e_hf=np.abs(xn-hs[1]).max()/(np.abs(hs[1]).max()+1e-9)
print(f"[numpy-ref vs HF hs[1]] norm_err={e_hf:.3e}  (should be ~0 to confirm reference correctness)")

# Compare against shim dump per op. Note: the harness dumps qr/kr in [H,s,hd] (BNSD) order (post-permute, post-RoPE),
# so compare against qh/kh (the same BNSD-ordered tensors), not the [s,H*hd] layout.
refs={"h":h,"q":q,"k":k,"v":v,"qn":qn,"kn":kn,"qr":qh,"kr":kh,"attn":attn,"op":op,"x2":x2,"h2":h2,"mlp":mlp,"down":down,"xn":xn}
print("\n[shim vs numpy-ref] per-op normalized error:")
for name,ref in refs.items():
    p=os.path.join(DAT,f"l0_{name}.bin")
    if not os.path.exists(p): print(f"  {name:5s} (no dump)"); continue
    sh=np.fromfile(p,dtype=np.float32).astype(np.float64).reshape(ref.shape)
    e=np.abs(sh-ref).max()/(np.abs(ref).max()+1e-9)
    flag=" <<<<" if e>1e-3 else ""
    print(f"  {name:5s} err={e:.3e}{flag}")
