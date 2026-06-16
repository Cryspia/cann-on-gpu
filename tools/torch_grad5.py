#!/usr/bin/env python3
# PyTorch references for normalization FORWARD ops, matching each shim's verified convention:
#   layernorm   = F.layer_norm(x,[D],w,b,eps)
#   rmsnorm     = x*rsqrt(mean(x^2,-1)+eps)*gamma
#   groupnorm   = F.group_norm(x,G,w,b,eps)        (x[N,C,H,W], w/b[C])
#   batchnorm   = F.batch_norm(x,mean,var,w,b,training=False,eps)  (eval)
#   instancenorm= F.instance_norm(x,w=w,b=b,eps)
#   addrmsnorm  = rs=x+res; y=rmsnorm(rs,gamma)    -> .y and .rs
import sys, numpy as np, torch, torch.nn.functional as F
op, pre = sys.argv[1], sys.argv[2]
def Lf(name, n, shape=None):
    a=np.fromfile(pre+name,dtype=np.float32).astype(np.float64); t=torch.from_numpy(a)
    return t.reshape(shape) if shape else t
def save(name,t): t.detach().numpy().astype(np.float32).tofile(pre+name)
R,Dd = 8,16                 # layernorm/rmsnorm/addrmsnorm
N,C,Hh = 2,4,3              # 4D norms; groups=2
G,EPS = 2,1e-5
def rms(x,g): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+EPS)*g

if op=="layernorm":
    x=Lf(".x",R*Dd,(R,Dd)); w=Lf(".w",Dd,(Dd,)); b=Lf(".b",Dd,(Dd,))
    save(".out",F.layer_norm(x,(Dd,),w,b,EPS))
elif op=="rmsnorm":
    x=Lf(".x",R*Dd,(R,Dd)); g=Lf(".g",Dd,(Dd,)); save(".out",rms(x,g))
elif op=="addrmsnorm":
    x=Lf(".x",R*Dd,(R,Dd)); res=Lf(".res",R*Dd,(R,Dd)); g=Lf(".g",Dd,(Dd,))
    rs=x+res; save(".y",rms(rs,g)); save(".rs",rs)
elif op=="groupnorm":
    x=Lf(".x",N*C*Hh*Hh,(N,C,Hh,Hh)); w=Lf(".w",C,(C,)); b=Lf(".b",C,(C,))
    save(".out",F.group_norm(x,G,w,b,EPS))
elif op=="batchnorm":
    x=Lf(".x",N*C*Hh*Hh,(N,C,Hh,Hh)); w=Lf(".w",C,(C,)); b=Lf(".b",C,(C,)); m=Lf(".m",C,(C,)); v=Lf(".v",C,(C,))
    save(".out",F.batch_norm(x,m,v,w,b,training=False,eps=EPS))
elif op=="instancenorm":
    x=Lf(".x",N*C*Hh*Hh,(N,C,Hh,Hh)); w=Lf(".w",C,(C,)); b=Lf(".b",C,(C,))
    save(".out",F.instance_norm(x,weight=w,bias=b,eps=EPS))
else:
    sys.stderr.write("torch_grad5: no ref for %s\n"%op); sys.exit(2)
