#!/usr/bin/env python3
# PyTorch references for normalization FORWARD ops, matching each shim's verified convention:
#   layernorm   = F.layer_norm(x,[D],w,b,eps)
#   rmsnorm     = x*rsqrt(mean(x^2,-1)+eps)*gamma
#   groupnorm   = F.group_norm(x,G,w,b,eps)        (x[N,C,H,W], w/b[C])
#   batchnorm   = F.batch_norm(x,mean,var,w,b,training=False,eps)  (eval)
#   instancenorm= F.instance_norm(x,w=w,b=b,eps)
#   addrmsnorm  = rs=x+res; y=rmsnorm(rs,gamma)    -> .y and .rs
from torch_common import *
op, pre = sys.argv[1], sys.argv[2]
R,Dd = 8,16                 # layernorm/rmsnorm/addrmsnorm
N,C,Hh = 2,4,3              # 4D norms; groups=2
G,EPS = 2,1e-5
def rms(x,g): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+EPS)*g

if op=="layernorm":
    x=loadf(pre,".x",(R,Dd)); w=loadf(pre,".w",(Dd,)); b=loadf(pre,".b",(Dd,))
    savef(pre,".out",F.layer_norm(x,(Dd,),w,b,EPS))
elif op=="rmsnorm":
    x=loadf(pre,".x",(R,Dd)); g=loadf(pre,".g",(Dd,)); savef(pre,".out",rms(x,g))
elif op=="addrmsnorm":
    x=loadf(pre,".x",(R,Dd)); res=loadf(pre,".res",(R,Dd)); g=loadf(pre,".g",(Dd,))
    rs=x+res; savef(pre,".y",rms(rs,g)); savef(pre,".rs",rs)
elif op=="groupnorm":
    x=loadf(pre,".x",(N,C,Hh,Hh)); w=loadf(pre,".w",(C,)); b=loadf(pre,".b",(C,))
    savef(pre,".out",F.group_norm(x,G,w,b,EPS))
elif op=="batchnorm":
    x=loadf(pre,".x",(N,C,Hh,Hh)); w=loadf(pre,".w",(C,)); b=loadf(pre,".b",(C,)); m=loadf(pre,".m",(C,)); v=loadf(pre,".v",(C,))
    savef(pre,".out",F.batch_norm(x,m,v,w,b,training=False,eps=EPS))
elif op=="instancenorm":
    x=loadf(pre,".x",(N,C,Hh,Hh)); w=loadf(pre,".w",(C,)); b=loadf(pre,".b",(C,))
    savef(pre,".out",F.instance_norm(x,weight=w,bias=b,eps=EPS))
else:
    no_ref("torch_norm", op)
