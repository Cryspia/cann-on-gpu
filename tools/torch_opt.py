#!/usr/bin/env python3
# PyTorch refs for optimizers (one update step from fresh state: m=v=buf=0, step=1). torch.optim for the
# builtins (exact, independent); closed-form-from-spec for Lamb/Lars/FusedEmaAdam (no torch builtin).
from torch_common import *
op, pre = sys.argv[1], sys.argv[2]
N=64; lr,b1,b2,eps,wd = 0.01,0.9,0.999,1e-8,0.01
mom,alpha,rho,trust,emad = 0.9,0.99,0.9,0.001,0.999
P0=loadf(pre,".p")
G =loadf(pre,".g")
def save(t): savef(pre, ".out", t)
def via(make):
    p=torch.nn.Parameter(P0.clone()); o=make(p); p.grad=G.clone(); o.step(); return p.data
if   op=="adam":     save(via(lambda p: torch.optim.Adam([p],lr=lr,betas=(b1,b2),eps=eps,weight_decay=wd)))
elif op=="adamw":    save(via(lambda p: torch.optim.AdamW([p],lr=lr,betas=(b1,b2),eps=eps,weight_decay=wd)))
elif op=="adagrad":  save(via(lambda p: torch.optim.Adagrad([p],lr=lr,eps=eps,weight_decay=wd)))
elif op=="rmsprop":  save(via(lambda p: torch.optim.RMSprop([p],lr=lr,alpha=alpha,eps=eps,weight_decay=wd)))
elif op=="adamax":   save(via(lambda p: torch.optim.Adamax([p],lr=lr,betas=(b1,b2),eps=eps,weight_decay=wd)))
elif op=="adadelta": save(via(lambda p: torch.optim.Adadelta([p],lr=lr,rho=rho,eps=eps,weight_decay=wd)))
elif op=="momentum": save(via(lambda p: torch.optim.SGD([p],lr=lr,momentum=mom,weight_decay=wd,dampening=0.0,nesterov=False)))
elif op=="lamb":     # m=v=0,step=1 -> mhat=g, vhat=g^2 ; update=g/(|g|+eps)+wd*P ; trust=||P||/||update||
    g=G; mhat=g; vhat=g*g; upd=mhat/(torch.sqrt(vhat)+eps)+wd*P0
    tr=(P0.norm()/upd.norm()) if (P0.norm()>0 and upd.norm()>0) else torch.tensor(1.0); save(P0 - lr*tr*upd)
elif op=="lars":     # buf=0: v=local_lr*(g+wd*P); local_lr=trust*||P||/(||g||+wd*||P||+eps)
    g=G; gn=g.norm(); pn=P0.norm(); llr=trust*pn/(gn+wd*pn+eps); v=llr*(g+wd*P0); save(P0 - lr*v)
elif op=="fusedemaadam":  # AdamW base (m=v=0,step=1): param update then ema tracks param (compare param)
    p=(1-lr*wd)*P0; mhat=G; vhat=G*G; save(p - lr*mhat/(torch.sqrt(vhat)+eps))
else: no_ref("torch_opt", op)
