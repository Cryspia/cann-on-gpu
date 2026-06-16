#!/usr/bin/env python3
# PyTorch-autograd references for conv / pool / index backward ops (independent of our analytic grads).
#   conv:         y=conv2d(x,w,b,stride=1,pad=1); y.backward(go) -> .gi (dX), .gw (dW), .gb (dB)
#   avgpool:      y=avg_pool2d(x,2);             y.backward(go) -> .gi
#   maxpool:      y=max_pool2d(x,2);             y.backward(go) -> .gi
#   adaptiveavg:  y=adaptive_avg_pool2d(x,(4,4));y.backward(go) -> .gi
#   embedding:    out=embedding(ids,w);          out.backward(grad) -> .gw (dWeight)
#   gather:       y=gather(x,dim,idx);           y.backward(go) -> .gi
import sys, numpy as np, torch, torch.nn.functional as F
op, pre = sys.argv[1], sys.argv[2]
def Lf(name, n, shape=None):
    a = np.fromfile(pre+name, dtype=np.float32).astype(np.float64)
    t = torch.from_numpy(a); return t.reshape(shape) if shape else t
def Li(name, shape):
    a = np.fromfile(pre+name, dtype=np.int64); return torch.from_numpy(a).reshape(shape)
def save(name, t): t.detach().numpy().astype(np.float32).tofile(pre+name)

# shapes shared with grad_check3.cpp
N,Cin,H,Cout,K = 2,3,6,4,3                     # conv
PN,PC,PH = 2,3,8                               # pool (in 8x8 -> 4x4)
V,D,Ln = 10,5,12                               # embedding
GA,GB,GL = 6,4,5                               # gather=index_select: x[GA,GB], idx[GL] along dim0 -> out[GL,GB]

if op=="conv":
    x=Lf(".x",N*Cin*H*H,(N,Cin,H,H)).requires_grad_(True)
    w=Lf(".w",Cout*Cin*K*K,(Cout,Cin,K,K)).requires_grad_(True)
    b=Lf(".b",Cout,(Cout,)).requires_grad_(True)
    go=Lf(".go",N*Cout*H*H,(N,Cout,H,H))
    y=F.conv2d(x,w,b,stride=1,padding=1); y.backward(go)
    save(".gi",x.grad); save(".gw",w.grad); save(".gb",b.grad)
elif op in ("avgpool","maxpool","adaptiveavg"):
    x=Lf(".x",PN*PC*PH*PH,(PN,PC,PH,PH)).requires_grad_(True)
    go=Lf(".go",PN*PC*4*4,(PN,PC,4,4))
    if   op=="avgpool":     y=F.avg_pool2d(x,2)
    elif op=="maxpool":     y=F.max_pool2d(x,2)
    else:                   y=F.adaptive_avg_pool2d(x,(4,4))
    y.backward(go); save(".gi",x.grad)
elif op=="embedding":
    w=Lf(".w",V*D,(V,D)).requires_grad_(True); ids=Li(".ids",(Ln,)); grad=Lf(".grad",Ln*D,(Ln,D))
    out=F.embedding(ids,w); out.backward(grad); save(".gw",w.grad)
elif op=="gather":   # shim aclnnGather = index_select (1-D index along dim0), not torch.gather
    x=Lf(".x",GA*GB,(GA,GB)).requires_grad_(True); idx=Li(".idx",(GL,)); go=Lf(".go",GL*GB,(GL,GB))
    y=x.index_select(0,idx); y.backward(go); save(".gi",x.grad)
else:
    sys.stderr.write("torch_grad3: no ref for %s\n"%op); sys.exit(2)
