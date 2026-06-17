#!/usr/bin/env python3
# PyTorch-autograd references for conv / pool / index backward ops (independent of our analytic grads).
#   conv:         y=conv2d(x,w,b,stride=1,pad=1); y.backward(go) -> .gi (dX), .gw (dW), .gb (dB)
#   avgpool:      y=avg_pool2d(x,2);             y.backward(go) -> .gi
#   maxpool:      y=max_pool2d(x,2);             y.backward(go) -> .gi
#   adaptiveavg:  y=adaptive_avg_pool2d(x,(4,4));y.backward(go) -> .gi
#   embedding:    out=embedding(ids,w);          out.backward(grad) -> .gw (dWeight)
#   gather:       y=gather(x,dim,idx);           y.backward(go) -> .gi
from torch_common import *
op, pre = sys.argv[1], sys.argv[2]

# shapes shared with nn_grad_check.cpp
N,Cin,H,Cout,K = 2,3,6,4,3                     # conv
PN,PC,PH = 2,3,8                               # pool (in 8x8 -> 4x4)
V,D,Ln = 10,5,12                               # embedding
GA,GB,GL = 6,4,5                               # gather=index_select: x[GA,GB], idx[GL] along dim0 -> out[GL,GB]

if op=="conv":
    x=loadf(pre,".x",(N,Cin,H,H)).requires_grad_(True)
    w=loadf(pre,".w",(Cout,Cin,K,K)).requires_grad_(True)
    b=loadf(pre,".b",(Cout,)).requires_grad_(True)
    go=loadf(pre,".go",(N,Cout,H,H))
    y=F.conv2d(x,w,b,stride=1,padding=1); y.backward(go)
    savef(pre,".gi",x.grad); savef(pre,".gw",w.grad); savef(pre,".gb",b.grad)
elif op in ("avgpool","maxpool","adaptiveavg"):
    x=loadf(pre,".x",(PN,PC,PH,PH)).requires_grad_(True)
    go=loadf(pre,".go",(PN,PC,4,4))
    if   op=="avgpool":     y=F.avg_pool2d(x,2)
    elif op=="maxpool":     y=F.max_pool2d(x,2)
    else:                   y=F.adaptive_avg_pool2d(x,(4,4))
    y.backward(go); savef(pre,".gi",x.grad)
elif op=="embedding":
    w=loadf(pre,".w",(V,D)).requires_grad_(True); ids=loadi(pre,".ids",(Ln,)); grad=loadf(pre,".grad",(Ln,D))
    out=F.embedding(ids,w); out.backward(grad); savef(pre,".gw",w.grad)
elif op=="gather":   # shim aclnnGather = index_select (1-D index along dim0), not torch.gather
    x=loadf(pre,".x",(GA,GB)).requires_grad_(True); idx=loadi(pre,".idx",(GL,)); go=loadf(pre,".go",(GL,GB))
    y=x.index_select(0,idx); y.backward(go); savef(pre,".gi",x.grad)
else:
    no_ref("torch_nn_grad", op)
