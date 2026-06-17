#!/usr/bin/env python3
# PyTorch-autograd references for activation/softmax backward. y=f(x); y.backward(go) -> x.grad (.gi).
from torch_common import *
op, pre = sys.argv[1], sys.argv[2]
N, R, C = 4096, 8, 16
if op in ("softmaxbwd","logsoftmaxbwd"):
    x=loadf(pre,".x",(R,C)).requires_grad_(True); go=loadf(pre,".go",(R,C))
    y=(torch.softmax(x,1) if op=="softmaxbwd" else torch.log_softmax(x,1)); y.backward(go); savef(pre,".gi",x.grad)
else:
    x=loadf(pre,".x").requires_grad_(True); go=loadf(pre,".go")
    if   op=="elu":        y=F.elu(x,1.0)
    elif op=="hardtanh":   y=F.hardtanh(x,-0.5,0.5)
    elif op=="leakyrelu":  y=F.leaky_relu(x,0.1)
    elif op=="softshrink": y=F.softshrink(x,0.5)
    elif op=="threshold":  y=F.threshold(x,0.0,0.0)
    elif op=="hardshrink": y=F.hardshrink(x,0.5)
    else: no_ref("torch_bwd", op)
    y.backward(go); savef(pre,".gi",x.grad)
