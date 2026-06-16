#!/usr/bin/env python3
# PyTorch-autograd references for activation/softmax backward. y=f(x); y.backward(go) -> x.grad (.gi).
import sys, numpy as np, torch, torch.nn.functional as F
op, pre = sys.argv[1], sys.argv[2]
N, R, C = 4096, 8, 16
def L(name,n,shape=None):
    a=np.fromfile(pre+name,dtype=np.float32).astype(np.float64); t=torch.from_numpy(a)
    return (t.reshape(shape) if shape else t)
def save(t): t.detach().numpy().astype(np.float32).tofile(pre+".gi")
if op in ("softmaxbwd","logsoftmaxbwd"):
    x=L(".x",R*C,(R,C)).requires_grad_(True); go=L(".go",R*C,(R,C))
    y=(torch.softmax(x,1) if op=="softmaxbwd" else torch.log_softmax(x,1)); y.backward(go); save(x.grad)
else:
    x=L(".x",N).requires_grad_(True); go=L(".go",N)
    if   op=="elu":        y=F.elu(x,1.0)
    elif op=="hardtanh":   y=F.hardtanh(x,-0.5,0.5)
    elif op=="leakyrelu":  y=F.leaky_relu(x,0.1)
    elif op=="softshrink": y=F.softshrink(x,0.5)
    elif op=="threshold":  y=F.threshold(x,0.0,0.0)
    elif op=="hardshrink": y=F.hardshrink(x,0.5)
    else: sys.stderr.write("torch_bwd: no ref %s\n"%op); sys.exit(2)
    y.backward(go); save(x.grad)
