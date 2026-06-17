#!/usr/bin/env python3
# PyTorch refs for generators + activations + RNN (LSTM/GRU via nn modules with set weights, float64).
from torch_common import *
op, pre = sys.argv[1], sys.argv[2]
def Lf(name,shape): return loadf(pre, name, shape)
def save(t): savef(pre, ".out", t)
S,Bt,I,H = 3,2,4,5
if   op=="arange":   save(torch.arange(0.5,5.0,0.5))
elif op=="eye":      save(torch.eye(4))
elif op=="linspace": save(torch.linspace(0.0,1.0,8))
elif op=="hardtanh": save(torch.clamp(Lf(".a",16),-0.5,0.5))
elif op=="logsigmoidfwd": save(F.logsigmoid(Lf(".a",16)))
elif op=="renorm":   save(torch.renorm(Lf(".a",(4,5)),2,0,1.0))
elif op=="lstm":
    m=nn.LSTM(I,H,1).double()
    with torch.no_grad():
        m.weight_ih_l0.copy_(Lf(".wih",(4*H,I))); m.weight_hh_l0.copy_(Lf(".whh",(4*H,H)))
        m.bias_ih_l0.copy_(Lf(".bih",(4*H,))); m.bias_hh_l0.copy_(Lf(".bhh",(4*H,)))
    y,_=m(Lf(".x",(S,Bt,I)),(Lf(".h0",(Bt,H)).unsqueeze(0),Lf(".c0",(Bt,H)).unsqueeze(0))); save(y)
elif op=="gru":
    m=nn.GRU(I,H,1).double()
    with torch.no_grad():
        m.weight_ih_l0.copy_(Lf(".wih",(3*H,I))); m.weight_hh_l0.copy_(Lf(".whh",(3*H,H)))
        m.bias_ih_l0.copy_(Lf(".bih",(3*H,))); m.bias_hh_l0.copy_(Lf(".bhh",(3*H,)))
    y,_=m(Lf(".x",(S,Bt,I)),Lf(".h0",(Bt,H)).unsqueeze(0)); save(y)
else: no_ref("torch_gen", op)
