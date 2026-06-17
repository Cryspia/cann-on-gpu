#!/usr/bin/env python3
# PyTorch-autograd reference for activation backward ops. Reads <prefix>.x (forward input) and
# <prefix>.go (upstream gradient), runs y=f(x); y.backward(go), writes x.grad as the reference
# gradInput (raw fp32) to <ref.bin>. Autograd is independent of our hand-derived analytic f'.
from torch_common import *

op, prefix, out = sys.argv[1], sys.argv[2], sys.argv[3]
x  = loadf(prefix, ".x").requires_grad_(True)
go = loadf(prefix, ".go")

F_ = {
    "relu":       lambda t: torch.relu(t),
    "gelu":       lambda t: F.gelu(t),                       # exact erf gelu (matches our GeluBackward)
    "fastgelu":   lambda t: F.gelu(t, approximate="tanh"),   # tanh-approx (matches our FastGeluBackward)
    "silu":       lambda t: F.silu(t),
    "softplus":   lambda t: F.softplus(t),
    "hardswish":  lambda t: F.hardswish(t),
    "sigmoid":    lambda t: torch.sigmoid(t),
    "tanh":       lambda t: torch.tanh(t),
    "hardsigmoid":lambda t: F.hardsigmoid(t),
    "logsigmoid": lambda t: F.logsigmoid(t),
    "mish":       lambda t: F.mish(t),
    "selu":       lambda t: F.selu(t),
}
if op not in F_: no_ref("torch_actgrad", op)
y = F_[op](x)
y.backward(go)
savef("", out, x.grad)
