#!/usr/bin/env python3
# PyTorch-autograd reference for activation backward ops. Reads <prefix>.x (forward input) and
# <prefix>.go (upstream gradient), runs y=f(x); y.backward(go), writes x.grad as the reference
# gradInput (raw fp32) to <ref.bin>. Autograd is independent of our hand-derived analytic f'.
import sys, numpy as np, torch, torch.nn.functional as F

op, prefix, out = sys.argv[1], sys.argv[2], sys.argv[3]
x  = torch.from_numpy(np.fromfile(prefix + ".x",  dtype=np.float32).astype(np.float64)).requires_grad_(True)
go = torch.from_numpy(np.fromfile(prefix + ".go", dtype=np.float32).astype(np.float64))

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
if op not in F_: sys.stderr.write("torch_grad: no forward for %s\n" % op); sys.exit(2)
y = F_[op](x)
y.backward(go)
x.grad.numpy().astype(np.float32).tofile(out)
