#!/usr/bin/env python3
# PyTorch-autograd references for loss + norm backward ops (independent of our analytic gradients).
#   loss ops:  gi = d/dself [ loss(self,target,reduction='none') ] applied with upstream go  -> <pre>.gi
#   rmsnorm:   y = x*rsqrt(mean(x^2)+eps)*gamma ; y.backward(gy) -> <pre>.gx, <pre>.gg
#   layernorm: y = layer_norm(x,[D],gamma,beta,eps) ; y.backward(gy) -> <pre>.gx, <pre>.gg, <pre>.gb
import sys, numpy as np, torch, torch.nn.functional as F
op, pre = sys.argv[1], sys.argv[2]
def L(name, n, shape=None):
    a = np.fromfile(pre + name, dtype=np.float32).astype(np.float64)
    t = torch.from_numpy(a)
    return t.reshape(shape) if shape else t
def save(name, t): t.detach().numpy().astype(np.float32).tofile(pre + name)

N, R, D = 4096, 16, 64
if op in ("l1","smoothl1","mse","bce","kldiv","softmargin"):
    self = L(".self", N).requires_grad_(True); tgt = L(".tgt", N); go = L(".go", N)
    if   op=="l1":         loss = F.l1_loss(self, tgt, reduction="none")
    elif op=="smoothl1":   loss = F.smooth_l1_loss(self, tgt, reduction="none", beta=1.0)
    elif op=="mse":        loss = F.mse_loss(self, tgt, reduction="none")
    elif op=="bce":        loss = F.binary_cross_entropy(self, tgt, reduction="none")
    elif op=="kldiv":      loss = F.kl_div(self, tgt, reduction="none", log_target=False)
    else:                  loss = F.soft_margin_loss(self, tgt, reduction="none")
    loss.backward(go); save(".gi", self.grad)
elif op=="rmsnorm":
    x = L(".x", R*D, (R,D)).requires_grad_(True); gamma = L(".gamma", D).requires_grad_(True); gy = L(".gy", R*D, (R,D))
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    y = x * rms * gamma
    y.backward(gy); save(".gx", x.grad); save(".gg", gamma.grad)
elif op=="layernorm":
    x = L(".x", R*D, (R,D)).requires_grad_(True); gamma = L(".gamma", D).requires_grad_(True); beta = L(".beta", D).requires_grad_(True); gy = L(".gy", R*D, (R,D))
    y = F.layer_norm(x, (D,), weight=gamma, bias=beta, eps=1e-5)
    y.backward(gy); save(".gx", x.grad); save(".gg", gamma.grad); save(".gb", beta.grad)
else:
    sys.stderr.write("torch_grad2: no ref for %s\n" % op); sys.exit(2)
