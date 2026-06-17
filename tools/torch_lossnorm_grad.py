#!/usr/bin/env python3
# PyTorch-autograd references for loss + norm backward ops (independent of our analytic gradients).
#   loss ops:  gi = d/dself [ loss(self,target,reduction='none') ] applied with upstream go  -> <pre>.gi
#   rmsnorm:   y = x*rsqrt(mean(x^2)+eps)*gamma ; y.backward(gy) -> <pre>.gx, <pre>.gg
#   layernorm: y = layer_norm(x,[D],gamma,beta,eps) ; y.backward(gy) -> <pre>.gx, <pre>.gg, <pre>.gb
from torch_common import *
op, pre = sys.argv[1], sys.argv[2]

N, R, D = 4096, 16, 64
if op in ("l1","smoothl1","mse","bce","kldiv","softmargin"):
    self = loadf(pre,".self").requires_grad_(True); tgt = loadf(pre,".tgt"); go = loadf(pre,".go")
    if   op=="l1":         loss = F.l1_loss(self, tgt, reduction="none")
    elif op=="smoothl1":   loss = F.smooth_l1_loss(self, tgt, reduction="none", beta=1.0)
    elif op=="mse":        loss = F.mse_loss(self, tgt, reduction="none")
    elif op=="bce":        loss = F.binary_cross_entropy(self, tgt, reduction="none")
    elif op=="kldiv":      loss = F.kl_div(self, tgt, reduction="none", log_target=False)
    else:                  loss = F.soft_margin_loss(self, tgt, reduction="none")
    loss.backward(go); savef(pre, ".gi", self.grad)
elif op=="rmsnorm":
    x = loadf(pre,".x",(R,D)).requires_grad_(True); gamma = loadf(pre,".gamma").requires_grad_(True); gy = loadf(pre,".gy",(R,D))
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    y = x * rms * gamma
    y.backward(gy); savef(pre,".gx", x.grad); savef(pre,".gg", gamma.grad)
elif op=="layernorm":
    x = loadf(pre,".x",(R,D)).requires_grad_(True); gamma = loadf(pre,".gamma").requires_grad_(True); beta = loadf(pre,".beta").requires_grad_(True); gy = loadf(pre,".gy",(R,D))
    y = F.layer_norm(x, (D,), weight=gamma, bias=beta, eps=1e-5)
    y.backward(gy); savef(pre,".gx", x.grad); savef(pre,".gg", gamma.grad); savef(pre,".gb", beta.grad)
else:
    no_ref("torch_lossnorm_grad", op)
