#!/usr/bin/env python3
# Independent reference oracle for the golden cross-check harness, using PyTorch (CPU) instead of
# the Ascend cannsim. Reads the same deterministic fp32 inputs the harness generated (<prefix>.x[/.y])
# and writes the reference output as raw fp32 to <golden.bin> — so cannsim_golden_check's `check`
# can compare the shim against PyTorch. PyTorch is implementation-independent from our shim, so this
# catches "buggy-vs-buggy" cases where a hand-written CPU test reference shares the shim's mistake.
import sys, numpy as np, torch

op, prefix, out = sys.argv[1], sys.argv[2], sys.argv[3]
x = torch.from_numpy(np.fromfile(prefix + ".x", dtype=np.float32).astype(np.float64))
def y():
    return torch.from_numpy(np.fromfile(prefix + ".y", dtype=np.float32).astype(np.float64))

UN = {
    "relu": lambda: torch.relu(x), "exp": lambda: torch.exp(x), "sqrt": lambda: torch.sqrt(x),
    "abs": lambda: torch.abs(x), "neg": lambda: -x, "sigmoid": lambda: torch.sigmoid(x),
    "silu": lambda: torch.nn.functional.silu(x), "gelu": lambda: torch.nn.functional.gelu(x),
    "tanh": lambda: torch.tanh(x), "erf": lambda: torch.erf(x), "erfc": lambda: torch.erfc(x),
    "log": lambda: torch.log(x), "ln": lambda: torch.log(x), "log2": lambda: torch.log2(x), "log10": lambda: torch.log10(x),
    "sin": lambda: torch.sin(x), "cos": lambda: torch.cos(x), "tan": lambda: torch.tan(x),
    "asin": lambda: torch.asin(x), "acos": lambda: torch.acos(x), "atan": lambda: torch.atan(x),
    "asinh": lambda: torch.asinh(x), "acosh": lambda: torch.acosh(x), "atanh": lambda: torch.atanh(x),
    "sinh": lambda: torch.sinh(x), "cosh": lambda: torch.cosh(x),
    "rsqrt": lambda: torch.rsqrt(x), "reciprocal": lambda: torch.reciprocal(x),
    "ceil": lambda: torch.ceil(x), "floor": lambda: torch.floor(x), "round": lambda: torch.round(x),
    "rint": lambda: torch.round(x), "trunc": lambda: torch.trunc(x), "sign": lambda: torch.sign(x),
    "frac": lambda: torch.frac(x), "lgamma": lambda: torch.lgamma(x), "digamma": lambda: torch.digamma(x),
    "sinc": lambda: torch.sinc(x), "erfinv": lambda: torch.erfinv(x),
    "expm1": lambda: torch.expm1(x), "log1p": lambda: torch.log1p(x), "exp2": lambda: torch.exp2(x),
    "softsign": lambda: torch.nn.functional.softsign(x),
}
BIN = {
    "add": lambda: x + y(), "sub": lambda: x - y(), "mul": lambda: x * y(), "div": lambda: x / y(),
    "max": lambda: torch.maximum(x, y()), "min": lambda: torch.minimum(x, y()),
    "power": lambda: torch.pow(x, y()), "hypot": lambda: torch.hypot(x, y()),
    "fmod": lambda: torch.fmod(x, y()), "atan2": lambda: torch.atan2(x, y()),
    "xlogy": lambda: torch.xlogy(x, y()),
}
if op in UN:   r = UN[op]()
elif op in BIN: r = BIN[op]()
else: sys.stderr.write("torch_oracle: no reference for op %s\n" % op); sys.exit(2)
r.numpy().astype(np.float32).tofile(out)
