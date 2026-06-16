#!/usr/bin/env python3
# PyTorch references for linalg ops. Unique-output ops compared directly to torch.linalg; non-unique
# decompositions (qr/svd) are checked by reconstruction inside grad_check7.cpp (qr needs no torch ref,
# svd also gets its singular values .S compared). Inputs are produced by the C++ gen and read here.
import sys, numpy as np, torch
op, pre = sys.argv[1], sys.argv[2]
n, c3, m3 = 4, 3, 3
def Lf(name, shape):
    return torch.from_numpy(np.fromfile(pre+name,dtype=np.float32).astype(np.float64)).reshape(shape)
def save(name,t): np.asarray(t,dtype=np.float64).astype(np.float32).tofile(pre+name)

if op=="inverse":        save(".out", torch.linalg.inv(Lf(".A",(n,n))))
elif op=="det":          save(".out", torch.linalg.det(Lf(".A",(n,n))).reshape(1))
elif op=="slogdet":
    s,l = torch.linalg.slogdet(Lf(".A",(n,n))); save(".sign", s.reshape(1)); save(".logabs", l.reshape(1))
elif op=="cholesky":     save(".out", torch.linalg.cholesky(Lf(".A",(n,n))))     # lower
elif op=="svd":          save(".S", torch.linalg.svdvals(Lf(".A",(n,n))))         # sorted desc
elif op=="qr":           pass                                                     # reconstruction-only
elif op=="solve":        save(".out", torch.linalg.solve(Lf(".A",(n,n)), Lf(".B",(n,n))))
elif op=="triangularsolve":
    save(".out", torch.linalg.solve_triangular(Lf(".A",(n,n)), Lf(".B",(n,n)), upper=True, left=True, unitriangular=False))
elif op=="cross":        save(".out", torch.linalg.cross(Lf(".u",(c3,)), Lf(".v",(c3,))))
elif op=="trace":        save(".out", torch.trace(Lf(".A",(n,n))).reshape(1))
elif op=="diag":         save(".out", torch.diag(Lf(".A",(n,n))))                 # 2D -> diagonal
elif op=="matrixpower":  save(".out", torch.linalg.matrix_power(Lf(".A",(n,n)), 3))
elif op=="dot":          save(".out", torch.dot(Lf(".u",(n,)), Lf(".v",(n,))).reshape(1))
elif op=="outer":        save(".out", torch.outer(Lf(".u",(n,)), Lf(".v",(m3,))))
else: sys.stderr.write("torch_grad7: no ref for %s\n"%op); sys.exit(2)
