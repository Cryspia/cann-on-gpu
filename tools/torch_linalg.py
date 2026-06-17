#!/usr/bin/env python3
# PyTorch references for linalg ops. Unique-output ops compared directly to torch.linalg; non-unique
# decompositions (qr/svd) are checked by reconstruction inside linalg_check.cpp (qr needs no torch ref,
# svd also gets its singular values .S compared). Inputs are produced by the C++ gen and read here.
from torch_common import *
op, pre = sys.argv[1], sys.argv[2]
n, c3, m3 = 4, 3, 3
def Lf(name, shape): return loadf(pre, name, shape)

if op=="inverse":        savef(pre,".out", torch.linalg.inv(Lf(".A",(n,n))))
elif op=="det":          savef(pre,".out", torch.linalg.det(Lf(".A",(n,n))).reshape(1))
elif op=="slogdet":
    s,l = torch.linalg.slogdet(Lf(".A",(n,n))); savef(pre,".sign", s.reshape(1)); savef(pre,".logabs", l.reshape(1))
elif op=="cholesky":     savef(pre,".out", torch.linalg.cholesky(Lf(".A",(n,n))))     # lower
elif op=="svd":          savef(pre,".S", torch.linalg.svdvals(Lf(".A",(n,n))))         # sorted desc
elif op=="qr":           pass                                                          # reconstruction-only
elif op=="solve":        savef(pre,".out", torch.linalg.solve(Lf(".A",(n,n)), Lf(".B",(n,n))))
elif op=="triangularsolve":
    savef(pre,".out", torch.linalg.solve_triangular(Lf(".A",(n,n)), Lf(".B",(n,n)), upper=True, left=True, unitriangular=False))
elif op=="cross":        savef(pre,".out", torch.linalg.cross(Lf(".u",(c3,)), Lf(".v",(c3,))))
elif op=="trace":        savef(pre,".out", torch.trace(Lf(".A",(n,n))).reshape(1))
elif op=="diag":         savef(pre,".out", torch.diag(Lf(".A",(n,n))))                 # 2D -> diagonal
elif op=="matrixpower":  savef(pre,".out", torch.linalg.matrix_power(Lf(".A",(n,n)), 3))
elif op=="dot":          savef(pre,".out", torch.dot(Lf(".u",(n,)), Lf(".v",(n,))).reshape(1))
elif op=="outer":        savef(pre,".out", torch.outer(Lf(".u",(n,)), Lf(".v",(m3,))))
else: no_ref("torch_linalg", op)
