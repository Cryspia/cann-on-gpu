#!/usr/bin/env python3
# Shared helpers for the torch_*.py PyTorch reference oracles. Each oracle is paired 1:1 with a
# *_check.cpp harness: the C++ side writes deterministic fp32 inputs to <prefix><name> via `gen`,
# the oracle reads them, computes a PyTorch (CPU, float64) reference, and writes it back as fp32;
# the C++ `check` then compares the shim against this reference. PyTorch is implementation-independent
# from the shim, so this catches "buggy-vs-buggy" errors a hand-written CPU reference would share.
#
# The load/save incantations were copy-pasted across every oracle; they live here once. Re-exports
# sys/np/torch/nn/F so an oracle needs only `from torch_common import *`.
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["sys", "np", "torch", "nn", "F",
           "loadf", "loadi", "loadi32", "loadu8", "savef", "savei", "no_ref"]


def _load(pre, name, dtype, cast, shape):
    t = torch.from_numpy(np.fromfile(pre + name, dtype=dtype).astype(cast))
    return t.reshape(shape) if shape is not None else t

# fp32 file -> float64 tensor (the reference dtype). The legacy oracles passed a redundant element
# count alongside the shape; np.fromfile reads the whole file, so only the shape is needed here.
def loadf(pre, name, shape=None):   return _load(pre, name, np.float32, np.float64, shape)
def loadi(pre, name, shape=None):   return _load(pre, name, np.int64,   np.int64,   shape)   # raw int64
def loadi32(pre, name, shape=None): return _load(pre, name, np.int32,   np.int64,   shape)   # int32 file -> int64
def loadu8(pre, name, shape=None):  return _load(pre, name, np.uint8,    np.uint8,   shape)   # bytes (-> .bool())


# Write a tensor/array back as fp32 (resp. int64). atleast_1d keeps 0-d scalars writable; tofile
# always flattens in C order, so shape/contiguity of the input does not matter.
def savef(pre, name, t):
    a = t.detach().numpy() if hasattr(t, "detach") else t
    np.atleast_1d(np.asarray(a, dtype=np.float32)).tofile(pre + name)

def savei(pre, name, t):
    a = t.detach().numpy() if hasattr(t, "detach") else t
    np.atleast_1d(np.asarray(a, dtype=np.int64)).tofile(pre + name)


def no_ref(tool, op):
    sys.stderr.write("%s: no ref %s\n" % (tool, op)); sys.exit(2)
