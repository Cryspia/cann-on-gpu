# libascendcl.so + libhccl.so — CUDA backend for the CANN/AscendCL API (cuBLASLt/cuDNN/NCCL/hand-written kernels).
# API contract headers live in the top-level ../include/ (backend-agnostic, shared across GPU backends); this directory is the NVIDIA CUDA implementation. Run `source ../env.sh` first.
# cudart is statically linked → the .so can be copied directly to a peer machine without a CUDA toolkit.
GPU_ARCH    ?= sm_121
ACL_INCLUDE ?= $(error Run `source env.sh` first: ACL_INCLUDE is not set)
NCCL_DIR    ?= $(error Run `source env.sh` first: NCCL_DIR is not set)
CUDNN_DIR   ?= $(error Run `source env.sh` first: CUDNN_DIR is not set)
CUDA_HOME   ?= /usr/local/cuda

# All backend sources are compiled into libascendcl.so. hccl.cu is excluded here — it is built
# separately into libhccl.so (see the libhccl rule below). New op files are picked up automatically.
SRCS := $(filter-out src/hccl.cu,$(wildcard src/*.cu)) $(wildcard src/ops/*.cu)

# Top-level shared API contract headers (aclnnop/, hccl/). Use INCDIR to avoid conflicts with the INCLUDE environment variable.
INCDIR := ../include
NVCCFLAGS := -std=c++17 -arch=$(GPU_ARCH) -O2 -Xcompiler -fPIC -cudart static \
             -I$(ACL_INCLUDE) -I$(INCDIR) -Isrc -I$(CUDNN_DIR)/include

all: lib/libascendcl.so lib/libhccl.so

# libascendcl links libhccl so its mc2 ops (HcclAllReduce, …) resolve via a recorded DT_NEEDED dependency
# (clean DAG: hccl is NCCL-backed and never references ascendcl). rpath $ORIGIN finds the sibling .so.
lib/libascendcl.so: $(SRCS) src/internal.h $(INCDIR)/aclnnop/aclnn_add.h $(INCDIR)/aclnnop/aclnn_ops.h lib/libhccl.so
	@mkdir -p lib
	nvcc $(NVCCFLAGS) -shared $(SRCS) \
	    -L$(CUDA_HOME)/lib64 -lcublasLt -lcublas -lcusolver -lcufft -L$(CUDNN_DIR)/lib -lcudnn \
	    -Llib -lhccl -L$(NCCL_DIR)/lib \
	    -Xlinker -rpath -Xlinker $(CUDNN_DIR)/lib -Xlinker -rpath -Xlinker '$$ORIGIN' -o $@

lib/libhccl.so: src/hccl.cu $(INCDIR)/hccl/hccl.h
	@mkdir -p lib
	nvcc $(NVCCFLAGS) -I$(NCCL_DIR)/include -shared src/hccl.cu -L$(NCCL_DIR)/lib -lnccl -o $@

clean:
	rm -rf lib

.PHONY: all clean
