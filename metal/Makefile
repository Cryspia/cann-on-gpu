# libascendcl.dylib + default.metallib — Apple Metal backend for the CANN/AscendCL API.
# Implements the SAME contract as ../include/ (shared with the CUDA backend); this directory is the Metal impl.
# Host layer is Objective-C++ (.mm) so it can drive MPS / MPSGraph (Objective-C-only frameworks);
# compute kernels are hand-written MSL (.metal) compiled into one default.metallib loaded at runtime.
#
# Run `source ../env.sh` first (BACKEND=metal). env.sh scopes Xcode via DEVELOPER_DIR (PROJECT-LOCAL —
# it does NOT change your global `xcode-select -p`) and exports METAL / METALLIB / METAL_STD / ACL_INCLUDE.

ACL_INCLUDE ?= $(error Run `source ../env.sh` first: ACL_INCLUDE is not set)

# Metal Toolchain compiler lives in a cryptex mount; env.sh exports these. Fall back to a glob if run standalone.
METAL_BIN     ?= $(firstword $(wildcard /var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-*/Metal.xctoolchain/usr/bin))
METAL         ?= $(METAL_BIN)/metal
METALLIB      ?= $(METAL_BIN)/metallib
METAL_STD     ?= metal3.1
DEVELOPER_DIR ?= /Applications/Xcode.app/Contents/Developer
export DEVELOPER_DIR

INCDIR := ../include

# .mm = host/runtime + op dispatch (calls MPS/MPSGraph/Accelerate); .metal = hand-written compute kernels.
# hccl.mm is excluded here — it builds separately into libhccl.dylib (single-rank stub; mirrors the CUDA backend).
MM_SRCS    := $(filter-out src/hccl.mm,$(wildcard src/*.mm)) $(wildcard src/ops/*.mm)
METAL_SRCS := $(wildcard src/*.metal) $(wildcard src/ops/*.metal)
OBJS       := $(MM_SRCS:%.mm=build/%.o)
AIRS       := $(METAL_SRCS:%.metal=build/%.air)

CXX        := clang++
CXXFLAGS   := -std=c++17 -ObjC++ -fobjc-arc -O2 -fPIC -I$(ACL_INCLUDE) -I$(INCDIR) -Isrc
FRAMEWORKS := -framework Metal -framework Foundation \
              -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Accelerate
MTLFLAGS   := -std=$(METAL_STD) -fno-fast-math -Isrc   # precise math: hand-written kernels need ~1e-6, not fast-math

all: lib/libascendcl.dylib lib/libhccl.dylib lib/default.metallib

# HCCL single-rank stub (no Metal frameworks needed; pure host memcpy collectives).
lib/libhccl.dylib: src/hccl.mm $(INCDIR)/hccl/hccl.h
	@mkdir -p lib
	$(CXX) $(CXXFLAGS) -dynamiclib -install_name @rpath/libhccl.dylib src/hccl.mm -o $@

# Every compute kernel links into one default.metallib; the runtime loads it next to the dylib at aclInit.
lib/default.metallib: $(AIRS)
	@mkdir -p lib
	$(METALLIB) $(AIRS) -o $@

build/%.air: %.metal
	@mkdir -p $(dir $@)
	$(METAL) $(MTLFLAGS) -c $< -o $@

build/%.o: %.mm
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# @rpath install_name so test/tool binaries resolve it via -Wl,-rpath (mirrors the CUDA backend's $ORIGIN rpath).
lib/libascendcl.dylib: $(OBJS) | lib/default.metallib
	@mkdir -p lib
	$(CXX) $(CXXFLAGS) -dynamiclib -install_name @rpath/libascendcl.dylib $(OBJS) $(FRAMEWORKS) -o $@

clean:
	rm -rf build lib

.PHONY: all clean
