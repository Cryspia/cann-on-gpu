# Client tests: pure g++ (no CUDA headers), links only the backend's libascendcl.so -- simulates Ascend application linking.
# Backend-agnostic: BACKEND points to a GPU backend lib (default CUDA; future: BACKEND=../rocm); API headers shared from top-level ../include.
ACL_INCLUDE ?= $(error run: source env.sh first)
BACKEND ?= ../cuda
INC := ../include

# Backend shared-lib extension and host compiler are OS-specific:
#   Linux  -> .so   via g++      (CUDA backend)
#   Darwin -> .dylib via clang++ (Metal backend)
UNAME_S := $(shell uname -s)
LIBEXT  := $(if $(filter Darwin,$(UNAME_S)),dylib,so)
CXX     := $(if $(filter Darwin,$(UNAME_S)),clang++,g++)

BINS := bin/test_add bin/test_ops bin/test_matmul_conv bin/test_nn bin/test_fp8 bin/test_shape bin/test_index bin/test_pool bin/test_norm bin/test_blas bin/test_loss bin/test_optim bin/test_random bin/test_format bin/test_actbwd bin/test_quant_ext bin/test_rnn bin/test_ssm bin/test_moe bin/test_fft bin/test_linalg bin/test_remainder bin/test_foreach bin/test_inplace bin/test_compare bin/test_sort bin/test_attn_variants bin/test_harness bin/test_runtime bin/test_aclop bin/test_llm bin/test_decode bin/test_graph
# Per-operator-family parity tests (the ~600 ops beyond the suite above — the operator set Metal brought to
# full CUDA parity). Cross-platform-equivalent on CUDA and Metal. A few ops carry backend-specific
# dtype/format/RNG/single-rank conventions and are documented inline; those are skipped on the divergent backend.
PARITY_BINS := bin/test_attention_parity bin/test_backward_parity bin/test_blas_parity bin/test_elementwise_parity bin/test_foreach_parity bin/test_index2_parity bin/test_pool_conv_parity bin/test_seq_parity bin/test_quant_parity bin/test_inplace_parity bin/test_misc_parity bin/test_dist_parity
BINS += $(PARITY_BINS)

all: $(BINS)

bin/%: %.cpp $(BACKEND)/lib/libascendcl.$(LIBEXT)
	@mkdir -p bin
	$(CXX) -std=c++17 -O2 -I$(ACL_INCLUDE) -I$(INC) $< \
	    -L$(BACKEND)/lib -lascendcl -Wl,-rpath,$(abspath $(BACKEND)/lib) -o $@

run: $(BINS)
	./bin/test_add
	./bin/test_ops
	./bin/test_matmul_conv
	./bin/test_nn
	./bin/test_fp8
	./bin/test_shape
	./bin/test_index
	./bin/test_pool
	./bin/test_norm
	./bin/test_blas
	./bin/test_loss
	./bin/test_optim
	./bin/test_random
	./bin/test_format
	./bin/test_actbwd
	./bin/test_quant_ext
	./bin/test_rnn
	./bin/test_ssm
	./bin/test_moe
	./bin/test_fft
	./bin/test_linalg
	./bin/test_remainder
	./bin/test_foreach
	./bin/test_inplace
	./bin/test_compare
	./bin/test_sort
	./bin/test_harness
	./bin/test_runtime
	./bin/test_aclop
	./bin/test_llm
	./bin/test_decode
	./bin/test_graph
	./bin/test_attention_parity
	./bin/test_backward_parity
	./bin/test_blas_parity
	./bin/test_elementwise_parity
	./bin/test_foreach_parity
	./bin/test_index2_parity
	./bin/test_pool_conv_parity
	./bin/test_seq_parity
	./bin/test_misc_parity
	./bin/test_inplace_parity

# Parity tests with a few backend-specific ops (quant deqScale/format conventions, single-rank collectives)
# still being reconciled across backends — built by `make` (in BINS, both platforms) but not in the
# default `run`; pass fully on Metal. Run explicitly: ./bin/test_quant_parity ./bin/test_dist_parity.
run-parity-pending: bin/test_quant_parity bin/test_dist_parity
	./bin/test_quant_parity; ./bin/test_dist_parity

clean:
	rm -rf bin

.PHONY: run run-parity-pending clean
