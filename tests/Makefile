# Client tests: pure g++ (no CUDA headers), links only the backend's libascendcl.so -- simulates Ascend application linking.
# Backend-agnostic: BACKEND points to a GPU backend lib (default CUDA; future: BACKEND=../rocm); API headers shared from top-level ../include.
ACL_INCLUDE ?= $(error run: source env.sh first)
BACKEND ?= ../cuda
INC := ../include

BINS := bin/test_add bin/test_ops bin/test_matmul_conv bin/test_nn bin/test_fp8 bin/test_shape bin/test_index bin/test_pool bin/test_norm bin/test_blas bin/test_loss bin/test_optim bin/test_random bin/test_format bin/test_actbwd bin/test_quant_ext bin/test_rnn bin/test_ssm bin/test_moe bin/test_fft bin/test_linalg bin/test_remainder bin/test_foreach bin/test_inplace bin/test_compare bin/test_sort bin/test_attn_variants bin/test_harness bin/test_runtime bin/test_aclop bin/test_llm bin/test_decode bin/test_graph

all: $(BINS)

bin/%: %.cpp $(BACKEND)/lib/libascendcl.so
	@mkdir -p bin
	g++ -std=c++17 -O2 -I$(ACL_INCLUDE) -I$(INC) $< \
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

clean:
	rm -rf bin

.PHONY: run clean
