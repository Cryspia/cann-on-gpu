# Qwen3-0.6B Real-Model End-to-End Comparison

> 中文版 / Chinese: [README.zh-CN.md](README.zh-CN.md)

Loads real open-source model weights, runs a full forward pass through the backend's aclnn operator chain, and cross-checks the HF reference logits / greedy-decode tokens one by one. Pure ACL client (links only `libascendcl.so`, the same way an Ascend application links).

## Run

```bash
bash build_qwen3.sh                          # default Qwen/Qwen3-0.6B, from ModelScope
bash build_qwen3.sh Qwen/Qwen3-0.6B "Once upon a time"
bash build_qwen3.sh --hf                     # download from HuggingFace instead of ModelScope
```

On the first run it downloads the model and exports weights to `data/`; if `data/` exists it is skipped (delete `data/` to force re-export). Requires a conda env with `torch` + `transformers` + `safetensors` (CPU build is fine). ModelScope mode also needs the `modelscope` package, which the build script installs automatically if missing.

## Download source

Weights download from **ModelScope by default**. Pass `--hf` (or set `QWEN_USE_HF=1`) to download from HuggingFace instead. The same model id works for both hubs.

## Files

- `export_qwen3.py` — resolves the model (ModelScope by default, HuggingFace on `--hf`), exports all weights (fp32 raw, Linear in native `[out,in]` layout), input token ids, reference logits (both fp32 and bf16), greedy-decoded tokens, and the RoPE `inv_freq`.
- `run_qwen3.cpp` — backend forward harness: Embedding → [RMSNorm → QKV(no bias) → per-head QK-RMSNorm → RoPE → GQA causal attention → O+residual → RMSNorm → SwiGlu MLP(silu(gate)·up) + residual] × 28 → RMSNorm → tied lm_head. Validates (1) prefill logits at every position and per-position top-1, (2) greedy-decoded token hits.
- `build_qwen3.sh` — one-shot export + compile + run.
- `check_l0.py` / `check_hf_l0.py` — per-layer / per-operator numpy fp64 recomputation + HF forward-hook comparison tools for bisecting a mismatch.

## Architecture notes (vs Qwen2)

Qwen3 differs from Qwen2 in the attention block: the q/k/v projections have **no bias**, each head's query and key go through a per-head **RMSNorm** (`q_norm`/`k_norm` over `head_dim`) applied before RoPE, and `head_dim` is explicit (128) rather than `hidden_size / num_heads`. The MLP (SwiGLU) and the rest of the stack are unchanged.

## Precision modes and results

```bash
./run_qwen3 data             # fp32: logits error 5.4e-6, top-1 5/5, greedy 8/8 matches HF-fp32
QWEN_FP16=1 ./run_qwen3 data # fp16: logits error 5.5e-3, top-1 5/5, greedy 8/8 matches HF-fp32
QWEN_BF16=1 ./run_qwen3 data # bf16: vs HF-bf16 ref, logits error 2.3e-2, top-1 5/5, greedy 8/8 matches
```

- fp32 reproduces HF token by token at the fp32 rounding level accumulated over 28 layers.
- The bf16 run is compared against HF's own **bf16** output: the backend's bf16 semantics (tensors stored as bf16, operators accumulate internally in fp32 — RmsNorm `float` accumulation, GEMM `CUBLAS_COMPUTE_32F`, softmax fp32) are faithful to CANN/PyTorch, so the fair baseline is HF bf16 (pure bf16 inference itself drifts from the fp32 greedy decode — this is inherent to bf16, not an implementation issue).

## Debugging

`QWEN_DUMP_HIDDEN=1 ./run_qwen3 data` dumps the per-layer residual stream; `QWEN_DUMP_L0=1 ./run_qwen3 data` dumps intermediate tensors for every operator in layer 0. Use `check_l0.py` / `check_hf_l0.py` to bisect a mismatch operator by operator.
