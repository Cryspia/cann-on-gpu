# Qwen3-Next ("Qwen3.5") hybrid-model adaptation

> 中文版 / Chinese: [README.zh-CN.md](README.zh-CN.md)

End-to-end cross-check that the cann-on-gpu shim can run a **Qwen3-Next** forward — the hybrid
linear-attention / SSM model (HF class `Qwen3NextForCausalLM`, `model_type=qwen3_next`) that earlier
could not be reproduced because it needs gated-delta / causal-conv / gated-RMSNorm operators.

Because no small pretrained Qwen3-Next checkpoint exists (only the 80B-A3B release), the test validates the
**architecture mapping** against HuggingFace on a tiny randomly-initialized instance: identical random weights
are dumped from HF, the shim recomputes the forward with `aclnn*` ops, and every stage is compared by
normalized error.

## What is exercised

- **GatedDeltaNet linear-attention layer** (`run_gdn`): in-proj → causal depthwise `aclnnCausalConv1d`+SiLU →
  `beta=sigmoid(b)`, `g=-exp(A_log)·softplus(a+dt_bias)` → head repeat → L2-norm + scaled `aclnnGatedDeltaRule`
  (matches HF `chunk_gated_delta_rule`) → gated RMSNorm (`RmsNorm`·`Silu`·`Mul`) → out-proj.
- **Full tiny model** (`run_qwen35`): token embedding → per layer { zero-centered `aclnnRmsNorm`(·(1+w)) →
  GatedDeltaNet *or* gated full-attention (output-gate, per-head QK-norm, partial RoPE via `aclnnApplyRotaryPosEmb`
  on the rotary slice, GQA `aclnnFlashAttentionScore`) → residual → RMSNorm → sparse MoE (`aclnnMoeGatingTopKSoftmax`
  router + per-expert SwiGLU + sigmoid-gated shared expert) → residual } → final norm → lm_head.

## Run

```bash
source ../../env.sh
./build_qwen35.sh        # exports HF reference, builds, runs both checks
```

Requires the `cann-gpu` conda env with `torch` + `transformers>=5.12` (CPU-only is fine).

## Result

All stages match the HF reference: the GatedDeltaNet layer to ~8e-7, and the full model's per-layer hidden
states, final-norm output, and **logits to ~4e-6** with the greedy next token identical. This confirms the
operator set (P18 SSM ops, P15 MoE routing, P7 gated norm, partial RoPE) composes into a correct Qwen3-Next forward.
