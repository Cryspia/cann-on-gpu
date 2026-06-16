#!/usr/bin/env python3
# End-to-end real open-source small-model demo: export Qwen3-0.6B weights + reference logits + greedy decode for the C++ shim forward cross-check.
# Usage: python export_qwen3.py [model_id] [out_dir] [prompt]
#   Model weights download from ModelScope by default; pass --hf (or set QWEN_USE_HF=1) to download from HuggingFace instead.
# Outputs (out_dir/):
#   config.txt   : D L Hq Hkv hd F V eps theta S n_gen tied
#   ids.bin      : int64[S]    input token ids
#   gen.bin      : int64[n_gen] HF greedy decode token ids (reference)
#   logits.bin   : fp32[S*V]   HF reference logits (all positions)
#   embed.bin    : fp32[V*D]   embed_tokens.weight (native [V,D]; tied -> also used as lm_head)
#   gnorm.bin    : fp32[D]     model.norm.weight
#   inv_freq.bin : fp32[hd/2]  real HF RoPE frequencies
#   l{L}.{g1,q.w,k.w,v.w,qn,kn,o.w,g2,gate.w,up.w,down.w} : per-layer weights (Linear in native [out,in] layout)
#     Qwen3 vs Qwen2: attention projections have NO bias; q/k get a per-head RMSNorm (qn/kn, size head_dim) applied before RoPE.
import sys, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- CLI ----
argv = [a for a in sys.argv[1:] if a != "--hf"]
USE_HF = ("--hf" in sys.argv) or bool(os.environ.get("QWEN_USE_HF"))
MODEL  = argv[0] if len(argv) > 0 else "Qwen/Qwen3-0.6B"
OUT    = argv[1] if len(argv) > 1 else os.path.join(os.path.dirname(__file__), "data")
PROMPT = argv[2] if len(argv) > 2 else "The capital of France is"
N_GEN  = 8
os.makedirs(OUT, exist_ok=True)

# ---- Resolve a local model path: ModelScope by default, HuggingFace on opt-in ----
def resolve_model(model_id):
    if USE_HF:
        print(f"[src] HuggingFace: {model_id}")
        return model_id          # transformers fetches from the HF hub
    try:
        from modelscope import snapshot_download
    except ImportError:
        sys.exit("[FATAL] modelscope not installed. Install it (`pip install modelscope`) or pass --hf to use HuggingFace.")
    print(f"[src] ModelScope: {model_id}")
    return snapshot_download(model_id)

def w(name, arr):
    arr = np.ascontiguousarray(arr.detach().to(torch.float32).cpu().numpy())
    arr.tofile(os.path.join(OUT, name))
    return arr.shape

MODEL_PATH = resolve_model(MODEL)
print(f"[load] {MODEL}")
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
model.eval()
cfg = model.config
D, L = cfg.hidden_size, cfg.num_hidden_layers
Hq, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
hd = getattr(cfg, "head_dim", None) or (D // Hq)   # Qwen3 head_dim is explicit and may differ from hidden_size//num_heads
F = cfg.intermediate_size
Vsz = cfg.vocab_size
eps = cfg.rms_norm_eps
# rope_theta location varies across transformers versions; do not parse it -- read the real inv_freq straight from the model's rotary_emb
# (the C++ side uses this to build cos/sin), which sidesteps theta-parsing pitfalls entirely.
inv_freq = model.model.rotary_emb.inv_freq.detach().cpu().numpy().astype(np.float32)   # [hd/2]
theta = float(getattr(cfg, "rope_theta", 0.0)) or -1.0
tied = bool(getattr(cfg, "tie_word_embeddings", False))
print(f"[cfg] D={D} L={L} Hq={Hq} Hkv={Hkv} hd={hd} F={F} V={Vsz} eps={eps} theta={theta} tied={tied}")

ids = tok(PROMPT, return_tensors="pt").input_ids
S = ids.shape[1]
print(f"[prompt] {PROMPT!r} -> {S} tokens: {ids[0].tolist()}")

with torch.no_grad():
    out = model(ids, output_hidden_states=True)
    logits = out.logits[0]                      # [S, V]
    # hidden_states: tuple(L+1) of [1,S,D] -- hs[0]=embedding output, hs[i]=residual stream after layer i. Dumped for layer-by-layer bisect.
    hs = torch.stack([h[0] for h in out.hidden_states], dim=0)   # [L+1, S, D]
    hs.to(torch.float32).cpu().numpy().astype(np.float32).tofile(os.path.join(OUT, "hidden.bin"))
    print(f"[dbg] dumped hidden_states {tuple(hs.shape)} -> hidden.bin")
    # Greedy decode N_GEN tokens (used for token-level cross-check)
    gen = model.generate(ids, max_new_tokens=N_GEN, do_sample=False, num_beams=1)
    gen_ids = gen[0, S:].tolist()
print(f"[ref] greedy next {N_GEN}: {gen_ids}  ({tok.decode(gen_ids)!r})")

# ---- bf16 reference (fair comparison for shim bf16: vanilla bf16 also diverges from fp32 greedy decode) ----
# CANN/PyTorch bf16 semantics: tensors stored as bf16, operators accumulate internally in fp32 (norm/softmax/matmul all accumulate in fp32).
model_bf16 = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).eval()
with torch.no_grad():
    logits_bf16 = model_bf16(ids).logits[0].float()                     # [S,V]
    gen_bf16 = model_bf16.generate(ids, max_new_tokens=N_GEN, do_sample=False, num_beams=1)[0, S:].tolist()
print(f"[ref-bf16] greedy next {N_GEN}: {gen_bf16}  ({tok.decode(gen_bf16)!r})")
logits_bf16.cpu().numpy().astype(np.float32).tofile(os.path.join(OUT, "logits_bf16.bin"))
np.array(gen_bf16, dtype=np.int64).tofile(os.path.join(OUT, "gen_bf16.bin"))

# ---- dump ----
np.array(ids[0].tolist(), dtype=np.int64).tofile(os.path.join(OUT, "ids.bin"))
np.array(gen_ids, dtype=np.int64).tofile(os.path.join(OUT, "gen.bin"))
logits.detach().to(torch.float32).cpu().numpy().astype(np.float32).tofile(os.path.join(OUT, "logits.bin"))

sd = model.state_dict()
def g(name): return sd[name]
inv_freq.tofile(os.path.join(OUT, "inv_freq.bin"))   # [hd/2] real HF RoPE frequencies
w("embed.bin", g("model.embed_tokens.weight"))      # [V, D]
w("gnorm.bin", g("model.norm.weight"))              # [D]
for l in range(L):
    p = f"model.layers.{l}."
    w(f"l{l}.g1",     g(p+"input_layernorm.weight"))
    w(f"l{l}.q.w",    g(p+"self_attn.q_proj.weight"))   # no bias in Qwen3
    w(f"l{l}.k.w",    g(p+"self_attn.k_proj.weight"))
    w(f"l{l}.v.w",    g(p+"self_attn.v_proj.weight"))
    w(f"l{l}.qn",     g(p+"self_attn.q_norm.weight"))   # per-head RMSNorm weight [hd], applied before RoPE
    w(f"l{l}.kn",     g(p+"self_attn.k_norm.weight"))
    w(f"l{l}.o.w",    g(p+"self_attn.o_proj.weight"))
    w(f"l{l}.g2",     g(p+"post_attention_layernorm.weight"))
    w(f"l{l}.gate.w", g(p+"mlp.gate_proj.weight"))
    w(f"l{l}.up.w",   g(p+"mlp.up_proj.weight"))
    w(f"l{l}.down.w", g(p+"mlp.down_proj.weight"))
# lm_head (if tied, equals embed; otherwise export separately)
if not tied:
    w("lmhead.bin", g("lm_head.weight"))            # [V, D]

with open(os.path.join(OUT, "config.txt"), "w") as f:
    f.write(f"{D} {L} {Hq} {Hkv} {hd} {F} {Vsz} {eps:.10g} {theta:.10g} {S} {N_GEN} {1 if tied else 0}\n")
print(f"[done] exported to {OUT}")
