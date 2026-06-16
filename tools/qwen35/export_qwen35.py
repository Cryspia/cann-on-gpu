#!/usr/bin/env python3
# Export a tiny randomly-initialized Qwen3-Next (== "Qwen3.5") model + reference logits/hidden-states for the
# cann-on-gpu shim full-forward cross-check. No pretrained checkpoint exists at small scale, so we validate the
# architecture mapping (GatedDeltaNet linear-attn + gated full-attn + per-layer sparse MoE) against HF on random weights.
import os, numpy as np, torch
from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
torch.manual_seed(0)
OUT = os.path.join(os.path.dirname(__file__), "data_full"); os.makedirs(OUT, exist_ok=True)
def w(name, t): np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy()).tofile(os.path.join(OUT, name))

D, L, NH, NKV, HD = 32, 4, 4, 2, 8
cfg = Qwen3NextConfig(
    hidden_size=D, num_hidden_layers=L, num_attention_heads=NH, num_key_value_heads=NKV, head_dim=HD,
    intermediate_size=64, vocab_size=64, max_position_embeddings=64, rms_norm_eps=1e-6, partial_rotary_factor=0.25,
    linear_num_key_heads=2, linear_key_head_dim=16, linear_num_value_heads=4, linear_value_head_dim=16, linear_conv_kernel_dim=4,
    num_experts=4, num_experts_per_tok=2, shared_expert_intermediate_size=64, moe_intermediate_size=64, decoder_sparse_step=1,
    layer_types=['linear_attention','linear_attention','linear_attention','full_attention'])
m = Qwen3NextForCausalLM(cfg).eval()
for p in m.parameters(): torch.nn.init.normal_(p, 0.0, 0.2)
S = 6
ids = torch.arange(1, 1+S).unsqueeze(0) % cfg.vocab_size
with torch.no_grad():
    out = m(ids, output_hidden_states=True)
logits = out.logits[0]                      # [S,V]
hs = [h[0] for h in out.hidden_states]      # L+1 x [S,D]

nk,nv,hkd,hvd,ck = 2,16,16,16,4   # gdn dims (key/value head dims)
GNK, GHKD, GNV, GHVD = cfg.linear_num_key_heads, cfg.linear_key_head_dim, cfg.linear_num_value_heads, cfg.linear_value_head_dim
key_dim, value_dim = GNK*GHKD, GNV*GHVD
rotary_dim = int(HD * cfg.partial_rotary_factor)
with open(os.path.join(OUT,"meta.txt"),"w") as f:
    f.write(f"D={D} L={L} NH={NH} NKV={NKV} HD={HD} V={cfg.vocab_size} S={S} eps={cfg.rms_norm_eps} rotary_dim={rotary_dim}\n")
    f.write(f"GNK={GNK} GHKD={GHKD} GNV={GNV} GHVD={GHVD} key_dim={key_dim} value_dim={value_dim} conv_k={cfg.linear_conv_kernel_dim}\n")
    f.write(f"E={cfg.num_experts} topk={cfg.num_experts_per_tok} moe_inter={cfg.moe_intermediate_size} shared_inter={cfg.shared_expert_intermediate_size}\n")
    f.write("layer_types=" + ",".join(cfg.layer_types) + "\n")
w("ids.bin", ids[0].float()); w("logits.bin", logits)
w("embed.bin", m.model.embed_tokens.weight); w("lm_head.bin", m.lm_head.weight); w("fnorm.bin", m.model.norm.weight)
w("inv_freq.bin", m.model.rotary_emb.inv_freq)
for i,h in enumerate(hs): w(f"hs{i}.bin", h)          # hs0 = embeddings, hs{i} = after layer i-1
# ---- layer-0 sub-block dumps (localize shim mismatches): input_ln -> linear_attn -> residual -> post_ln -> moe ----
with torch.no_grad():
    ly0 = m.model.layers[0]
    ln0 = ly0.input_layernorm(hs[0].unsqueeze(0))
    la0 = ly0.linear_attn(hidden_states=ln0, cache_params=None, attention_mask=None)
    r1  = hs[0].unsqueeze(0) + la0
    pln0= ly0.post_attention_layernorm(r1)
    mo0 = ly0.mlp(pln0)
    if isinstance(mo0, tuple): mo0 = mo0[0]
    w("dbg_ln0.bin", ln0[0]); w("dbg_la0.bin", la0[0]); w("dbg_post0.bin", pln0[0]); w("dbg_moe0.bin", mo0[0])
gen = []
with torch.no_grad():                                  # greedy reference continuation
    cur = ids.clone()
    for _ in range(6):
        nt = m(cur).logits[0,-1].argmax().item(); gen.append(nt); cur = torch.cat([cur, torch.tensor([[nt]])],1)
np.array(gen, dtype=np.int64).tofile(os.path.join(OUT,"gen.bin"))

def deinter_qkvz(W):   # [192,D] per-k-head-blocked -> clean Wq,Wk,Wv,Wz
    blk=2*GHKD+2*GHVD*(GNV//GNK); vblk=GHVD*(GNV//GNK); qs=ks=vs=zs=None; Q=[];K=[];Vv=[];Z=[]
    for g in range(GNK):
        b=g*blk; Q.append(W[b:b+GHKD]);K.append(W[b+GHKD:b+2*GHKD]);Vv.append(W[b+2*GHKD:b+2*GHKD+vblk]);Z.append(W[b+2*GHKD+vblk:b+blk])
    return torch.cat(Q),torch.cat(K),torch.cat(Vv),torch.cat(Z)
def deinter_ba(W):
    bba=GNV//GNK; Bs=[];As=[]
    for g in range(GNK):
        b=g*2*bba; Bs.append(W[b:b+bba]); As.append(W[b+bba:b+2*bba])
    return torch.cat(Bs),torch.cat(As)

for i,lt in enumerate(cfg.layer_types):
    ly = m.model.layers[i]; p=f"l{i}."
    w(p+"input_ln.bin", ly.input_layernorm.weight)
    w(p+"post_ln.bin",  ly.post_attention_layernorm.weight)
    if lt=='linear_attention':
        la=ly.linear_attn
        Wq,Wk,Wv,Wz=deinter_qkvz(la.in_proj_qkvz.weight.detach()); Wb,Wa=deinter_ba(la.in_proj_ba.weight.detach())
        w(p+"Wq.bin",Wq);w(p+"Wk.bin",Wk);w(p+"Wv.bin",Wv);w(p+"Wz.bin",Wz);w(p+"Wb.bin",Wb);w(p+"Wa.bin",Wa)
        w(p+"conv.bin", la.conv1d.weight.squeeze(1)); w(p+"A_log.bin", la.A_log); w(p+"dt_bias.bin", la.dt_bias)
        w(p+"gdn_norm.bin", la.norm.weight); w(p+"gdn_out.bin", la.out_proj.weight)
    else:
        sa=ly.self_attn
        # q_proj rows are per-head blocked [query:HD, gate:HD]; de-interleave into clean Wqq / Wqg ([NH*HD, D] each)
        Wqp = sa.q_proj.weight.detach(); Qq=[]; Qg=[]
        for hh in range(NH):
            b=hh*2*HD; Qq.append(Wqp[b:b+HD]); Qg.append(Wqp[b+HD:b+2*HD])
        w(p+"Wqq.bin", torch.cat(Qq)); w(p+"Wqg.bin", torch.cat(Qg))
        w(p+"k_proj.bin", sa.k_proj.weight); w(p+"v_proj.bin", sa.v_proj.weight); w(p+"o_proj.bin", sa.o_proj.weight)
        w(p+"q_norm.bin", sa.q_norm.weight); w(p+"k_norm.bin", sa.k_norm.weight)
    mlp=ly.mlp
    w(p+"router.bin", mlp.gate.weight)                 # [E,D]
    w(p+"egu.bin", mlp.experts.gate_up_proj)           # [E, 2*moe_inter, D]
    w(p+"edn.bin", mlp.experts.down_proj)              # [E, D, moe_inter]
    w(p+"sgate.bin", mlp.shared_expert.gate_proj.weight); w(p+"sup.bin", mlp.shared_expert.up_proj.weight); w(p+"sdown.bin", mlp.shared_expert.down_proj.weight)
    w(p+"sgate_g.bin", mlp.shared_expert_gate.weight)  # [1,D]
print(f"[ok] tiny Qwen3-Next exported: D={D} L={L} layers={cfg.layer_types} E={cfg.num_experts} topk={cfg.num_experts_per_tok} -> {OUT}")
print(f"     logits[-1,:5]={logits[-1,:5].tolist()}  gen={gen}")
