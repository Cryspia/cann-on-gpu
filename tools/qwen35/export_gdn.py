#!/usr/bin/env python3
# Export a standalone Qwen3-Next GatedDeltaNet (linear-attention) layer + a random input and the HF reference output,
# for cross-checking the cann-on-gpu shim forward (CausalConv1d + GatedDeltaRule + gated-RMSNorm + projections).
# Outputs (data_gdn/): meta.txt, hin.bin, and per-weight .bin; ref_out.bin (HF), plus staged intermediates.
import os, numpy as np, torch
from transformers import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet
torch.manual_seed(0)
OUT = os.path.join(os.path.dirname(__file__), "data_gdn"); os.makedirs(OUT, exist_ok=True)
def w(name, t): np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy()).tofile(os.path.join(OUT, name))

B, S, D = 1, 6, 32
cfg = Qwen3NextConfig(hidden_size=D, linear_num_key_heads=2, linear_key_head_dim=16,
    linear_num_value_heads=4, linear_value_head_dim=16, linear_conv_kernel_dim=4, num_hidden_layers=1, rms_norm_eps=1e-6)
gdn = Qwen3NextGatedDeltaNet(cfg, layer_idx=0).eval()
# randomize params (default init is near-zero for some; make them non-trivial)
for p in gdn.parameters(): torch.nn.init.normal_(p, mean=0.0, std=0.3)
hin = torch.randn(B, S, D)
with torch.no_grad():
    ref = gdn(hin)

nk, nv, hkd, hvd, ck = gdn.num_k_heads, gdn.num_v_heads, gdn.head_k_dim, gdn.head_v_dim, gdn.conv_kernel_size
with open(os.path.join(OUT, "meta.txt"), "w") as f:
    f.write(f"{B} {S} {D} {nk} {nv} {hkd} {hvd} {ck} {gdn.key_dim} {gdn.value_dim} {cfg.rms_norm_eps}\n")
w("hin.bin", hin)
w("ref_out.bin", ref)
w("in_proj_qkvz.bin", gdn.in_proj_qkvz.weight)   # [192,32]
w("in_proj_ba.bin", gdn.in_proj_ba.weight)        # [8,32]
# de-interleave the per-k-head-blocked in_proj rows into clean per-output split weights (simplifies the C++):
#   qkvz row block per k-head = [q:hkd, k:hkd, v:hvd*nv/nk, z:hvd*nv/nk];  ba block = [b:nv/nk, a:nv/nk]
Wqkvz = gdn.in_proj_qkvz.weight.detach(); Wba = gdn.in_proj_ba.weight.detach()
blk = 2*hkd + 2*hvd*(nv//nk); vblk = hvd*(nv//nk)
qs=[]; ks=[]; vs=[]; zs=[]
for gi in range(nk):
    base=gi*blk
    qs.append(Wqkvz[base:base+hkd]); ks.append(Wqkvz[base+hkd:base+2*hkd])
    vs.append(Wqkvz[base+2*hkd:base+2*hkd+vblk]); zs.append(Wqkvz[base+2*hkd+vblk:base+blk])
w("Wq.bin", torch.cat(qs)); w("Wk.bin", torch.cat(ks)); w("Wv.bin", torch.cat(vs)); w("Wz.bin", torch.cat(zs))
bba = nv//nk; bs=[]; as_=[]
for gi in range(nk):
    base=gi*2*bba
    bs.append(Wba[base:base+bba]); as_.append(Wba[base+bba:base+2*bba])
w("Wb.bin", torch.cat(bs)); w("Wa.bin", torch.cat(as_))
w("conv1d.bin", gdn.conv1d.weight.squeeze(1))     # [128,4]
w("A_log.bin", gdn.A_log)                          # [nv]
w("dt_bias.bin", gdn.dt_bias)                      # [nv]
w("norm.bin", gdn.norm.weight)                     # [hvd]
w("out_proj.bin", gdn.out_proj.weight)             # [32,64]

# ---- staged intermediates (recompute with the recurrent rule to mirror the shim) ----
with torch.no_grad():
    qkvz = gdn.in_proj_qkvz(hin); ba = gdn.in_proj_ba(hin)
    q,k,v,z,b,a = gdn.fix_query_key_value_ordering(qkvz, ba)
    q,k,v = (x.reshape(x.shape[0],x.shape[1],-1) for x in (q,k,v))
    mixed = torch.cat((q,k,v),-1).transpose(1,2)                       # [B,128,S]
    mixed = torch.nn.functional.silu(gdn.conv1d(mixed)[...,:S]).transpose(1,2)  # [B,S,128]
    w("conv_out.bin", mixed)
    q,k,v = torch.split(mixed,[gdn.key_dim,gdn.key_dim,gdn.value_dim],-1)
    q=q.reshape(B,S,nk,hkd); k=k.reshape(B,S,nk,hkd); v=v.reshape(B,S,nv,hvd)
    beta=b.sigmoid(); g=-gdn.A_log.float().exp()*torch.nn.functional.softplus(a.float()+gdn.dt_bias)
    q=q.repeat_interleave(nv//nk,2); k=k.repeat_interleave(nv//nk,2)
    core,_ = gdn.recurrent_gated_delta_rule(q,k,v,g=g,beta=beta,initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=True)
    w("core.bin", core)                                                # [B,S,nv,hvd] pre-norm
print(f"[ok] GDN export: B={B} S={S} D={D} nk={nk} nv={nv} hkd={hkd} hvd={hvd} conv_k={ck} -> {OUT}")
print(f"     ref_out[0,0,:4] = {ref[0,0,:4].tolist()}")
