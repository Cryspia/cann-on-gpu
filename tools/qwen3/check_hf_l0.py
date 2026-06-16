#!/usr/bin/env python3
# Use forward hooks to capture real outputs of each HF Qwen3 layer-0 submodule, compare against the numpy recomputation, and locate where numpy/shim diverges from HF.
# Loads the model from ModelScope by default; pass --hf (or set QWEN_USE_HF=1) to load from HuggingFace.
import numpy as np, torch, os, sys
from transformers import AutoModelForCausalLM
DAT=os.path.join(os.path.dirname(__file__),"data")
cfg=open(os.path.join(DAT,"config.txt")).read().split()
Dm,L,Hq,Hkv,hd,F,V=[int(x) for x in cfg[:7]]; eps=float(cfg[7]); S=int(cfg[9])
ids=np.fromfile(os.path.join(DAT,"ids.bin"),dtype=np.int64)
MODEL=os.environ.get("QWEN_MODEL","Qwen/Qwen3-0.6B")
USE_HF=("--hf" in sys.argv) or bool(os.environ.get("QWEN_USE_HF"))

def resolve(model_id):
    if USE_HF: return model_id
    from modelscope import snapshot_download
    return snapshot_download(model_id)

m=AutoModelForCausalLM.from_pretrained(resolve(MODEL),dtype=torch.float32).eval()
cap={}
lay=m.model.layers[0]
def hook(name):
    def f(mod,inp,out): cap[name]=(out[0] if isinstance(out,tuple) else out).detach().float().numpy()
    return f
lay.input_layernorm.register_forward_hook(hook("ln1"))
lay.self_attn.register_forward_hook(hook("attn_out"))
lay.post_attention_layernorm.register_forward_hook(hook("ln2"))
lay.mlp.register_forward_hook(hook("mlp_out"))
lay.self_attn.q_proj.register_forward_hook(hook("q"))
lay.self_attn.k_proj.register_forward_hook(hook("k"))
lay.self_attn.v_proj.register_forward_hook(hook("v"))
with torch.no_grad():
    out=m(torch.tensor(ids)[None],output_hidden_states=True)
hs=torch.stack([h[0] for h in out.hidden_states],0).numpy()
x=hs[0].astype(np.float64)

def r(n,c): return np.fromfile(os.path.join(DAT,n),dtype=np.float32).astype(np.float64).reshape(c)
def rms(x,g):
    v=(x*x).mean(-1,keepdims=True); return x/np.sqrt(v+eps)*g
def cmp(name,mine,ref):
    ref=ref.astype(np.float64).reshape(mine.shape)
    e=np.abs(mine-ref).max()/(np.abs(ref).max()+1e-9)
    print(f"  {name:9s} numpy-vs-HF err={e:.3e}{'  <<<' if e>1e-4 else ''}")

g1=r("l0.g1",(Dm,)); h=rms(x,g1); cmp("ln1",h,cap["ln1"][0])
qw=r("l0.q.w",(Hq*hd,Dm)); kw=r("l0.k.w",(Hkv*hd,Dm)); vw=r("l0.v.w",(Hkv*hd,Dm))   # no bias in Qwen3
q=h@qw.T; k=h@kw.T; v=h@vw.T
cmp("q_proj",q,cap["q"][0]); cmp("k_proj",k,cap["k"][0]); cmp("v_proj",v,cap["v"][0])
# Qwen3 per-head QK-RMSNorm (over head_dim), before RoPE
qn_w=r("l0.qn",(hd,)); kn_w=r("l0.kn",(hd,))
qn=rms(q.reshape(S,Hq,hd),qn_w).reshape(S,Hq*hd)
kn=rms(k.reshape(S,Hkv,hd),kn_w).reshape(S,Hkv*hd)
# Compare locally computed attention against HF attn_out
half=hd//2; inv=np.fromfile(os.path.join(DAT,"inv_freq.bin"),dtype=np.float32).astype(np.float64)
ang=np.arange(S)[:,None]*inv[None,:]
cos=np.concatenate([np.cos(ang)]*2,-1); sin=np.concatenate([np.sin(ang)]*2,-1)
def rope(t,H):
    t=t.reshape(S,H,hd); x1=t[...,:half]; x2=t[...,half:]; rot=np.concatenate([-x2,x1],-1)
    return (t*cos[:,None,:]+rot*sin[:,None,:]).reshape(S,H*hd)
qr=rope(qn,Hq); kr=rope(kn,Hkv); scale=1/np.sqrt(hd)
qh=qr.reshape(S,Hq,hd).transpose(1,0,2); kh=kr.reshape(S,Hkv,hd).transpose(1,0,2); vh=v.reshape(S,Hkv,hd).transpose(1,0,2)
grp=Hq//Hkv; out_=np.zeros((Hq,S,hd)); mask=np.triu(np.ones((S,S))*-1e30,1)
for hq in range(Hq):
    sc=qh[hq]@kh[hq//grp].T*scale+mask; sc-=sc.max(-1,keepdims=True); e=np.exp(sc); p=e/e.sum(-1,keepdims=True); out_[hq]=p@vh[hq//grp]
attn=out_.transpose(1,0,2).reshape(S,Hq*hd)
ow=r("l0.o.w",(Dm,Hq*hd)); op=attn@ow.T
cmp("attn_blk",op,cap["attn_out"][0])
x2=x+op; g2=r("l0.g2",(Dm,)); h2=rms(x2,g2); cmp("ln2(x2)",h2,cap["ln2"][0])
gw=r("l0.gate.w",(F,Dm)); uw=r("l0.up.w",(F,Dm)); dw=r("l0.down.w",(Dm,F))
gate=h2@gw.T; up=h2@uw.T; sg=gate/(1+np.exp(-gate)); mlp=sg*up; down=mlp@dw.T
cmp("mlp_blk",down,cap["mlp_out"][0])
