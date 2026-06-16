# Qwen3-Next（“Qwen3.5”）混合模型适配

> English: [README.md](README.md)

端到端验证 cann-on-gpu shim 能够跑通 **Qwen3-Next** 前向 —— 这是混合线性注意力 / SSM 模型
（HF 类 `Qwen3NextForCausalLM`，`model_type=qwen3_next`），此前因为需要门控 delta、因果卷积、门控
RMSNorm 等算子而无法复现。

由于不存在小规模的 Qwen3-Next 预训练权重（仅发布了 80B-A3B），本测试用一个随机初始化的小模型对
HuggingFace 验证**架构映射**：从 HF dump 出同一份随机权重，shim 用 `aclnn*` 算子重算前向，逐阶段按
归一化误差比对。

## 覆盖内容

- **GatedDeltaNet 线性注意力层**（`run_gdn`）：in-proj → 因果深度可分 `aclnnCausalConv1d`+SiLU →
  `beta=sigmoid(b)`、`g=-exp(A_log)·softplus(a+dt_bias)` → 头复制 → L2 归一 + 缩放后的
  `aclnnGatedDeltaRule`（与 HF `chunk_gated_delta_rule` 一致）→ 门控 RMSNorm（`RmsNorm`·`Silu`·`Mul`）→ out-proj。
- **完整小模型**（`run_qwen35`）：词嵌入 → 每层 { 零中心 `aclnnRmsNorm`(·(1+w)) → GatedDeltaNet 或门控全注意力
  （输出门、逐头 QK-norm、对旋转切片用 `aclnnApplyRotaryPosEmb` 做 partial RoPE、GQA `aclnnFlashAttentionScore`）
  → 残差 → RMSNorm → 稀疏 MoE（`aclnnMoeGatingTopKSoftmax` 路由 + 逐专家 SwiGLU + sigmoid 门控共享专家）→ 残差 }
  → 末层归一 → lm_head。

## 运行

```bash
source ../../env.sh
./build_qwen35.sh        # 导出 HF 参考、编译、运行两个检查
```

需要 `cann-gpu` conda 环境，含 `torch` + `transformers>=5.12`（纯 CPU 即可）。

## 结果

所有阶段与 HF 参考一致：GatedDeltaNet 层约 8e-7，完整模型的逐层隐藏状态、末层归一输出，以及 **logits 约 4e-6**，
贪心下一个 token 完全一致。这证明算子集（P18 的 SSM 算子、P15 的 MoE 路由、P7 的门控归一、partial RoPE）
能够组合出正确的 Qwen3-Next 前向。
