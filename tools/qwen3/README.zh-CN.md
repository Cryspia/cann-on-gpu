# Qwen3-0.6B 真实模型端到端对拍

> English: [README.md](README.md)

加载真实开源模型权重，用本项目后端的 aclnn 算子链跑完整前向，与 HuggingFace 参考 logits / 贪心续写 token 逐一对拍。纯 ACL 客户端（只链 `libascendcl.so`，同昇腾应用的链接方式）。

## 跑法

```bash
bash build_qwen3.sh                          # 默认 Qwen/Qwen3-0.6B，从 ModelScope 下载
bash build_qwen3.sh Qwen/Qwen3-0.6B "Once upon a time"
bash build_qwen3.sh --hf                     # 改从 HuggingFace 下载
```

首次会下载模型并导出权重到 `data/`；`data/` 存在则跳过（删 `data/` 强制重导）。依赖 conda env 装了 `torch` + `transformers` + `safetensors`（CPU 版即可）。ModelScope 模式还需 `modelscope` 包，缺失时构建脚本会自动安装。

## 下载来源

权重**默认从 ModelScope 下载**。加 `--hf`（或设 `QWEN_USE_HF=1`）则改从 HuggingFace 下载。两个站点用同一个 model id。

## 文件

- `export_qwen3.py` —— 解析模型（默认 ModelScope，`--hf` 走 HuggingFace），导出全权重（fp32 raw，Linear 为 native `[out,in]`）、输入 token ids、参考 logits（fp32 与 bf16 两份）、贪心续写 token、RoPE 的 `inv_freq`。
- `run_qwen3.cpp` —— 后端前向 harness：Embedding → [RMSNorm → QKV(无 bias) → 逐头 QK-RMSNorm → RoPE → GQA 因果注意力 → O+残差 → RMSNorm → SwiGlu MLP(silu(gate)·up) + 残差] × 28 → RMSNorm → tied lm_head。校验 (1) prefill 全位置 logits 与逐位 top-1，(2) 贪心续写逐 token 命中。
- `build_qwen3.sh` —— 一键导出 + 编译 + 运行。
- `check_l0.py` / `check_hf_l0.py` —— 逐层 / 逐算子的 numpy fp64 复算 + HF forward hook 对照工具，用于定位差异算子。

## 架构要点（相对 Qwen2）

Qwen3 与 Qwen2 的差异集中在注意力块：q/k/v 投影**无 bias**，每个头的 query/key 在 RoPE 前各过一次逐头 **RMSNorm**（`q_norm`/`k_norm`，对 `head_dim` 归一），且 `head_dim` 是显式的（128），而非 `hidden_size / num_heads`。MLP（SwiGLU）及其余结构不变。

## 精度模式与结果

```bash
./run_qwen3 data             # fp32：logits 误差 5.4e-6，top-1 5/5，贪心续写 8/8 与 HF-fp32 一致
QWEN_FP16=1 ./run_qwen3 data # fp16：logits 误差 5.5e-3，top-1 5/5，贪心续写 8/8 与 HF-fp32 一致
QWEN_BF16=1 ./run_qwen3 data # bf16：对 HF-bf16 参考，logits 误差 2.3e-2，top-1 5/5，贪心续写 8/8 一致
```

- fp32 逐 token 复现 HF，误差为 28 层 fp32 累积舍入级。
- bf16 与 HF 的 bf16 输出对比：本项目的 bf16 语义（张量 bf16 存储 + 算子内部 fp32 累加，即 RmsNorm `float` 累加、GEMM `CUBLAS_COMPUTE_32F`、softmax fp32）忠实于 CANN/PyTorch，故公平基准须用 HF 自己的 bf16 输出（纯 bf16 推理本身就会偏离 fp32 的贪心续写，这是 bf16 精度固有，不是实现问题）。

## 调试

`QWEN_DUMP_HIDDEN=1 ./run_qwen3 data` dump 逐层残差流；`QWEN_DUMP_L0=1 ./run_qwen3 data` dump 层 0 各算子中间张量。配合 `check_l0.py` / `check_hf_l0.py` 逐算子对拍定位差异。
