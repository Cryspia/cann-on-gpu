# 性能基准

> English: [BENCHMARK.md](BENCHMARK.md)

cann-on-gpu CUDA 后端的性能测量方法与结果。

> 数据采集于 NVIDIA GB10（`sm_121`，统一内存 UMA，LPDDR5X ~273 GB/s），CUDA 13，cuBLASLt / cuDNN 后端。
> **这些数字是 GB10 专属、不可外推**——GB10 的低带宽 UMA 会掩盖大量访存优化的收益。换到高带宽独显（HBM/GDDR7）须整体重测，见 [`cuda/TODO.md`](cuda/TODO.zh-CN.md)。

## 价值主张

GPU 后端的意义在于：**比官方 CPU 无卡仿真快几个数量级地拿到可对拍的结果**。

- 单算子验证：无卡仿真 **40–85 s/算子**（构建 + 指令级仿真）vs GPU 后端 **µs–ms**。
- 整网前向：20 算子单层 transformer，GPU **1.67 ms/次**；无卡仿真连单个算子都要数十秒。
- 量级差 **~10⁴–10⁵×**。且指令级仿真复杂度 O(work)，大张量不可行；GPU 随硬件吞吐扩展。

| 算子 | 无卡仿真 wall-clock | 仿真器件时间 |
|---|---|---|
| Transpose（259 条指令） | 39.6 s | 1.7 µs |
| ReduceSum（363） | 45.5 s | 2.2 µs |
| Add（30899） | 84.6 s | 6.7 µs |

turnaround 由"构建 + 指令级仿真"主导并随工作量上升；器件时间本身仅 µs 级。

## 测量方法

- 工具：`tools/bench.cpp`（纯 ACL 客户端，仅链 `libascendcl.so`，模拟昇腾应用的链接方式）。
- 计时：`std::chrono` wall-clock，**含 GetWorkspaceSize 执行器构建**（即真实每次调用成本）；warmup 5–10 次后多迭代取平均；workspace 缓冲预分配复用。
- 派生指标：GEMM / Attention 报 GFLOP/s（attention 按 QKᵀ+PV ≈ 4·B·N·S²·D）；访存型报有效带宽 GB/s。
- 复现：

```bash
source env.sh
cd tools
g++ -std=c++17 -O2 -I"$ACL_INCLUDE" -I../include bench.cpp \
    -L../cuda/lib -lascendcl -Wl,-rpath,../cuda/lib -o bench
./bench
```

- decode-loop CUDA Graph 延迟：`tools/decode_graph/build_decode_graph.sh`。

## 结果

下表为优化后达到的数值，"加速"相对未优化的朴素实现。

| 算子（配置） | 结果 | 加速 | 说明 |
|---|---|---|---|
| GEMM fp16 / bf16 4096³ | ~90–94 TFLOP/s | — | cuBLASLt 近峰值 |
| Attention 默认路径 S=2048（B4 N8 D64） | 20.9 ms / 1647 GFLOP/s | 2.7× | 在线 softmax flash（免物化 S/P）+ WMMA QKᵀ tensor-core；workspace 536 MB→0 |
| └ S=1024 / 512 / 128 | 5.08 / 1.39 / 0.11 ms | 2.6–2.7× | |
| Attention 性能路径 S=2048 | 12.0 ms / 2863 GFLOP/s | — | cuBLASLt 批量 tensor-core，标准 MHA 吞吐最高（物化 S/P） |
| PagedAttention 解码 B64 N8 L4096 D64 | 4.4 ms / 122 GFLOP/s | 12.5× | warp-per-row + 寄存器 online softmax |
| RMSNorm 4096×1024 | 309 GB/s | 5.3× | 一行一 block 合并访存 + warp 归约 |
| Softmax 末维 4096×1024 | 324 GB/s | — | 合并访存快路径 |
| Add / ReduceSum 4M（fp32） | 247 / 292 GB/s | — | 已近带宽峰值 |
| tiny LLM 前向（20 算子/层） | 1.67 ms/次 | — | 端到端组合 |
| 显存 malloc+free / 对 | 0.075 µs | ~3700× | 缓存分配器（`ACL_NO_CACHE_ALLOC=1` 对照） |
| decode-loop（单 token，2 层 GQA） | 0.173 → 0.100 ms/token | 1.72× | 整循环 CUDA Graph 重放，省逐 kernel launch 开销 |

## 关于 GB10 的说明

GB10 是统一内存、低带宽、单架构 SASS、单卡。因此：

- **访存优化收益被掩盖**：如 128-bit 向量化访存、降访存流量的算子融合，在 GB10 上看不出差别，须到 HBM/GDDR7 高带宽卡才显效。
- **launch-bound 优化可测**：decode 是 launch-bound，CUDA Graph 重放的收益不被带宽掩盖，是 GB10 上少数能测出真实收益的方向。
- 所有数字换卡须重测；高带宽卡上才适合做 cann vs 原生 CUDA 的对比基准（见 `cuda/TODO.md`）。

> 所有性能优化均保持容差对拍全绿（不牺牲正确性），且保留环境变量开关可回退到未优化路径做对照
> （`ACL_NO_CACHE_ALLOC` / `ATTN_NO_PERF` / `PAGED_SCALAR` / `FLASH_NO_WMMA` / `MXFP8_NO_HW` 等）。
