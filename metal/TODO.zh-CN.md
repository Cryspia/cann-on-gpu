# Metal 后端 —— 路线图

> English: [TODO.md](TODO.md)

Apple Metal 后端实现与 CUDA 后端*相同*的 `acl*` 运行时 + `aclnn*` 算子 + `Hccl*` 契约,因此 `../tests/`、`../tools/` 里后端无关的客户端无需改源码即可链接(`make -C ../tests BACKEND=../metal`)。

**算子覆盖已完成**——与 CUDA 后端完全对齐(导出的每个 `aclnn*` 算子),所有后端无关测试二进制通过,Qwen3-0.6B 端到端对齐 HuggingFace(logits 3.82e-6,贪心 8/8)。本文件追踪的是**剩余或可优化**的部分:受限于本机没有的硬件的条目、GPU 加速的潜力(大量长尾算子为正确性走主机端),以及集合通信的多设备互联。

> 范围与整个项目一致:**功能 / 容差级**验证 + 跨架构后端探索——不追求与昇腾逐位一致,也不与 CUDA 比性能。

目标机器:**Apple M4 Max**(40 核 GPU,128GB 统一内存,Metal 4)。

---

## A. 受限于本机没有的硬件

- **Apple M5 GPU Neural Accelerator。** M5 一代在每个 GPU 核心里加入了用于加速 GEMM 的 Neural Accelerator(矩阵/张量单元);本机是 **M4 Max**,没有该单元,故无法在此测试或使用。等有 M5 级机器时,matmul 路径(MPS GEMM 与量化 GEMM 的 decode→GEMM 尾段)是接入它的自然位置。
- **真昇腾 golden 对照 + 逐位 NZ / fractal 保真。** 昇腾专有的融合 / 量化 / 分布式 / NZ 格式算子,本地通过结构不变量、退化等价、以及**与 CUDA 后端对拍**(在一台 NVIDIA GB10 参考机上)验证。要与真昇腾 golden 再核对一遍、并确认 NZ fractal 布局逐位匹配昇腾原生布局,都需要真昇腾硬件,本机没有。这与 CUDA 后端携带的永久 out-of-scope 条目相同。

## B. GPU 加速潜力(性能,非正确性)

matmul、注意力、卷积 / 池化、elementwise、softmax、norm 等算子族都在 GPU 上跑(MPS 或 MSL)。剩余:

- **不 materialize scores 的长序列注意力。** 注意力把 QKᵀ 和 PV 走批量 MPS GEMM,会 materialize [B·H, Sq, Skv] 的 scores,因而受内存上限约束,超过上限即回退到每线程在线 softmax kernel。一个 `simdgroup_matrix` 的 FlashAttention 式 kernel 可保住 matmul 硬件的速度、且不必 materialize S²。
- **少数刻意留在主机端的算子族。** 稠密线性代数(inverse/solve/det/cholesky/qr/svd/eigh/lu)走 Accelerate/LAPACK、在统一内存上(无设备拷贝),另有个别 niche 卷积(ConvTbc、可变形卷积)仍走主机循环。这些都不是瓶颈。

## C. 多设备集合通信 / 互联

`Hccl*` 是单 rank 恒等 stub(一台 Mac = 一个设备;多 rank 归约为退化)。真正的多 Mac 路径会在下列之上实现集合通信:

- **雷电(Thunderbolt)互联**(用雷电线直接桥接两台或多台 Mac),或
- **雷电网卡提供的 RDMA**,

并置于同一份 `include/hccl/hccl.h` 契约之后——对应 CUDA 后端的 NCCL-over-RoCE 传输。`Hccl*` 的 API 与语义已经是合适的接缝;要做的是用真正的雷电 / RDMA 传输替换单 rank stub,让契约保持稳定、由后端提供其原生 fabric。
