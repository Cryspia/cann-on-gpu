# include/ —— 后端无关的 CANN / AscendCL API 契约头

> English: [README.md](README.md)

cann-on-gpu 自声明的 **API 契约**：任何 GPU 后端都实现**这同一套**头，客户端（`../tests/`、`../tools/`）只依赖它。
故放在顶层共享，而非某个后端目录下——为多 GPU 后端（`../cuda/`，将来可加 `../rocm/` 等）预留。

## 内容

- `aclnnop/` —— aclnn 算子声明：`aclnn_ops.h`（全算子族）、`aclnn_add.h`。按 CANN 文档化的稳定签名自声明
  （所用 CANN 发行包不含 ops-nn 组件的算子头，故自声明；HCCL 集合通信声明同此处理）。
- `hccl/` —— `hccl.h`：HCCL 控制面 + 集合通信声明（公共发行头未含 `HcclAllReduce` 等，自声明；类型复用发行包的 `hccl_types.h`）。

## 不在这里的头

`acl/`、`aclnn/acl_meta.h`（aclTensor / aclOpExecutor 元契约）、`version` 等来自 CANN 发行包，由 `../deploy.sh` 取进环境（`$ACL_INCLUDE`），不随本仓库。
构建时 `-I$ACL_INCLUDE -I./include` 并存。

## 谁实现 / 谁使用

- **实现方**（后端）：`../cuda/`（NVIDIA CUDA）。将来可加平行后端目录，实现同一套头。
- **使用方**（客户端）：`../tests/` 测试、`../tools/` 工具，链后端产出的 `libascendcl.so` / `libhccl.so`，如昇腾应用般只见 API。
