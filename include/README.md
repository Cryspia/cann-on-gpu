# include/ — Backend-agnostic CANN / AscendCL API contract headers

> 中文版 / Chinese: [README.zh-CN.md](README.zh-CN.md)

cann-on-gpu's self-declared **API contract**: every GPU backend implements **this same set** of headers, and clients (`../tests/`, `../tools/`) depend only on them.
They live at the top-level shared location rather than under any single backend directory — reserved for multiple GPU backends (`../cuda/`, with `../rocm/` and others to follow).

## Contents

- `aclnnop/` — aclnn operator declarations: `aclnn_ops.h` (full operator family) and `aclnn_add.h`. Self-declared following the stable signatures documented by CANN
  (the CANN distribution package used does not ship operator headers for the ops-nn component, hence the self-declaration; HCCL collective-communication declarations are handled the same way).
- `hccl/` — `hccl.h`: HCCL control-plane and collective-communication declarations (`HcclAllReduce` and friends are absent from the public distribution headers, so they are self-declared; types are reused from the distribution package's `hccl_types.h`).

## Headers not kept here

`acl/`, `aclnn/acl_meta.h` (aclTensor / aclOpExecutor meta-contract), `version`, and similar files come from the CANN distribution package; `../deploy.sh` installs them into the environment (`$ACL_INCLUDE`) and they are not tracked in this repository.
At build time both `-I$ACL_INCLUDE` and `-I./include` are passed together.

## Who implements / who uses

- **Implementors** (backends): `../cuda/` (NVIDIA CUDA). Parallel backend directories can be added later to implement the same set of headers.
- **Consumers** (clients): `../tests/` tests and `../tools/` utilities, linking the `libascendcl.so` / `libhccl.so` produced by a backend — they see only the API, just like a real Ascend application.
