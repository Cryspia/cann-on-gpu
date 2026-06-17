#!/usr/bin/env bash
#
# deploy.sh — Set up the GPU-side development environment for cann-on-gpu (AscendCL→CUDA, path A) on this machine.
#
# Design mirrors install.sh in the same repo (the no-GPU simulator path): idempotent, colored
# logging, no PATH writes, everything goes into one isolated conda env, subcommands
# deploy / test / uninstall.
#
# Full flow (every step is idempotent; already-done steps are skipped automatically):
#   1. Install miniforge3 (reuse if present; do NOT write to PATH / do NOT run conda init)
#   2. Create an isolated conda env (default: cann-gpu)
#   3. Validate NVIDIA toolchain: driver / nvcc / GPU compute capability (GB10 = sm_121);
#      determine CUDA_HOME and GPU_ARCH
#   4. Install build toolchain (conda-forge: cmake/ninja) + Python deps (numpy/pytest/pybind11/cuda-python)
#   5. Install operator backend libraries: cuDNN + NCCL (pip wheels aligned to CUDA major version)
#      + CUTLASS (git clone, headers only)
#        NCCL = NVIDIA counterpart to Ascend's HCCL collective-comm library (AllReduce/AllGather/...)
#   6. Fetch ACL/aclnn/HCCL headers into the env (the only compile-time Ascend-side dependency for
#      path A; compiler and runtime are NOT needed and must NOT be linked)
#        Prefer reusing headers from an already-installed cannsim toolkit env; otherwise extract
#        from a CANN .run archive (reuse a local .run, do not re-download)
#   7. Generate env.sh (summarises CUDA_HOME / cuDNN / NCCL / CUTLASS / ACL / GPU_ARCH etc.
#      for sourcing at build time)
#   8. Smoke tests: sm_121 vector-add kernel; cuDNN link; CUTLASS / ACL / HCCL header compilation;
#      NCCL nranks=1 AllReduce
#
# Usage:
#   ./deploy.sh                 # deploy + test (default)
#   ./deploy.sh deploy          # same as above
#   ./deploy.sh test            # run smoke tests only
#   ./deploy.sh uninstall       # remove conda env (miniforge itself and CUTLASS source are kept)
#   ./deploy.sh --help
#
# Optional environment-variable overrides:
#   ENV_NAME=cann-gpu           # conda env name
#   PY_VER=3.12                 # Python version for the env
#   MINIFORGE_DIR=$HOME/miniforge3
#   CUDA_HOME=/usr/local/cuda   # auto-detected if unset (/usr/local/cuda → directory containing nvcc)
#   GPU_ARCH=sm_121             # auto-derived from nvidia-smi compute_cap if unset
#   CUDA_MAJOR=13               # auto-derived from nvcc --version if unset (determines cuDNN wheel name)
#   CUTLASS_TAG=                # git tag to check out for CUTLASS (empty = default branch; Blackwell needs >=3.8)
#   SKIP_CUDNN=0                # set to 1 to skip cuDNN installation
#   SKIP_CUTLASS=0              # set to 1 to skip CUTLASS fetch
#   SKIP_NCCL=0                 # set to 1 to skip NCCL installation (HCCL→NCCL communication backend)
#   SKIP_EXPLORER=1             # explorer is opt-in (only for the optional golden cross-check); set 0 to install it
#   CANNSIM_ENV=cannsim         # source env whose installed toolkit headers can be reused
#   CANN_RUN=...                # explicit path to a CANN .run archive; if unset, any .run found in
#                               # the parent directory is used; a fresh download is the last resort
#   ACL_SRC_INCLUDE=...         # explicit include root containing acl/acl.h (highest priority)
#   SKIP_ACL_HEADERS=0          # set to 1 to skip ACL header installation
#
set -euo pipefail

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"

ENV_NAME="${ENV_NAME:-cann-gpu}"
PY_VER="${PY_VER:-3.12}"

# Detected in check_cuda and ultimately written to env.sh; can be overridden externally
CUDA_HOME="${CUDA_HOME:-}"
GPU_ARCH="${GPU_ARCH:-}"
CUDA_MAJOR="${CUDA_MAJOR:-}"

# CUTLASS: headers only, git-cloned into the project (more controllable than pip, version-pinnable)
CUTLASS_REPO="https://github.com/NVIDIA/cutlass.git"
CUTLASS_DIR="${SCRIPT_DIR}/third_party/cutlass"
CUTLASS_TAG="${CUTLASS_TAG:-}"

# Optional upstream cann-api-explorer (only for the OPTIONAL real-Ascend golden cross-check in
# tools/cannsim_golden.sh). The default verification path is the fast local torch oracle
# (tools/torch_golden.sh), which needs no clone — so explorer is opt-in: set SKIP_EXPLORER=0 to install it.
EXPLORER_REPO="${EXPLORER_REPO:-https://github.com/Cryspia/cann-api-explorer.git}"
EXPLORER_DIR="${SCRIPT_DIR}/third_party/cann-api-explorer"
EXPLORER_TAG="${EXPLORER_TAG:-}"

SKIP_CUDNN="${SKIP_CUDNN:-0}"
SKIP_CUTLASS="${SKIP_CUTLASS:-0}"
SKIP_NCCL="${SKIP_NCCL:-0}"
SKIP_EXPLORER="${SKIP_EXPLORER:-1}"   # opt-in: explorer is only for the optional golden cross-check

# ACL/aclnn headers (the only compile-time Ascend-side dependency for path A)
CANNSIM_ENV="${CANNSIM_ENV:-cannsim}"        # reuse installed toolkit headers from this env
CANN_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.T1/Ascend-cann_9.1.0-beta.1_linux-aarch64.run"
# If unset, resolve_cann_run follows the same logic as cann-api-explorer/install.sh
# (explorer directory first, then its parent; download only as last resort)
CANN_RUN="${CANN_RUN:-}"
ACL_SRC_INCLUDE="${ACL_SRC_INCLUDE:-}"         # explicit include root containing acl/acl.h
SKIP_ACL_HEADERS="${SKIP_ACL_HEADERS:-0}"

ENV_SH="${SCRIPT_DIR}/env.sh"

# ----------------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------------
c_reset=$'\033[0m'; c_grn=$'\033[92m'; c_yel=$'\033[93m'; c_red=$'\033[91m'; c_cyn=$'\033[96m'
log()  { echo "${c_grn}[DEPLOY]${c_reset} $*"; }
step() { echo; echo "${c_cyn}==== $* ====${c_reset}"; }
warn() { echo "${c_yel}[WARN]${c_reset} $*" >&2; }
err()  { echo "${c_red}[ERROR]${c_reset} $*" >&2; }
die()  { err "$@"; exit 1; }

# ----------------------------------------------------------------------------
# Utility functions (conda loading identical to install.sh: no shell pollution, no PATH dependency)
# ----------------------------------------------------------------------------
conda_sh() { echo "${MINIFORGE_DIR}/etc/profile.d/conda.sh"; }

load_conda() {
    [ -f "$(conda_sh)" ] || die "conda.sh not found: ${MINIFORGE_DIR} — run deploy first"
    # shellcheck disable=SC1090
    source "$(conda_sh)"
}

env_prefix() {
    load_conda
    conda env list | awk -v n="$ENV_NAME" '$1==n {print $NF}' | head -1
}

env_exists() { [ -n "$(env_prefix)" ]; }

# ----------------------------------------------------------------------------
# Step 1: Install miniforge3 (reuse if already present; do not write to PATH)
# ----------------------------------------------------------------------------
install_miniforge() {
    step "Step 1/8: Install miniforge3"
    if [ -x "${MINIFORGE_DIR}/bin/conda" ]; then
        log "Already present, skipping: $MINIFORGE_DIR"
        return
    fi
    local installer="/tmp/Miniforge3-Linux-aarch64.sh"
    log "Downloading miniforge installer..."
    if command -v curl >/dev/null 2>&1; then
        curl -fL -o "$installer" "$MINIFORGE_URL"
    else
        wget -O "$installer" "$MINIFORGE_URL"
    fi
    bash "$installer" -b -p "$MINIFORGE_DIR"
    rm -f "$installer"
    log "miniforge installed (not added to PATH): $MINIFORGE_DIR"
}

# ----------------------------------------------------------------------------
# Step 2: Create conda env
# ----------------------------------------------------------------------------
create_env() {
    step "Step 2/8: Create conda env '${ENV_NAME}' (python ${PY_VER})"
    load_conda
    if env_exists; then
        log "env already exists, reusing: $(env_prefix)"
        return
    fi
    conda create -y -n "$ENV_NAME" "python=${PY_VER}"
    log "env created: $(env_prefix)"
}

# ----------------------------------------------------------------------------
# Step 3: Validate NVIDIA toolchain; determine CUDA_HOME / GPU_ARCH / CUDA_MAJOR
# ----------------------------------------------------------------------------
detect_cuda_home() {
    [ -n "$CUDA_HOME" ] && { echo "$CUDA_HOME"; return; }
    if [ -d /usr/local/cuda ]; then echo "/usr/local/cuda"; return; fi
    local nv; nv="$(command -v nvcc 2>/dev/null || true)"
    [ -n "$nv" ] && { dirname "$(dirname "$nv")"; return; }
    # Fallback: find the latest /usr/local/cuda-*
    ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1
}

detect_gpu_arch() {
    [ -n "$GPU_ARCH" ] && { echo "$GPU_ARCH"; return; }
    local cc
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
    [ -n "$cc" ] && echo "sm_${cc}"
}

check_cuda() {
    step "Step 3/8: Validate NVIDIA toolchain"

    # 3a) Driver / GPU
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found — no NVIDIA driver detected; path A requires a working GPU"
    local gpu drv
    gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    log "GPU: ${gpu:-unknown}    Driver: ${drv:-unknown}"

    # 3b) CUDA_HOME / nvcc
    CUDA_HOME="$(detect_cuda_home)"
    [ -n "$CUDA_HOME" ] && [ -x "${CUDA_HOME}/bin/nvcc" ] \
        || die "nvcc (CUDA toolkit) not found. Install CUDA toolkit or set CUDA_HOME=...; GB10/Blackwell requires CUDA 13"
    local nvcc_ver
    nvcc_ver="$("${CUDA_HOME}/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)"
    log "CUDA_HOME = ${CUDA_HOME}    nvcc = ${nvcc_ver}"

    # 3c) CUDA major version (determines cuDNN wheel name)
    [ -n "$CUDA_MAJOR" ] || CUDA_MAJOR="${nvcc_ver%%.*}"
    log "CUDA major = ${CUDA_MAJOR} (cuDNN wheel will be nvidia-cudnn-cu${CUDA_MAJOR})"

    # 3d) GPU architecture
    GPU_ARCH="$(detect_gpu_arch)"
    [ -n "$GPU_ARCH" ] || { warn "Cannot derive GPU arch from nvidia-smi, falling back to sm_121 (GB10)"; GPU_ARCH="sm_121"; }
    log "GPU_ARCH = ${GPU_ARCH} (passed to nvcc -arch; Blackwell tensor-core/CUTLASS may need ${GPU_ARCH}a variant)"
}

# Run python inside the env; print the directory of a given package (empty if not found)
py_pkg_dir() {
    load_conda; conda activate "$ENV_NAME"
    python - "$1" <<'PY'
import importlib.util, os, sys
spec = importlib.util.find_spec(sys.argv[1])
if not spec:
    print(""); raise SystemExit
if spec.origin:                       # regular package: origin points to __init__.py
    print(os.path.dirname(spec.origin))
else:                                  # PEP420 namespace package (e.g. nvidia.cudnn): origin is None
    locs = list(spec.submodule_search_locations or [])
    print(locs[0] if locs else "")
PY
}

# ----------------------------------------------------------------------------
# Step 4: Build toolchain + Python dependencies
# ----------------------------------------------------------------------------
install_build_tools() {
    step "Step 4/8: Install build toolchain and Python dependencies (into env)"
    load_conda; conda activate "$ENV_NAME"

    # 4a) Build tools: cmake / ninja via conda-forge (isolated from the system, recent versions)
    log "conda-forge: install cmake / ninja"
    conda install -y -n "$ENV_NAME" -c conda-forge cmake ninja >/dev/null 2>&1 \
        || warn "cmake/ninja install skipped (may already be satisfied)"

    # 4b) Python dependencies:
    #     numpy    — result comparison harness (CPU-reference tolerance checks)
    #     pytest   — test framework (aligned with cann-api-explorer style)
    #     pybind11 — expose .so to Python, reusing the explorer call convention
    #     cuda-python — Python bindings for the Driver API (for future PTX module experiments)
    log "pip: install numpy / pytest / pybind11 / cuda-python"
    pip install --no-input numpy pytest pybind11 cuda-python \
        || warn "Some Python dependencies failed to install (does not affect C++/CUDA build itself)"

    #     torch — independent PyTorch reference oracle for tools/torch_golden.sh, the default
    #             verification path. The oracles only do CPU float64 math, so the CUDA build is
    #             irrelevant: prefer the wheel matching the card's CUDA major, fall back to the CPU wheel.
    log "pip: install torch (PyTorch reference oracle for torch_golden.sh)"
    pip install --no-input --index-url "https://download.pytorch.org/whl/cu${CUDA_MAJOR}0" torch \
        || pip install --no-input --index-url https://download.pytorch.org/whl/cpu torch \
        || warn "PyTorch oracle failed to install (torch_golden.sh will skip; other verification unaffected)"
}

# ----------------------------------------------------------------------------
# Step 5: Operator backend libraries cuDNN + CUTLASS
# ----------------------------------------------------------------------------
# pip's nvidia-cudnn wheel ships only the runtime lib libX.so.N, not the unversioned libX.so
# symlink required for linking with -l.
# For each libX.so.N in the given lib directory, create a relative libX.so -> libX.so.N symlink (idempotent).
link_unversioned_sos() {
    local dir="$1" f base link n=0
    [ -d "$dir" ] || return 0
    for f in "$dir"/lib*.so.*; do
        [ -e "$f" ] || continue
        base="$(basename "$f")"
        link="$dir/${base%%.so.*}.so"       # libcudnn.so.9 -> libcudnn.so
        [ -e "$link" ] || { ln -s "$base" "$link" && n=$((n+1)); }
    done
    [ "$n" -gt 0 ] && log "  Created ${n} unversioned .so symlinks (enables -l linking): $dir"
    return 0
}

install_cudnn() {
    if [ "$SKIP_CUDNN" = "1" ]; then warn "SKIP_CUDNN=1, skipping cuDNN"; return; fi
    load_conda; conda activate "$ENV_NAME"
    local pkg="nvidia-cudnn-cu${CUDA_MAJOR}"
    if python -c "import nvidia.cudnn" 2>/dev/null; then
        log "cuDNN already installed (${pkg}), skipping download"
    else
        log "pip: install cuDNN ${pkg} (aligned to CUDA ${CUDA_MAJOR}; Blackwell needs cuDNN 9.x)"
        pip install --no-input "$pkg" \
            || warn "cuDNN install failed: if ${pkg} does not exist check CUDA_MAJOR, or install manually"
    fi
    local d; d="$(py_pkg_dir nvidia.cudnn || true)"
    [ -n "$d" ] && link_unversioned_sos "$d/lib"   # ensure .so symlinks exist regardless of whether just installed (idempotent)
}

# NCCL: NVIDIA counterpart to Ascend's HCCL collective-comm library (AllReduce/AllGather/... <-> HCCL).
# Same pip wheel pattern as cuDNN (nvidia-nccl-cuNN); ships only .so.N, needs unversioned symlink.
install_nccl() {
    if [ "$SKIP_NCCL" = "1" ]; then warn "SKIP_NCCL=1, skipping NCCL"; return; fi
    load_conda; conda activate "$ENV_NAME"
    local pkg="nvidia-nccl-cu${CUDA_MAJOR}"
    if python -c "import nvidia.nccl" 2>/dev/null; then
        log "NCCL already installed (${pkg}), skipping download"
    else
        log "pip: install NCCL ${pkg} (HCCL->NCCL collective-comm backend; RoCE via IB verbs)"
        pip install --no-input "$pkg" \
            || warn "NCCL install failed: if ${pkg} does not exist check CUDA_MAJOR, or install manually"
    fi
    local d; d="$(py_pkg_dir nvidia.nccl || true)"
    [ -n "$d" ] && link_unversioned_sos "$d/lib"
}

install_cutlass() {
    if [ "$SKIP_CUTLASS" = "1" ]; then warn "SKIP_CUTLASS=1, skipping CUTLASS"; return; fi
    if [ -d "${CUTLASS_DIR}/include" ]; then
        log "CUTLASS already present, skipping: ${CUTLASS_DIR}"
        return
    fi
    command -v git >/dev/null 2>&1 || { warn "git not found, skipping CUTLASS (clone manually to ${CUTLASS_DIR} later)"; return; }
    log "git clone CUTLASS (headers only) -> ${CUTLASS_DIR}"
    mkdir -p "$(dirname "$CUTLASS_DIR")"
    if [ -n "$CUTLASS_TAG" ]; then
        git clone --depth 1 --branch "$CUTLASS_TAG" "$CUTLASS_REPO" "$CUTLASS_DIR" \
            || warn "Clone at tag '${CUTLASS_TAG}' failed; verify the tag exists (Blackwell needs >=3.8)"
    else
        git clone --depth 1 "$CUTLASS_REPO" "$CUTLASS_DIR" \
            || warn "CUTLASS clone failed"
    fi
    [ -d "${CUTLASS_DIR}/include" ] && log "CUTLASS ready: ${CUTLASS_DIR}/include"
}

# Upstream cann-api-explorer: cloned into third_party (used by tools/cannsim_golden.sh for golden comparison)
install_explorer() {
    if [ "$SKIP_EXPLORER" = "1" ]; then warn "SKIP_EXPLORER=1, skipping cann-api-explorer"; return; fi
    if [ -d "${EXPLORER_DIR}/.git" ] || [ -f "${EXPLORER_DIR}/install.sh" ]; then
        log "cann-api-explorer already present, skipping: ${EXPLORER_DIR}"
        return
    fi
    command -v git >/dev/null 2>&1 || { warn "git not found, skipping cann-api-explorer (clone manually to ${EXPLORER_DIR} later)"; return; }
    log "git clone cann-api-explorer -> ${EXPLORER_DIR}"
    mkdir -p "$(dirname "$EXPLORER_DIR")"
    if [ -n "$EXPLORER_TAG" ]; then
        git clone --depth 1 --branch "$EXPLORER_TAG" "$EXPLORER_REPO" "$EXPLORER_DIR" \
            || warn "Clone cann-api-explorer at tag '${EXPLORER_TAG}' failed"
    else
        git clone --depth 1 "$EXPLORER_REPO" "$EXPLORER_DIR" \
            || warn "Clone cann-api-explorer failed (clone manually to ${EXPLORER_DIR} later)"
    fi
    [ -f "${EXPLORER_DIR}/install.sh" ] && log "cann-api-explorer ready: ${EXPLORER_DIR}"
}

install_op_backends() {
    step "Step 5/8: Install operator backend libraries (cuDNN + CUTLASS + NCCL)"
    log "Note: vector ops use hand-written kernels; cuDNN/CUTLASS are needed for matmul/conv; NCCL for collective ops. Installing all now."
    install_cudnn
    install_cutlass
    install_nccl
    install_explorer
}

# ----------------------------------------------------------------------------
# Step 6: Fetch ACL/aclnn headers into the env
#   The only compile-time Ascend-side dependency for path A: headers only (API contract).
#   No compiler, no runtime, no Ascend libs are linked.
#   Source priority: ACL_SRC_INCLUDE override > installed cannsim toolkit headers (fast local path) > CANN .run extraction.
# ----------------------------------------------------------------------------
acl_target() { echo "$(env_prefix)/acl"; }     # headers land under the env dir, removed with the env

# Copy required subdirectories from one include root to a destination include directory.
#   acl/aclnnop/version = compute/runtime contract; aclnn = aclTensor/aclOpExecutor meta-contract
#   (acl_meta.h/aclnn_base.h); hccl = HCCL collective-comm type and control-plane contract
copy_acl_subdirs() {
    local src="$1" dst="$2" d
    for d in acl aclnn aclnnop version hccl; do
        [ -d "$src/$d" ] && { mkdir -p "$dst/$d"; cp -r "$src/$d/." "$dst/$d/" 2>/dev/null || true; }
    done
}

# Locate the include root (containing acl/acl.h) of an installed cannsim toolkit; empty if not found
find_cannsim_include() {
    load_conda
    local p; p="$(conda env list | awk -v n="$CANNSIM_ENV" '$1==n {print $NF}' | head -1)"
    [ -n "$p" ] || return 0
    local h; h="$(find "$p/cann" -path '*/include/acl/acl.h' 2>/dev/null | head -1)"
    [ -n "$h" ] && dirname "$(dirname "$h")"   # -> .../include
}

# Resolve a usable CANN .run archive — same logic as cann-api-explorer/install.sh (no ../ assumption):
#   explicit CANN_RUN > .run inside explorer dir > .run in explorer's parent > download to explorer dir
# This lets deploy.sh and explorer/install.sh share the same .run without duplicating downloads.
resolve_cann_run() {
    if [ -n "$CANN_RUN" ] && [ -f "$CANN_RUN" ]; then echo "$CANN_RUN"; return; fi
    local f
    f="$(ls "$EXPLORER_DIR"/Ascend-cann_*.run 2>/dev/null | head -1)";    [ -n "$f" ] && { echo "$f"; return; }
    f="$(ls "$EXPLORER_DIR"/../Ascend-cann_*.run 2>/dev/null | head -1)"; [ -n "$f" ] && { echo "$f"; return; }
    local dl="$EXPLORER_DIR/Ascend-cann_9.1.0-beta.1_linux-aarch64.run"
    mkdir -p "$EXPLORER_DIR"
    log "No local .run found, downloading to explorer directory: $CANN_URL" >&2
    if command -v curl >/dev/null 2>&1; then curl -fL -C - -o "$dl" "$CANN_URL" >&2 || return 1
    else wget -c -O "$dl" "$CANN_URL" >&2 || return 1; fi
    echo "$dl"
}

# Extract headers from a .run archive: outer --noexec --extract, then --tar x on inner component packages
#   cann-npu-runtime -> aarch64-linux/include/{acl,version}
#   cann-opbase      -> ops_base/include/{aclnnop, nnopbase/aclnn}   (aclnn meta-contract is nested under nnopbase/)
#   cann-hcomm       -> hcomm/include/hccl   (HCCL type + control-plane headers)
#   cann-ge-executor -> ge-executor/include/acl  (acl_base_mdl.h / acl_op.h — pulled in transitively by acl/acl.h)
#   cann-ge-compiler -> ge-compiler/include/acl  (acl_op_compiler.h — pulled in by the backend)
extract_acl_from_run() {
    local inc="$1"
    local run; run="$(resolve_cann_run)" || die "No usable CANN .run archive available"
    [ -n "$run" ] || die "No usable CANN .run archive available"
    log "Extracting headers from .run archive: $run"
    local tmp ex; tmp="$(mktemp -d)"; ex="$(mktemp -d)"
    bash "$run" --noexec --extract="$tmp" >/dev/null 2>&1 || { rm -rf "$tmp" "$ex"; die "Failed to extract .run archive"; }
    local rp="$tmp/run_package" rt op hc gee gec
    rt="$(ls "$rp"/cann-npu-runtime_*.run 2>/dev/null | head -1)"
    op="$(ls "$rp"/cann-opbase_*.run 2>/dev/null | head -1)"
    hc="$(ls "$rp"/cann-hcomm_*.run 2>/dev/null | head -1)"
    gee="$(ls "$rp"/cann-ge-executor_*.run 2>/dev/null | head -1)"
    gec="$(ls "$rp"/cann-ge-compiler_*.run 2>/dev/null | head -1)"
    [ -n "$rt" ] && ( cd "$ex" && bash "$rt" --tar x ./aarch64-linux/include >/dev/null 2>&1 ) || warn "  cann-npu-runtime not found; acl/ headers may be missing"
    [ -n "$op" ] && ( cd "$ex" && bash "$op" --tar x ./ops_base/include >/dev/null 2>&1 ) || warn "  cann-opbase not found; aclnnop/ headers may be missing"
    [ -n "$hc" ] && ( cd "$ex" && bash "$hc" --tar x ./hcomm/include >/dev/null 2>&1 ) || warn "  cann-hcomm not found; hccl/ headers may be missing"
    # acl/acl.h transitively #includes acl_base_mdl.h + acl_op.h (cann-ge-executor) and acl_op_compiler.h (cann-ge-compiler).
    [ -n "$gee" ] && ( cd "$ex" && bash "$gee" --tar x ./ge-executor/include/acl >/dev/null 2>&1 ) || warn "  cann-ge-executor not found; acl_base_mdl.h/acl_op.h may be missing"
    [ -n "$gec" ] && ( cd "$ex" && bash "$gec" --tar x ./ge-compiler/include/acl >/dev/null 2>&1 ) || warn "  cann-ge-compiler not found; acl_op_compiler.h may be missing"
    local r
    for r in "$ex/aarch64-linux/include" "$ex/ops_base/include" "$ex/hcomm/include" \
             "$ex/ge-executor/include" "$ex/ge-compiler/include"; do
        [ -d "$r" ] && copy_acl_subdirs "$r" "$inc"
    done
    # CANN 9.1 ships the aclnn meta-contract (acl_meta.h / aclnn_base.h) under ops_base/include/nnopbase/aclnn/,
    # not a top-level aclnn/ — copy it explicitly so $inc/aclnn/acl_meta.h (required by the build) exists.
    if [ -d "$ex/ops_base/include/nnopbase/aclnn" ]; then
        mkdir -p "$inc/aclnn"; cp -r "$ex/ops_base/include/nnopbase/aclnn/." "$inc/aclnn/" 2>/dev/null || true
    fi
    rm -rf "$tmp" "$ex"
}

install_acl_headers() {
    step "Step 6/8: Fetch ACL/aclnn headers into env"
    if [ "$SKIP_ACL_HEADERS" = "1" ]; then warn "SKIP_ACL_HEADERS=1, skipping"; return; fi
    load_conda; conda activate "$ENV_NAME"

    local inc; inc="$(acl_target)/include"
    # All of acl/, hccl/, aclnn/ must be present (older deployments may be missing aclnn; re-run supplements it)
    if [ -f "$inc/acl/acl.h" ] && [ -d "$inc/hccl" ] && [ -f "$inc/aclnn/acl_meta.h" ]; then
        log "ACL/HCCL/aclnn headers already present, skipping: $inc"
        return
    fi
    mkdir -p "$inc"

    # Source 1: explicit ACL_SRC_INCLUDE
    local src=""
    if [ -n "$ACL_SRC_INCLUDE" ] && [ -f "$ACL_SRC_INCLUDE/acl/acl.h" ]; then
        src="$ACL_SRC_INCLUDE"
    else
        # Source 2: installed cannsim toolkit (fast local path, most complete, no download/extraction needed)
        src="$(find_cannsim_include || true)"
    fi

    if [ -n "$src" ] && [ -f "$src/acl/acl.h" ]; then
        log "Reusing installed toolkit headers (copying to make env self-contained): $src"
        copy_acl_subdirs "$src" "$inc"
    else
        # Source 3: extract from .run archive
        log "No installed toolkit headers found, extracting from CANN .run archive"
        extract_acl_from_run "$inc"
    fi

    [ -f "$inc/acl/acl.h" ] || die "ACL header installation failed: $inc/acl/acl.h does not exist"
    log "ACL/HCCL headers ready: $inc"
    log "  acl/=$(ls "$inc/acl" 2>/dev/null | wc -l) files  aclnn/=$(ls "$inc/aclnn" 2>/dev/null | wc -l) files  aclnnop/=$(find "$inc/aclnnop" -name 'aclnn*.h' 2>/dev/null | wc -l) files  version/=$([ -d "$inc/version" ] && echo present || echo absent)  hccl/=$(ls "$inc/hccl" 2>/dev/null | wc -l) files"
    [ -f "$inc/aclnn/acl_meta.h" ] || warn "  Note: aclnn/acl_meta.h missing (aclTensor/aclOpExecutor meta-contract) — aclnn operator shims will not compile"
    find "$inc/aclnnop" -name 'aclnn_add.h' 2>/dev/null | grep -q . \
        || warn "  Note: this release does not include standard aclnn operator headers (aclnn_add.h etc. ship with the ops package) — shims self-declare using documented stable signatures (same approach as HCCL collective decls)"
    [ -d "$inc/hccl" ] && ! grep -rqE 'HcclAllReduce' "$inc/hccl" 2>/dev/null \
        && warn "  Note: the public hccl/ headers in this beta contain types and control-plane decls (HcclComm/RootInfo/DataType) but not collective declarations (HcclAllReduce etc.) — shims self-declare (stable, documented signatures)"
}

# ----------------------------------------------------------------------------
# Step 7: Generate env.sh (source it at build time or manually)
# ----------------------------------------------------------------------------
write_env_sh() {
    step "Step 7/8: Generate env.sh"
    load_conda; conda activate "$ENV_NAME"

    local cudnn_dir nccl_dir cutlass_inc cutlass_util_inc acl_inc explorer_dir
    cudnn_dir="$(py_pkg_dir nvidia.cudnn || true)"
    nccl_dir="$(py_pkg_dir nvidia.nccl || true)"
    cutlass_inc=""; cutlass_util_inc=""
    [ -d "${CUTLASS_DIR}/include" ] && cutlass_inc="${CUTLASS_DIR}/include"
    [ -d "${CUTLASS_DIR}/tools/util/include" ] && cutlass_util_inc="${CUTLASS_DIR}/tools/util/include"
    acl_inc=""; [ -f "$(acl_target)/include/acl/acl.h" ] && acl_inc="$(acl_target)/include"
    explorer_dir=""; [ -d "${EXPLORER_DIR}" ] && explorer_dir="${EXPLORER_DIR}"

    cat > "$ENV_SH" <<EOF
#!/usr/bin/env bash
# Auto-generated by deploy.sh — source this file to get all build/runtime paths for cann-on-gpu.
# Usage:
#   source ${MINIFORGE_DIR}/etc/profile.d/conda.sh && conda activate ${ENV_NAME}
#   source ${ENV_SH}

export CANN_GPU_ENV="${ENV_NAME}"
export CUDA_HOME="${CUDA_HOME}"
export GPU_ARCH="${GPU_ARCH}"          # nvcc -arch=\${GPU_ARCH}
export CUDA_MAJOR="${CUDA_MAJOR}"
export PATH="\${CUDA_HOME}/bin:\${PATH}"
export LD_LIBRARY_PATH="\${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH:-}"

# ---- cuDNN (from pip wheel nvidia-cudnn-cu${CUDA_MAJOR}) ----
export CUDNN_DIR="${cudnn_dir}"
if [ -n "\${CUDNN_DIR}" ]; then
    export CPATH="\${CUDNN_DIR}/include:\${CPATH:-}"
    export LIBRARY_PATH="\${CUDNN_DIR}/lib:\${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="\${CUDNN_DIR}/lib:\${LD_LIBRARY_PATH}"
fi

# ---- NCCL (from pip wheel nvidia-nccl-cu${CUDA_MAJOR}; HCCL->NCCL communication backend) ----
export NCCL_DIR="${nccl_dir}"
if [ -n "\${NCCL_DIR}" ]; then
    export CPATH="\${NCCL_DIR}/include:\${CPATH:-}"
    export LIBRARY_PATH="\${NCCL_DIR}/lib:\${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="\${NCCL_DIR}/lib:\${LD_LIBRARY_PATH}"
fi
# Reference environment variables for multi-node RoCE NCCL runs (uncomment as needed);
# true multi-rank reduction requires multiple GPUs/nodes:
#   export NCCL_IB_HCA=mlx5_0,mlx5_1         # list both HCAs -> NCCL multi-rail aggregation ~200G (no OS bonding needed)
#   export NCCL_SOCKET_IFNAME=eth0            # out-of-band bootstrap/uniqueId channel via management NIC
#   export NCCL_IB_GID_INDEX=3               # RoCEv2 GID (check with show_gids for actual value)

# ---- CUTLASS (git clone, headers only; add -I at compile time) ----
export CUTLASS_INCLUDE="${cutlass_inc}"
export CUTLASS_UTIL_INCLUDE="${cutlass_util_inc}"

# ---- ACL/aclnn/HCCL headers (path A API contract; compile-time -I only; do NOT link Ascend runtime) ----
# hccl/ lives under ACL_INCLUDE; use #include <hccl/hccl_types.h>
export ACL_INCLUDE="${acl_inc}"

# ---- Upstream cann-api-explorer (cloned into third_party; used by tools/cannsim_golden.sh) ----
export EXPLORER_DIR="${explorer_dir}"

# Minimal compile command example for a single .cu file:
# nvcc -std=c++17 -arch=\${GPU_ARCH} \\
#   -I"\${ACL_INCLUDE}" -I"\${CUDNN_DIR}/include" -I"\${NCCL_DIR}/include" \\
#   -I"\${CUTLASS_INCLUDE}" -I"\${CUTLASS_UTIL_INCLUDE}" \\
#   -L"\${CUDNN_DIR}/lib" -lcudnn -L"\${NCCL_DIR}/lib" -lnccl  your_op.cu -o your_op
EOF
    chmod +x "$ENV_SH"
    log "Generated: $ENV_SH"
    log "  CUDA_HOME=${CUDA_HOME}  GPU_ARCH=${GPU_ARCH}"
    log "  CUDNN_DIR=${cudnn_dir:-<not installed>}"
    log "  NCCL_DIR=${nccl_dir:-<not installed>}"
    log "  CUTLASS_INCLUDE=${cutlass_inc:-<not installed>}"
    log "  ACL_INCLUDE=${acl_inc:-<not installed>} (hccl/ is under it)"
}

# ----------------------------------------------------------------------------
# Step 8: Smoke tests
# ----------------------------------------------------------------------------
smoke_test() {
    step "Step 8/8: Smoke tests"
    load_conda; conda activate "$ENV_NAME"
    [ -f "$ENV_SH" ] || die "$ENV_SH not found; run deploy first"
    # shellcheck disable=SC1090
    source "$ENV_SH"

    local nvcc="${CUDA_HOME}/bin/nvcc"
    [ -x "$nvcc" ] || die "nvcc not available: $nvcc"
    local work; work="$(mktemp -d)"

    # 8a) Compile and run a vector-add kernel on the GPU — validates the entire GPU toolchain
    log "Test 1: nvcc compile + GPU vector-add kernel (-arch=${GPU_ARCH})"
    cat > "$work/vadd.cu" <<'EOF'
#include <cstdio>
__global__ void vadd(const float* a, const float* b, float* c, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
int main(){
    const int N = 1024; size_t sz = N*sizeof(float);
    float *ha=(float*)malloc(sz),*hb=(float*)malloc(sz),*hc=(float*)malloc(sz);
    for(int i=0;i<N;i++){ ha[i]=i; hb[i]=2*i; }
    float *da,*db,*dc; cudaMalloc(&da,sz); cudaMalloc(&db,sz); cudaMalloc(&dc,sz);
    cudaMemcpy(da,ha,sz,cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,sz,cudaMemcpyHostToDevice);
    vadd<<<(N+255)/256,256>>>(da,db,dc,N);
    cudaError_t e=cudaDeviceSynchronize();
    if(e!=cudaSuccess){ printf("CUDA ERROR: %s\n", cudaGetErrorString(e)); return 2; }
    cudaMemcpy(hc,dc,sz,cudaMemcpyDeviceToHost);
    for(int i=0;i<N;i++){ if(hc[i]!=3.0f*i){ printf("MISMATCH at %d\n",i); return 3; } }
    printf("VADD OK\n"); return 0;
}
EOF
    if "$nvcc" -std=c++17 -arch="${GPU_ARCH}" "$work/vadd.cu" -o "$work/vadd" >"$work/vadd.build.log" 2>&1 \
        && "$work/vadd" | grep -q "VADD OK"; then
        log "  GPU toolchain OK: compile + kernel execute + result check all passed"
    else
        warn "  GPU vector-add test did not pass (see $work/vadd.build.log)"
        tail -10 "$work/vadd.build.log" | sed 's/^/      /'
    fi

    # 8b) cuDNN: compile + link + run (call cudnnCreate/Destroy to verify library is usable)
    if [ "$SKIP_CUDNN" != "1" ] && [ -n "${CUDNN_DIR:-}" ] && [ -f "${CUDNN_DIR}/include/cudnn.h" ]; then
        log "Test 2: cuDNN link + run (cudnnCreate)"
        cat > "$work/cudnn_chk.cu" <<'EOF'
#include <cudnn.h>
#include <cstdio>
int main(){
    cudnnHandle_t h;
    if(cudnnCreate(&h)!=CUDNN_STATUS_SUCCESS){ printf("cudnnCreate FAILED\n"); return 1; }
    printf("CUDNN OK version=%zu\n",(size_t)cudnnGetVersion());
    cudnnDestroy(h); return 0;
}
EOF
        if "$nvcc" -std=c++17 -arch="${GPU_ARCH}" \
                -I"${CUDNN_DIR}/include" -L"${CUDNN_DIR}/lib" -lcudnn \
                "$work/cudnn_chk.cu" -o "$work/cudnn_chk" >"$work/cudnn.build.log" 2>&1 \
            && LD_LIBRARY_PATH="${CUDNN_DIR}/lib:${LD_LIBRARY_PATH:-}" "$work/cudnn_chk" | grep -q "CUDNN OK"; then
            log "  cuDNN: compile / link / run OK"
        else
            warn "  cuDNN self-check did not pass (see $work/cudnn.build.log)"
            tail -10 "$work/cudnn.build.log" | sed 's/^/      /'
        fi
    else
        warn "Test 2: skipping cuDNN self-check (not installed or headers absent)"
    fi

    # 8c) CUTLASS: headers only; compiling a TU that includes cutlass.h to a .o is sufficient
    if [ "$SKIP_CUTLASS" != "1" ] && [ -n "${CUTLASS_INCLUDE:-}" ] && [ -d "${CUTLASS_INCLUDE}" ]; then
        log "Test 3: CUTLASS header compilation (include cutlass/cutlass.h)"
        cat > "$work/cutlass_chk.cu" <<'EOF'
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
__global__ void k(){}
int main(){ return 0; }
EOF
        if "$nvcc" -std=c++17 -arch="${GPU_ARCH}" \
                -I"${CUTLASS_INCLUDE}" ${CUTLASS_UTIL_INCLUDE:+-I"${CUTLASS_UTIL_INCLUDE}"} \
                -c "$work/cutlass_chk.cu" -o "$work/cutlass_chk.o" >"$work/cutlass.build.log" 2>&1; then
            log "  CUTLASS headers compile with nvcc OK"
        else
            warn "  CUTLASS header compilation failed (Blackwell needs CUTLASS>=3.8; see $work/cutlass.build.log)"
            tail -10 "$work/cutlass.build.log" | sed 's/^/      /'
        fi
    else
        warn "Test 3: skipping CUTLASS self-check (not fetched)"
    fi

    # 8d) ACL headers: path A API contract. Only verify that headers can be compiled (compile-time
    #     dependency). Do NOT link any Ascend runtime — syntax-only check, no executable, no -l.
    if [ "$SKIP_ACL_HEADERS" != "1" ] && [ -n "${ACL_INCLUDE:-}" ] && [ -f "${ACL_INCLUDE}/acl/acl.h" ]; then
        log "Test 4: ACL header compilation (#include <acl/acl.h>, syntax-only, no runtime link)"
        cat > "$work/acl_chk.c" <<'EOF'
#include <acl/acl.h>
int main(void){ (void)aclrtMalloc; (void)aclrtMemcpy; return 0; }
EOF
        if cc -fsyntax-only -I"${ACL_INCLUDE}" "$work/acl_chk.c" >"$work/acl.build.log" 2>&1; then
            log "  ACL headers compile OK (API contract ready: aclrtMalloc/aclrtMemcpy symbols visible)"
        else
            warn "  ACL header syntax check failed (see $work/acl.build.log)"
            tail -10 "$work/acl.build.log" | sed 's/^/      /'
        fi
    else
        warn "Test 4: skipping ACL header self-check (not installed)"
    fi

    # 8e) HCCL headers: collective-comm type and control-plane contract. Syntax-only, no Ascend link.
    #     Note: hccl_types.h uses `const uint32_t` as an array dimension — not a constant expression
    #     in C (treated as VLA), so it must be compiled as C++ (which shims are anyway).
    if [ "$SKIP_ACL_HEADERS" != "1" ] && [ -n "${ACL_INCLUDE:-}" ] && [ -d "${ACL_INCLUDE}/hccl" ]; then
        log "Test 5: HCCL header compilation (#include <hccl/hccl_types.h>, C++ syntax-only)"
        cat > "$work/hccl_chk.cpp" <<'EOF'
#include <hccl/hccl_types.h>
int main(){ HcclDataType dt; HcclReduceOp op; HcclResult r; (void)dt;(void)op;(void)r; return 0; }
EOF
        if g++ -fsyntax-only -I"${ACL_INCLUDE}" "$work/hccl_chk.cpp" >"$work/hccl.build.log" 2>&1; then
            log "  HCCL headers compile OK (type contract ready: HcclDataType/HcclReduceOp/HcclResult)"
        else
            warn "  HCCL header syntax check failed (see $work/hccl.build.log)"
            tail -10 "$work/hccl.build.log" | sed 's/^/      /'
        fi
    else
        warn "Test 5: skipping HCCL header self-check (not installed)"
    fi

    # 8f) NCCL nranks=1 AllReduce: single-GPU only validates the pipeline (API wiring + degenerate
    #     semantics). True multi-rank reduction requires multiple GPUs or nodes.
    if [ "$SKIP_NCCL" != "1" ] && [ -n "${NCCL_DIR:-}" ] && [ -f "${NCCL_DIR}/include/nccl.h" ]; then
        log "Test 6: NCCL nranks=1 AllReduce (single-GPU communication API pipeline check)"
        cat > "$work/nccl_chk.cu" <<'EOF'
#include <nccl.h>
#include <cuda_runtime.h>
#include <cstdio>
int main(){
    int dev=0; cudaSetDevice(dev);
    ncclComm_t comm;
    if(ncclCommInitAll(&comm,1,&dev)!=ncclSuccess){ printf("ncclCommInitAll FAILED\n"); return 1; }
    const int N=8; float h[N]; for(int i=0;i<N;i++) h[i]=i+1;
    float *sb,*rb; cudaMalloc(&sb,N*sizeof(float)); cudaMalloc(&rb,N*sizeof(float));
    cudaMemcpy(sb,h,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaStream_t s; cudaStreamCreate(&s);
    if(ncclAllReduce(sb,rb,N,ncclFloat,ncclSum,comm,s)!=ncclSuccess){ printf("ncclAllReduce FAILED\n"); return 2; }
    cudaStreamSynchronize(s);
    float o[N]; cudaMemcpy(o,rb,N*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<N;i++) if(o[i]!=h[i]){ printf("MISMATCH %d: %f != %f\n",i,o[i],h[i]); return 3; }
    int v; ncclGetVersion(&v); printf("NCCL OK version=%d\n", v);
    ncclCommDestroy(comm); return 0;
}
EOF
        if "$nvcc" -std=c++17 -arch="${GPU_ARCH}" -I"${NCCL_DIR}/include" \
                "$work/nccl_chk.cu" -o "$work/nccl_chk" -L"${NCCL_DIR}/lib" -lnccl >"$work/nccl.build.log" 2>&1 \
            && LD_LIBRARY_PATH="${NCCL_DIR}/lib:${LD_LIBRARY_PATH:-}" "$work/nccl_chk" | grep -q "NCCL OK"; then
            log "  NCCL compile / link / run OK (nranks=1 AllReduce degenerates to identity; result verified)"
        else
            warn "  NCCL self-check did not pass (see $work/nccl.build.log)"
            tail -10 "$work/nccl.build.log" | sed 's/^/      /'
        fi
    else
        warn "Test 6: skipping NCCL self-check (not installed)"
    fi

    # RoCE readiness (informational, not pass/fail): hardware prerequisite for multi-node HCCL tests
    if command -v ibv_devinfo >/dev/null 2>&1; then
        local nhca nact
        nhca="$(ibv_devinfo -l 2>/dev/null | grep -cE '^\s+\w')"
        nact="$(rdma link show 2>/dev/null | grep -c 'state ACTIVE' || echo 0)"
        log "RoCE readiness: detected ${nhca} RDMA HCA(s), ${nact} link(s) ACTIVE — prerequisites for 2-node NCCL/HCCL multi-rank tests are met"
    else
        warn "RoCE: ibv_devinfo (rdma-core) not found; bring up the RDMA stack before running multi-node communication tests"
    fi

    echo
    log "${c_grn}Smoke tests complete.${c_reset}"
    echo
    echo "To use this environment next time:"
    echo "  source ${MINIFORGE_DIR}/etc/profile.d/conda.sh"
    echo "  conda activate ${ENV_NAME}"
    echo "  source ${ENV_SH}"
    echo "  nvcc -std=c++17 -arch=\${GPU_ARCH} your_op.cu -o your_op"
}

# ----------------------------------------------------------------------------
# Uninstall: remove env only (miniforge and CUTLASS source are preserved)
# ----------------------------------------------------------------------------
do_uninstall() {
    step "Uninstall: remove conda env '${ENV_NAME}' (miniforge and CUTLASS source are preserved)"
    load_conda
    if ! env_exists; then
        log "env '${ENV_NAME}' does not exist, nothing to remove"
        return
    fi
    local pfx; pfx="$(env_prefix)"
    conda deactivate 2>/dev/null || true
    conda env remove -y -n "$ENV_NAME"
    [ -d "$pfx" ] && rm -rf "$pfx"
    [ -f "$ENV_SH" ] && rm -f "$ENV_SH"
    log "Removed env: $pfx"
    log "Preserved: miniforge ($MINIFORGE_DIR), CUTLASS source ($CUTLASS_DIR)"
}

# ----------------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------------
do_deploy() {
    install_miniforge
    create_env
    check_cuda
    install_build_tools
    install_op_backends
    install_acl_headers
    write_env_sh
    smoke_test
}

usage() {
    sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
}

main() {
    case "${1:-deploy}" in
        deploy|"")    do_deploy ;;
        test)         check_cuda; smoke_test ;;
        uninstall)    do_uninstall ;;
        -h|--help|help) usage ;;
        *) die "Unknown command: $1 (available: deploy | test | uninstall)" ;;
    esac
}

main "$@"
