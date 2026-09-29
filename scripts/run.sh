#!/usr/bin/env bash
# 一键构建并运行某个实验 target，例如:
#   ./scripts/run.sh smoke_vector_add
set -euo pipefail

TARGET="${1:-smoke_vector_add}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 显式指定 nvcc：PATH 里的 /usr/bin/nvcc 是 apt 装的 CUDA 11.5（不支持 sm_90a），
# 系统默认 toolkit 在 /usr/local/cuda (12.9)。注意必须用 -D 传给 cmake，
# 环境变量的方式不会被 CMake 的编译器探测读取。
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"

cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$NVCC" >/dev/null
cmake --build "$ROOT/build" --target "$TARGET" -j"$(nproc)" >/dev/null

BIN="$(find "$ROOT/build" -type f -name "$TARGET" | head -1)"
[[ -n "$BIN" ]] || { echo "找不到 target: $TARGET"; exit 1; }

# cd 到仓库根目录，保证 results/ 落在统一位置
cd "$ROOT"
exec "$BIN"
