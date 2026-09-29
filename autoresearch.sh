#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Autoresearch benchmark entrypoint for vLLM Metal.
#
# Runs a fixed, offline suite of text-generation workloads through the
# in-process V1 engine and prints one `METRIC <name>=<value>` line per metric
# (see tools/benchmark/metal_suite_benchmark.py). Deterministic: fixed prompts,
# sampling params, seeds, model checkpoints from the local HF cache only.
#
# Exit status: 0 when every workload produced metrics, non-zero otherwise.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VENV="${VLLM_METAL_BENCH_VENV:-$HOME/.venv-metal-bench}"
PYTHON="$VENV/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Benchmark interpreter not found: $PYTHON" >&2
  echo "Set VLLM_METAL_BENCH_VENV to the venv holding vLLM + mlx." >&2
  exit 2
fi

# No prebuilt .metallibs on a source checkout: compile the extension and the
# shader sources in-process (cached for subsequent runs).
export VLLM_METAL_BUILD_FROM_SOURCE="${VLLM_METAL_BUILD_FROM_SOURCE:-1}"
# Single-process engine so memory sampling sees the whole workload.
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# Never touch the network: checkpoints must already be cached.
export HF_HUB_OFFLINE=1
# Keep FP32 opens deterministic across runs.
export MLX_ENABLE_TF32=0

exec "$PYTHON" tools/benchmark/metal_suite_benchmark.py "$@"
