#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <sequence>, for example: $0 3kitchen/video0" >&2
}

if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi

sequence="$1"
if [[ ! "$sequence" =~ ^[[:alnum:]_-]+/[[:alnum:]_.-]+$ || "$sequence" == *".."* ]]; then
    echo "Invalid sequence '$sequence'; expected a relative name such as 3kitchen/video0." >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

dataset_root="${DATASET_ROOT:-${FUNGRAPH3D_ROOT:-$repo_root/data/OpenFunGraph}/FunGraph3D}"
sequence_dir="$dataset_root/$sequence"
rgb_dir="$sequence_dir/rgb"
depth_dir="$sequence_dir/depth"

if [[ ! -d "$rgb_dir" ]]; then
    echo "RGB directory not found: $rgb_dir" >&2
    exit 1
fi
if [[ ! -d "$depth_dir" ]]; then
    echo "Depth directory not found: $depth_dir" >&2
    exit 1
fi
if [[ ! -f "$sequence_dir/images.txt" || ! -f "$sequence_dir/cameras.txt" ]]; then
    echo "COLMAP pose files images.txt/cameras.txt are missing under: $sequence_dir" >&2
    exit 1
fi

if [[ -z "${DEEPSEEK_API_KEY:-${DEEPSEEK_KEY:-${DEEPSEEK_TOKEN:-}}}" ]]; then
    echo "Set DEEPSEEK_API_KEY (or DEEPSEEK_KEY/DEEPSEEK_TOKEN) before running." >&2
    exit 1
fi
runtime_python="${HH_OFG_RUNTIME_PYTHON:-$(command -v python)}"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export RAMPP_REPO="${RAMPP_REPO:-$repo_root/third_party/recognize-anything}"
export SAM3_REPO="${SAM3_REPO:-$repo_root/third_party/sam3}"
export SAM3_BPE_PATH="${SAM3_BPE_PATH:-$SAM3_REPO/sam3/assets/bpe_simple_vocab_16e6.txt.gz}"

frontend_base_config="${FRONTEND_CONFIG:-$repo_root/configs/frontend2d.yaml}"
mapping_config="${MAPPING_CONFIG:-$repo_root/configs/mapping3d_paper_full.yaml}"
if [[ ! -f "$frontend_base_config" ]]; then
    echo "Frontend config not found: $frontend_base_config" >&2
    exit 1
fi
if [[ ! -f "$mapping_config" ]]; then
    echo "Mapping config not found: $mapping_config" >&2
    exit 1
fi

sequence_slug="${sequence//\//_}"
max_frames="${MAX_FRAMES:-0}"
if [[ ! "$max_frames" =~ ^[0-9]+$ ]]; then
    echo "MAX_FRAMES must be a non-negative integer, got: $max_frames" >&2
    exit 2
fi
output_suffix="full"
if (( max_frames > 0 )); then
    output_suffix="smoke${max_frames}"
fi
frontend_output="${FRONTEND_OUTPUT:-$repo_root/outputs/${sequence_slug}_frontend_${output_suffix}}"
mapping_output="${MAPPING_OUTPUT:-$repo_root/outputs/${sequence_slug}_hierarchy_lifting_${output_suffix}}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for GPU preflight but was not found." >&2
    exit 1
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    min_gpu_free_mb="${MIN_GPU_FREE_MB:-16000}"
    gpu_wait_interval_seconds="${GPU_WAIT_INTERVAL_SECONDS:-30}"
    gpu_wait_timeout_seconds="${GPU_WAIT_TIMEOUT_SECONDS:-0}"
    if [[ ! "$min_gpu_free_mb" =~ ^[0-9]+$ ]]; then
        echo "MIN_GPU_FREE_MB must be a non-negative integer, got: $min_gpu_free_mb" >&2
        exit 2
    fi
    if [[ ! "$gpu_wait_interval_seconds" =~ ^[1-9][0-9]*$ ]] || (( gpu_wait_interval_seconds > 60 )); then
        echo "GPU_WAIT_INTERVAL_SECONDS must be an integer in [1, 60], got: $gpu_wait_interval_seconds" >&2
        exit 2
    fi
    if [[ ! "$gpu_wait_timeout_seconds" =~ ^[0-9]+$ ]]; then
        echo "GPU_WAIT_TIMEOUT_SECONDS must be a non-negative integer, got: $gpu_wait_timeout_seconds" >&2
        exit 2
    fi

    gpu_wait_started_at="$SECONDS"
    while true; do
        read -r selected_gpu selected_free_mb < <(
            nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
                | awk -F, '{gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2); print $1, $2}' \
                | sort -k2,2nr \
                | sed -n '1p'
        )
        if [[ -z "${selected_gpu:-}" || ! "${selected_free_mb:-}" =~ ^[0-9]+$ ]]; then
            echo "Could not query available GPU memory with nvidia-smi." >&2
            exit 1
        fi
        if (( selected_free_mb >= min_gpu_free_mb )); then
            export CUDA_VISIBLE_DEVICES="$selected_gpu"
            echo "Auto-selected physical GPU $selected_gpu (${selected_free_mb} MiB free)."
            break
        fi

        elapsed_seconds=$((SECONDS - gpu_wait_started_at))
        if (( gpu_wait_timeout_seconds > 0 && elapsed_seconds >= gpu_wait_timeout_seconds )); then
            echo "Timed out waiting for ${min_gpu_free_mb} MiB of free GPU memory; best is GPU ${selected_gpu} with ${selected_free_mb} MiB." >&2
            nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader
            exit 1
        fi
        echo "Waiting for a GPU with at least ${min_gpu_free_mb} MiB free; best is GPU ${selected_gpu} with ${selected_free_mb} MiB (waited ${elapsed_seconds}s)."
        sleep "$gpu_wait_interval_seconds"
    done
else
    echo "Using caller-specified CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

runtime_config="$(mktemp "${TMPDIR:-/tmp}/hhofg_frontend_${sequence_slug}.XXXXXX.yaml")"
trap 'rm -f -- "$runtime_config"' EXIT
cp -- "$frontend_base_config" "$runtime_config"
sed -i "0,/^[[:space:]]*sequence:/s#^[[:space:]]*sequence:.*#  sequence: $sequence#" "$runtime_config"
sed -i "0,/^[[:space:]]*dataset_root:/s#^[[:space:]]*dataset_root:.*#  dataset_root: $dataset_root#" "$runtime_config"

rgb_count="$(find "$rgb_dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)"
echo "Sequence:        $sequence"
echo "RGB source:      $rgb_dir ($rgb_count files)"
echo "Frontend output: $frontend_output"
echo "Mapping output:  $mapping_output"
if (( max_frames > 0 )); then
    echo "Frame limit:     $max_frames (short validation mode)"
fi

frontend_args=(
    "$runtime_python" "$repo_root/scripts/run_frontend2d.py"
    --config "$runtime_config"
    --output-dir "$frontend_output"
    --overwrite
)
mapping_args=(
    "$runtime_python" "$repo_root/scripts/run_mapping3d.py"
    --config "$mapping_config"
    --frontend-run "$frontend_output"
    --output-dir "$mapping_output"
    --dataset-root "$dataset_root"
    --sequence "$sequence"
    --overwrite
)
if (( max_frames > 0 )); then
    frontend_args+=(--end "$max_frames")
    mapping_args+=(--max-frames "$max_frames")
fi

"${frontend_args[@]}"
"${mapping_args[@]}"

final_ply="$mapping_output/mapping3d/map/final_hierarchical_graph.ply"
if [[ ! -f "$final_ply" ]]; then
    echo "Run finished without the expected final PLY: $final_ply" >&2
    exit 1
fi

echo "Completed: $final_ply"
