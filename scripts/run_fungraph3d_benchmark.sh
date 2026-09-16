#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/run_fungraph3d_benchmark.sh [options]

Run every sequence in FunGraph3D/OpenFunGraph_split.txt at RGB stride 1, then
evaluate all 14 predictions with the HHOpenFunGraph paper protocol.

Options:
  --force             Re-run sequences even when a complete prediction exists.
  --dry-run           Check inputs and print the work without running models.
  --only SEQUENCE     Run one split sequence and skip aggregate evaluation.
  --start-at SEQUENCE Run this sequence and every later split entry, then
                      evaluate the complete split. Useful after interruption.
  --smoke-frames N    Process only the first N frames (requires --only).
  -h, --help          Show this help.

Environment overrides:
  FUNGRAPH3D_ROOT              Dataset parent containing FunGraph3D and GT files.
  OUTPUTS_ROOT                 Prediction/report root (default: repository outputs/).
  HH_OFG_EMBEDDING_PYTHON      Python used for CLIP/BERT text encoding.
  EVAL_OUTPUT                  Aggregate JSON report path.
  CUDA_VISIBLE_DEVICES         GPU exposed to each sequence run.
  MIN_GPU_FREE_MB              GPU preflight threshold used by the sequence runner.
  GPU_WAIT_INTERVAL_SECONDS    GPU polling interval, 1-60 seconds (default: 30).
  GPU_WAIT_TIMEOUT_SECONDS     Maximum wait; 0 waits indefinitely (default: 0).
EOF
}

force=0
dry_run=0
only_sequence=""
start_at_sequence=""
smoke_frames=0
while (( $# > 0 )); do
    case "$1" in
        --force)
            force=1
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --only)
            if (( $# < 2 )); then
                echo "--only requires a sequence." >&2
                exit 2
            fi
            only_sequence="$2"
            shift 2
            ;;
        --start-at)
            if (( $# < 2 )); then
                echo "--start-at requires a sequence." >&2
                exit 2
            fi
            start_at_sequence="$2"
            shift 2
            ;;
        --smoke-frames)
            if (( $# < 2 )); then
                echo "--smoke-frames requires a positive integer." >&2
                exit 2
            fi
            smoke_frames="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -n "$only_sequence" && -n "$start_at_sequence" ]]; then
    echo "--only and --start-at are mutually exclusive." >&2
    exit 2
fi
if [[ ! "$smoke_frames" =~ ^[0-9]+$ ]]; then
    echo "--smoke-frames must be a non-negative integer, got: $smoke_frames" >&2
    exit 2
fi
if (( smoke_frames > 0 )) && [[ -z "$only_sequence" ]]; then
    echo "--smoke-frames requires --only so validation output cannot be mistaken for a benchmark run." >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

benchmark_root="${FUNGRAPH3D_ROOT:-$repo_root/data/OpenFunGraph}"
scene_root="$benchmark_root/FunGraph3D"
split_path="$scene_root/OpenFunGraph_split.txt"
outputs_root="${OUTPUTS_ROOT:-$repo_root/outputs}"
runtime_python="${HH_OFG_RUNTIME_PYTHON:-$(command -v python)}"
embedding_python="${HH_OFG_EMBEDDING_PYTHON:-$runtime_python}"
eval_output="${EVAL_OUTPUT:-$outputs_root/fungraph3d_paper_eval.json}"
sequence_runner="$repo_root/run_full_sequence.sh"
evaluator="$repo_root/scripts/evaluate_fungraph3d.py"

for required_file in \
    "$split_path" \
    "$benchmark_root/annotations.json" \
    "$benchmark_root/relations.json" \
    "$benchmark_root/RootGT_Eval/all_labels.json" \
    "$benchmark_root/RootGT_Eval/all_edges.json" \
    "$benchmark_root/RootGT_Eval/all_labels_clip_embeddings.npy" \
    "$benchmark_root/RootGT_Eval/all_edges_bert_embeddings.npy" \
    "$sequence_runner" \
    "$evaluator" \
    "$runtime_python" \
    "$embedding_python"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file not found: $required_file" >&2
        exit 1
    fi
done

mapfile -t split_sequences < <(sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "$split_path")
if (( ${#split_sequences[@]} != 14 )); then
    echo "Expected 14 split sequences, got ${#split_sequences[@]}: $split_path" >&2
    exit 1
fi

declare -A seen_sequences=()
for sequence in "${split_sequences[@]}"; do
    if [[ ! "$sequence" =~ ^[[:alnum:]_-]+/[[:alnum:]_.-]+$ || "$sequence" == *".."* ]]; then
        echo "Invalid sequence in split: $sequence" >&2
        exit 1
    fi
    if [[ -n "${seen_sequences[$sequence]:-}" ]]; then
        echo "Duplicate sequence in split: $sequence" >&2
        exit 1
    fi
    seen_sequences[$sequence]=1
    sequence_dir="$scene_root/$sequence"
    for required_input in rgb depth images.txt cameras.txt; do
        if [[ ! -e "$sequence_dir/$required_input" ]]; then
            echo "Sequence input not found: $sequence_dir/$required_input" >&2
            exit 1
        fi
    done
    rgb_count="$(find "$sequence_dir/rgb" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)"
    if (( rgb_count == 0 )); then
        echo "No RGB images found in: $sequence_dir/rgb" >&2
        exit 1
    fi
done

sequences=("${split_sequences[@]}")
if [[ -n "$only_sequence" ]]; then
    if [[ -z "${seen_sequences[$only_sequence]:-}" ]]; then
        echo "Sequence '$only_sequence' is not present in $split_path" >&2
        exit 2
    fi
    sequences=("$only_sequence")
elif [[ -n "$start_at_sequence" ]]; then
    if [[ -z "${seen_sequences[$start_at_sequence]:-}" ]]; then
        echo "Sequence '$start_at_sequence' is not present in $split_path" >&2
        exit 2
    fi
    sequences=()
    include=0
    for sequence in "${split_sequences[@]}"; do
        if [[ "$sequence" == "$start_at_sequence" ]]; then
            include=1
        fi
        if (( include )); then
            sequences+=("$sequence")
        fi
    done
fi

if (( ! dry_run )) && [[ -z "${DEEPSEEK_API_KEY:-${DEEPSEEK_KEY:-${DEEPSEEK_TOKEN:-}}}" ]]; then
    echo "Set DEEPSEEK_API_KEY (or DEEPSEEK_KEY/DEEPSEEK_TOKEN) before running." >&2
    exit 1
fi

output_suffix="full"
if (( smoke_frames > 0 )); then
    output_suffix="smoke${smoke_frames}"
fi

prediction_is_complete() {
    local frontend_output="$1"
    local mapping_output="$2"
    local expected_frames="$3"
    local map_dir="$mapping_output/mapping3d/map"
    local graph_json="$map_dir/final_hierarchical_graph.json"
    local graph_ply="$map_dir/final_hierarchical_graph.ply"
    local map_nodes="$map_dir/map_nodes.json"
    local map_nodes_npz="$map_dir/map_nodes.npz"
    local frontend_summary="$frontend_output/summary.json"
    local mapping_summary="$mapping_output/mapping3d/summary.json"
    [[ -s "$graph_json" && -s "$graph_ply" && -s "$map_nodes" \
        && -s "$map_nodes_npz" && -s "$frontend_summary" \
        && -s "$mapping_summary" ]] || return 1
    "$runtime_python" - \
        "$graph_json" "$frontend_summary" "$mapping_summary" "$expected_frames" <<'PY'
import json
import sys
from pathlib import Path

try:
    graph = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    frontend = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    mapping = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
expected = int(sys.argv[4])
if not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("edges"), list):
    raise SystemExit(1)
# The evaluator compares semantic triplets.  A legacy graph with missing
# relation_text is not a complete benchmark prediction even if its PLY exists.
if any(not str(edge.get("relation_text", "")).strip()
       and edge.get("functional_status") != "pending" for edge in graph["edges"]):
    raise SystemExit(1)
frontend_counts = (
    int(frontend.get("num_input_frames", -1)),
    int(frontend.get("num_success_frames", -1)),
    int(frontend.get("num_failed_frames", -1)),
)
mapping_counts = (
    int(mapping.get("num_frames", -1)),
    int(mapping.get("num_frames_processed_this_invocation", -1)),
    int(mapping.get("num_frontend_frames", -1)),
    int(mapping.get("num_skipped_no_frontend", -1)),
)
if frontend_counts != (expected, expected, 0):
    raise SystemExit(1)
if mapping_counts != (expected, expected, expected, 0):
    raise SystemExit(1)
s2d = mapping.get("s2d", {})
if (not s2d.get("enabled") or s2d.get("partial_scoring", True)
        or int(s2d.get("available", -1)) != int(s2d.get("prefilter_pass", -2))):
    raise SystemExit(1)
PY
}

expected_run_frames() {
    local total_frames="$1"
    if (( smoke_frames > 0 && smoke_frames < total_frames )); then
        echo "$smoke_frames"
    else
        echo "$total_frames"
    fi
}

echo "FunGraph3D split: $split_path"
echo "Split sequences:  ${#split_sequences[@]}"
echo "Selected runs:    ${#sequences[@]}"
echo "RGB sampling:     every image (stride 1)"
echo "Outputs root:     $outputs_root"
if (( smoke_frames > 0 )); then
    echo "Mode:             smoke test, first $smoke_frames frames"
elif [[ -n "$only_sequence" ]]; then
    echo "Mode:             one full sequence; aggregate evaluation disabled"
elif [[ -n "$start_at_sequence" ]]; then
    echo "Mode:             resume suffix from $start_at_sequence plus aggregate evaluation"
else
    echo "Mode:             full benchmark plus aggregate evaluation"
fi

if (( dry_run )); then
    echo
    echo "Dry-run sequence plan:"
    for sequence in "${sequences[@]}"; do
        sequence_slug="${sequence//\//_}"
        rgb_count="$(find "$scene_root/$sequence/rgb" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)"
        expected_frames="$(expected_run_frames "$rgb_count")"
        echo "  $sequence: $expected_frames/$rgb_count RGB frames -> $outputs_root/${sequence_slug}_hierarchy_lifting_${output_suffix}"
    done
    if (( smoke_frames == 0 )) && [[ -z "$only_sequence" ]]; then
        echo "  evaluate ${#split_sequences[@]} predictions -> $eval_output"
    else
        echo "  aggregate evaluation skipped in subset/smoke mode"
    fi
    echo "Dry-run preflight passed."
    exit 0
fi

run_stamp="$(date +%Y%m%d_%H%M%S)"
log_dir="${RUN_LOG_DIR:-$outputs_root/fungraph3d_benchmark_logs/$run_stamp}"
mkdir -p -- "$outputs_root" "$log_dir" "$(dirname -- "$eval_output")"

completed=0
skipped=0
for sequence in "${sequences[@]}"; do
    sequence_slug="${sequence//\//_}"
    frontend_output="$outputs_root/${sequence_slug}_frontend_${output_suffix}"
    mapping_output="$outputs_root/${sequence_slug}_hierarchy_lifting_${output_suffix}"
    log_path="$log_dir/${sequence_slug}.log"
    rgb_count="$(find "$scene_root/$sequence/rgb" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)"
    expected_frames="$(expected_run_frames "$rgb_count")"

    echo
    echo "[$((completed + skipped + 1))/${#sequences[@]}] $sequence"
    if (( ! force )) && prediction_is_complete \
        "$frontend_output" "$mapping_output" "$expected_frames"; then
        echo "  reuse validated prediction: $mapping_output"
        skipped=$((skipped + 1))
        continue
    fi

    echo "  log: $log_path"
    if DATASET_ROOT="$scene_root" \
        FRONTEND_OUTPUT="$frontend_output" \
        MAPPING_OUTPUT="$mapping_output" \
        MAX_FRAMES="$smoke_frames" \
        "$sequence_runner" "$sequence" 2>&1 | tee "$log_path"; then
        :
    else
        statuses=("${PIPESTATUS[@]}")
        status=${statuses[0]}
        if (( status == 0 )); then status=${statuses[1]}; fi
        echo "Sequence failed ($status): $sequence; see $log_path" >&2
        exit "$status"
    fi
    if ! prediction_is_complete \
        "$frontend_output" "$mapping_output" "$expected_frames"; then
        echo "Sequence produced an incomplete or partial benchmark graph: $mapping_output" >&2
        exit 1
    fi
    completed=$((completed + 1))
done

echo
echo "Sequence stage finished: $completed run, $skipped reused."
if (( smoke_frames > 0 )) || [[ -n "$only_sequence" ]]; then
    echo "Aggregate evaluation skipped because this was not the complete 14-sequence run."
    exit 0
fi

# Revalidate every split prediction before invoking the strict evaluator.  This
# prevents a partial run from silently becoming a benchmark number.
for sequence in "${split_sequences[@]}"; do
    sequence_slug="${sequence//\//_}"
    frontend_output="$outputs_root/${sequence_slug}_frontend_full"
    mapping_output="$outputs_root/${sequence_slug}_hierarchy_lifting_full"
    rgb_count="$(find "$scene_root/$sequence/rgb" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)"
    if ! prediction_is_complete \
        "$frontend_output" "$mapping_output" "$rgb_count"; then
        echo "Cannot evaluate: incomplete prediction for $sequence at $mapping_output" >&2
        exit 1
    fi
done

eval_log="$log_dir/evaluation.log"
echo "Running strict 14-sequence paper-protocol evaluation..."
export HH_OFG_EMBEDDING_PYTHON="$embedding_python"
if "$runtime_python" "$evaluator" \
    --dataset-root "$benchmark_root" \
    --outputs-root "$outputs_root" \
    --embedding-python "$embedding_python" \
    --output "$eval_output" 2>&1 | tee "$eval_log"; then
    :
else
    statuses=("${PIPESTATUS[@]}")
    status=${statuses[0]}
    if (( status == 0 )); then status=${statuses[1]}; fi
    echo "Evaluation failed ($status); see $eval_log" >&2
    exit "$status"
fi

"$runtime_python" - "$eval_output" "${#split_sequences[@]}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
try:
    report = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Invalid evaluation report {path}: {exc}")
evaluated = report.get("evaluated_sequences")
missing = report.get("missing_sequences")
if not isinstance(evaluated, list) or len(evaluated) != expected:
    raise SystemExit(
        f"Evaluation postcondition failed: expected {expected} sequences, "
        f"got {len(evaluated) if isinstance(evaluated, list) else evaluated!r}"
    )
if missing:
    raise SystemExit(f"Evaluation report still has missing sequences: {missing}")
PY

echo "Benchmark completed: $eval_output"
