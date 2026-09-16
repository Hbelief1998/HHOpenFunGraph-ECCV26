#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/evaluate_fungraph3d.sh [options]

Evaluate existing HHOpenFunGraph predictions on the complete FunGraph3D split.
The default is strict: every split sequence must have a structurally complete
final graph before the paper-protocol evaluator is started.

Options:
  --check-only       Validate inputs and predictions without computing metrics.
  -h, --help         Show this help.

Environment overrides:
  FUNGRAPH3D_ROOT          Dataset parent containing FunGraph3D and GT files.
  OUTPUTS_ROOT             Prediction/report root (default: repository outputs/).
  EVAL_OUTPUT              JSON report path.
  EVAL_LOG                 Console log path.
  HH_OFG_RUNTIME_PYTHON    Python in the active runtime environment.
  HH_OFG_EMBEDDING_PYTHON  Python with compatible CLIP/BERT dependencies.
EOF
}

check_only=0
while (( $# > 0 )); do
    case "$1" in
        --check-only)
            check_only=1
            shift
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

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

dataset_root="${FUNGRAPH3D_ROOT:-$repo_root/data/OpenFunGraph}"
scene_root="$dataset_root/FunGraph3D"
split_path="$scene_root/OpenFunGraph_split.txt"
outputs_root="${OUTPUTS_ROOT:-$repo_root/outputs}"
runtime_python="${HH_OFG_RUNTIME_PYTHON:-$(command -v python)}"
embedding_python="${HH_OFG_EMBEDDING_PYTHON:-$runtime_python}"
evaluator="$repo_root/scripts/evaluate_fungraph3d.py"
eval_output="${EVAL_OUTPUT:-$outputs_root/fungraph3d_paper_eval.json}"
eval_log="${EVAL_LOG:-$outputs_root/fungraph3d_paper_eval.log}"

required_files=(
    "$split_path"
    "$dataset_root/annotations.json"
    "$dataset_root/relations.json"
    "$dataset_root/RootGT_Eval/all_labels.json"
    "$dataset_root/RootGT_Eval/all_edges.json"
    "$dataset_root/RootGT_Eval/all_labels_clip_embeddings.npy"
    "$dataset_root/RootGT_Eval/all_edges_bert_embeddings.npy"
    "$runtime_python"
    "$embedding_python"
    "$evaluator"
)
for path in "${required_files[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "Required file not found: $path" >&2
        exit 1
    fi
done

mapfile -t sequences < <(sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "$split_path")
if (( ${#sequences[@]} == 0 )); then
    echo "Split is empty: $split_path" >&2
    exit 1
fi

missing=()
invalid=()
for sequence in "${sequences[@]}"; do
    if [[ ! "$sequence" =~ ^[[:alnum:]_-]+/[[:alnum:]_.-]+$ || "$sequence" == *".."* ]]; then
        echo "Invalid sequence in split: $sequence" >&2
        exit 1
    fi
    sequence_slug="${sequence//\//_}"
    map_dir="$outputs_root/${sequence_slug}_hierarchy_lifting_full/mapping3d/map"
    graph_json="$map_dir/final_hierarchical_graph.json"
    graph_ply="$map_dir/final_hierarchical_graph.ply"
    map_json="$map_dir/map_nodes.json"
    map_npz="$map_dir/map_nodes.npz"
    if [[ ! -s "$graph_json" || ! -s "$graph_ply" || ! -s "$map_json" || ! -s "$map_npz" ]]; then
        missing+=("$sequence")
        continue
    fi
    if ! "$runtime_python" - "$graph_json" <<'PY'
import json
import sys
from pathlib import Path

try:
    graph = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
if not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("edges"), list):
    raise SystemExit(1)
if any(not str(edge.get("relation_text", "")).strip()
       and edge.get("functional_status") != "pending" for edge in graph["edges"]):
    raise SystemExit(1)
PY
    then
        invalid+=("$sequence")
    fi
done

echo "FunGraph3D split:      $split_path"
echo "Split sequences:       ${#sequences[@]}"
echo "Valid predictions:     $((${#sequences[@]} - ${#missing[@]} - ${#invalid[@]}))"
echo "Missing predictions:   ${#missing[@]}"
echo "Invalid predictions:   ${#invalid[@]}"
echo "Evaluation output:     $eval_output"

if (( ${#missing[@]} > 0 )); then
    printf '  missing: %s\n' "${missing[@]}"
fi
if (( ${#invalid[@]} > 0 )); then
    printf '  invalid: %s\n' "${invalid[@]}"
fi

if (( ${#missing[@]} > 0 || ${#invalid[@]} > 0 )); then
    echo "Strict evaluation refused incomplete/invalid predictions." >&2
    exit 1
fi
if (( check_only )); then
    echo "Prediction preflight passed."
    exit 0
fi

mkdir -p -- "$(dirname -- "$eval_output")" "$(dirname -- "$eval_log")"
eval_args=(
    "$runtime_python" "$evaluator"
    --dataset-root "$dataset_root"
    --outputs-root "$outputs_root"
    --embedding-python "$embedding_python"
    --output "$eval_output"
)
if "${eval_args[@]}" 2>&1 | tee "$eval_log"; then
    :
else
    statuses=("${PIPESTATUS[@]}")
    status=${statuses[0]}
    if (( status == 0 )); then status=${statuses[1]}; fi
    echo "Evaluation failed ($status); see $eval_log" >&2
    exit "$status"
fi

expected_count="${#sequences[@]}"
"$runtime_python" - "$eval_output" "$expected_count" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
expected_count = int(sys.argv[2])
try:
    report = json.loads(report_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Invalid evaluation report {report_path}: {exc}")
evaluated = report.get("evaluated_sequences")
missing = report.get("missing_sequences")
if not isinstance(evaluated, list) or len(evaluated) != expected_count:
    raise SystemExit(
        f"Evaluation postcondition failed: expected {expected_count} evaluated sequences, "
        f"got {len(evaluated) if isinstance(evaluated, list) else evaluated!r}"
    )
if missing:
    raise SystemExit(f"Strict evaluation report still has missing sequences: {missing}")
PY

echo "Evaluation completed: $eval_output"
echo "Evaluation log:       $eval_log"
