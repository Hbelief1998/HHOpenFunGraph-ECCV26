#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runner="$repo_root/run_fungraph3d_benchmark.sh"

usage() {
    cat <<'EOF'
Usage:
  ./run_fungraph3d_background.sh --dry-run
  ./run_fungraph3d_background.sh
  ./run_fungraph3d_background.sh --status RUN_DIRECTORY

Start a fresh 14-sequence, full-frame frontend + mapping + evaluation job.
nohup + setsid detach it from the terminal/SSH connection. Server shutdown,
reboot or explicit process termination will still stop the job.

Required for launch: DEEPSEEK_API_KEY (or DEEPSEEK_KEY / DEEPSEEK_TOKEN).
Optional: CUDA_VISIBLE_DEVICES; FUNGRAPH3D_ROOT; FRONTEND_CONFIG;
MAPPING_CONFIG; RUNS_ROOT (default outputs/fungraph3d_runs).
Without CUDA_VISIBLE_DEVICES, the existing runner waits for a free GPU.
Each invocation creates a new output directory; previous runs are preserved.
The evaluation reports node and directed functional-triplet recall.
EOF
}

# Internal detached worker. Its PID is also the process-group/session ID.
if [[ "${1:-}" == "--worker" ]]; then
    run_dir="$2"
    cd "$repo_root"
    umask 077
    finish() {
        local result=$?
        trap - EXIT
        if (( result == 0 )); then
            printf 'SUCCEEDED\n' > "$run_dir/status"
        else
            printf 'FAILED exit=%s\n' "$result" > "$run_dir/status"
        fi
        date -Is > "$run_dir/finished_at"
        # Publish the completion marker last, after all status fields.
        printf '%s\n' "$result" > "$run_dir/exit_code.tmp"
        mv -- "$run_dir/exit_code.tmp" "$run_dir/exit_code"
    }
    trap finish EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
    printf '%s\n' "$$" > "$run_dir/worker.pid"
    printf 'RUNNING\n' > "$run_dir/status"
    date -Is > "$run_dir/started_at"
    export OUTPUTS_ROOT="$run_dir/predictions"
    export RUN_LOG_DIR="$run_dir/logs"
    export EVAL_OUTPUT="$run_dir/fungraph3d_eval.json"
    export FRONTEND_CONFIG="$run_dir/frontend2d.yaml"
    export MAPPING_CONFIG="$run_dir/mapping3d.yaml"
    export PYTHONUNBUFFERED=1
    "$runner" --force
    exit 0
fi

if [[ "${1:-}" == "--status" && $# == 2 ]]; then
    run_dir="$2"
    [[ -d "$run_dir" ]] || { echo "Run directory not found: $run_dir" >&2; exit 1; }
    cat "$run_dir/status"
    if [[ -f "$run_dir/worker.pid" ]]; then
        pid="$(cat "$run_dir/worker.pid")"
        echo "Worker PID / process group: $pid"
        if [[ ! -f "$run_dir/exit_code" ]] && ! kill -0 "$pid" 2>/dev/null; then
            echo "Worker is no longer running; inspect master.log (no exit record)."
        fi
    fi
    [[ ! -f "$run_dir/exit_code" ]] || echo "Exit code: $(cat "$run_dir/exit_code")"
    echo "Log: $run_dir/master.log"
    exit 0
fi

dry_run=0
case "${1:-}" in
    --dry-run) dry_run=1 ;;
    -h|--help) usage; exit 0 ;;
    '') ;;
    *) usage >&2; exit 2 ;;
esac
(( $# <= 1 )) || { usage >&2; exit 2; }
runtime_python="${HH_OFG_RUNTIME_PYTHON:-$(command -v python)}"
cd "$repo_root"
for command in nohup setsid; do
    command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 1; }
done
frontend_config="${FRONTEND_CONFIG:-$repo_root/configs/frontend2d.yaml}"
mapping_config="${MAPPING_CONFIG:-$repo_root/configs/mapping3d_paper_full.yaml}"
# Validate the actual launch configuration, not only the dataset paths.
"$runtime_python" - "$frontend_config" "$mapping_config" "$repo_root" <<'PY'
import json
import os
import sys
from pathlib import Path
import yaml
front, mapping = [yaml.safe_load(os.path.expandvars(Path(p).read_text())) for p in sys.argv[1:3]]
for name, cfg in [('frontend', front), ('mapping', mapping)]:
    data = cfg['data']
    if (data.get('start', 0), data.get('end', -1), data.get('stride', 1)) != (0, -1, 1):
        raise SystemExit(f'{name}: a full run requires start=0, end=-1, stride=1')
edge = mapping['edge2d']
if not edge.get('enabled') or any(edge.get(k, 0) for k in ('max_edges_per_frame', 'max_edges_per_run')):
    raise SystemExit('Full benchmark requires enabled, uncapped S2D scoring')
if not mapping.get('output', {}).get('save_ply', True):
    raise SystemExit('Full benchmark requires output.save_ply=true')
required = [front['rampp']['checkpoint'], front['sam3']['checkpoint'],
            mapping['clip']['checkpoint_path'], mapping['edge2d']['llava']['model_path']]
for name in required:
    if not Path(name).exists():
        raise SystemExit(f'Model path not found: {name}')
print('Full-run config / models: PASS')
PY
runs_root="${RUNS_ROOT:-$repo_root/outputs/fungraph3d_runs}"
OUTPUTS_ROOT="$runs_root/<new-run>/predictions" \
    EVAL_OUTPUT="$runs_root/<new-run>/fungraph3d_eval.json" "$runner" --force --dry-run
if (( dry_run )); then
    exit 0
fi
if [[ -z "${DEEPSEEK_API_KEY:-${DEEPSEEK_KEY:-${DEEPSEEK_TOKEN:-}}}" ]]; then
    echo "Set DEEPSEEK_API_KEY before launching." >&2
    exit 1
fi
umask 077
mkdir -p -- "$runs_root"
runs_root="$(cd -- "$runs_root" && pwd)"
run_dir="$(mktemp -d "$runs_root/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
cp -- "$frontend_config" "$run_dir/frontend2d.yaml"
cp -- "$mapping_config" "$run_dir/mapping3d.yaml"
printf 'STARTING\n' > "$run_dir/status"
ln -sfn -- "$run_dir" "$runs_root/latest"
nohup setsid --fork bash "$repo_root/run_fungraph3d_background.sh" --worker "$run_dir" \
    </dev/null >>"$run_dir/master.log" 2>&1 &
for (( attempt=0; attempt<30; attempt++ )); do
    [[ -f "$run_dir/worker.pid" ]] && break
    sleep 0.1
done
if [[ ! -f "$run_dir/worker.pid" ]]; then
    echo "Worker did not start; inspect $run_dir/master.log" >&2
    exit 1
fi
echo "Detached job: $run_dir"
echo "Worker PID:   $(cat "$run_dir/worker.pid")"
echo "Main log:     $run_dir/master.log"
echo "Status:       $run_dir/status"
echo "Report:       $run_dir/fungraph3d_eval.json"
