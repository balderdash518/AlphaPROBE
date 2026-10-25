#!/usr/bin/env bash
# Run every AlphaPROBE experiment from the paper that this repo can run, one
# after another: factor generation (train_new_work.py, 20 iterations) followed
# by the adaptive factor combination on that run's pool_20.json.
#
#   bash scripts/run_paper_experiments.sh            # run the queue
#   bash scripts/run_paper_experiments.sh --list     # print the queue only
#   bash scripts/run_paper_experiments.sh main_sp500 main_csi300   # only these jobs
#
# A job that finished writes logs/paper/<name>.done and is skipped on the next
# start. Training runs with --resume, so a job interrupted mid-way (shutdown,
# crash) continues from its last finished iteration when the queue is started
# again. If a train_new_work.py run is already going (e.g. started by hand),
# the queue waits for it first.
# Not covered (no code in this repo): baselines from other projects, random /
# heuristic / MCTS retrievers, w/o Prior, w/o Likelihood, CoT generator.
set -u
cd "$(dirname "$0")/.."

PY=.venv/Scripts/python.exe
CONFIG=config/alphaprobe_main.json
LOG_DIR=logs/paper
mkdir -p "$LOG_DIR"
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8

# name|train_new_work.py arguments (on top of $CONFIG)
JOBS=(
  # Table 1 main results (+ the US test universe)
  "main_sp500|--instruments sp500"
  "main_csi300|--instruments csi300"
  "main_csi500|--instruments csi500"
  "main_csi1000|--instruments csi1000"
  # Figure 4 sensitivity on CSI 300 (main config is gamma=0.05, omega=0.10)
  "sens_omega0.00_csi300|--instruments csi300 --times_decay 0.0"
  "sens_omega0.05_csi300|--instruments csi300 --times_decay 0.05"
  "sens_omega0.15_csi300|--instruments csi300 --times_decay 0.15"
  "sens_omega0.20_csi300|--instruments csi300 --times_decay 0.20"
  "sens_gamma0.00_csi300|--instruments csi300 --depth_decay 0.0"
  "sens_gamma0.10_csi300|--instruments csi300 --depth_decay 0.10"
  "sens_gamma0.15_csi300|--instruments csi300 --depth_decay 0.15"
  "sens_gamma0.20_csi300|--instruments csi300 --depth_decay 0.20"
  # Table 2 ablations available through flags
  "abl_no_topology_csi300|--instruments csi300 --depth_decay 0.0 --times_decay 0.0"
  "abl_no_topology_csi500|--instruments csi500 --depth_decay 0.0 --times_decay 0.0"
  "abl_no_topology_csi1000|--instruments csi1000 --depth_decay 0.0 --times_decay 0.0"
  "abl_no_nlf_csi300|--instruments csi300 --no-separate_leaf_non_leaf"
  "abl_no_nlf_csi500|--instruments csi500 --no-separate_leaf_non_leaf"
  "abl_no_nlf_csi1000|--instruments csi1000 --no-separate_leaf_non_leaf"
)

if [ "${1:-}" = "--list" ]; then
  for job in "${JOBS[@]}"; do
    name=${job%%|*}
    state=pending; [ -f "$LOG_DIR/$name.done" ] && state=done
    printf '%-26s %-8s %s\n' "$name" "$state" "${job#*|}"
  done
  exit 0
fi

if [ $# -gt 0 ]; then
  selected=()
  for job in "${JOBS[@]}"; do
    for want in "$@"; do [ "${job%%|*}" = "$want" ] && selected+=("$job"); done
  done
  [ ${#selected[@]} -eq $# ] || { echo "unknown job name in: $* (see --list)"; exit 1; }
  JOBS=("${selected[@]}")
fi

training_running() {
  powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_new_work*' }).Count" | tr -d '\r'
}

note() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/queue.log"; }

while [ "$(training_running)" != "0" ]; do
  note "waiting for a running train_new_work.py to finish"
  sleep 600
done

for job in "${JOBS[@]}"; do
  name=${job%%|*}; args=${job#*|}
  [ -f "$LOG_DIR/$name.done" ] && { note "skip $name (done)"; continue; }
  instruments=$(sed -E 's/.*--instruments ([^ ]+).*/\1/' <<<"$args")
  train_log="$LOG_DIR/$name.train.log"

  note "train $name: $args"
  echo "===== $(date '+%F %T') start/resume =====" >> "$train_log"
  # shellcheck disable=SC2086
  # The per-iteration pool evaluation (log_freq) is monitoring only - the search never reads its
  # test metrics - and costs ~50 min per iteration on CSI 300, so by default only the last
  # iteration is evaluated. LOG_FREQ=1 restores per-iteration curves.
  if ! $PY train_new_work.py --config "$CONFIG" --seed 0 --resume --log_freq "${LOG_FREQ:-20}" $args >> "$train_log" 2>&1; then
    note "FAILED train $name (see $train_log); start the queue again to resume it"; continue
  fi
  run_dir=$(grep -a '^Log dir: ' "$train_log" | tail -1 | cut -d' ' -f3-)
  pool="$run_dir/pool_20.json"
  if [ ! -f "$pool" ]; then note "FAILED $name: $pool not found"; continue; fi

  note "combine $name: $pool"
  if ! $PY run_adaptive_combination.py --config "$CONFIG" --seed 0 --instruments "$instruments" \
        --expressions_file "$pool" > "$LOG_DIR/$name.combine.log" 2>&1; then
    note "FAILED combine $name (see $LOG_DIR/$name.combine.log)"; continue
  fi
  echo "$pool" > "$LOG_DIR/$name.done"
  note "done $name"
done
note "queue finished"
