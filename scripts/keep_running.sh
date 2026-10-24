#!/usr/bin/env bash
# Watchdog: keep re-running queue jobs until they are done.
#
#   bash scripts/keep_running.sh main_sp500 [more job names]
#
# Waits for any AlphaPROBE queue / training / combination process that is
# already running, then starts scripts/run_paper_experiments.sh for the given
# jobs (training resumes from its checkpoint) until every job has its .done
# file, giving up after MAX_ATTEMPTS. Start it detached so it survives the
# terminal or app that launched it, e.g. from PowerShell:
#   Start-Process -WindowStyle Hidden "C:\Program Files\Git\bin\bash.exe" `
#     -ArgumentList "scripts/keep_running.sh main_sp500" -WorkingDirectory C:\yifei\AlphaPROBE
set -u
cd "$(dirname "$0")/.."
LOG=logs/paper/watchdog.log
MAX_ATTEMPTS=${MAX_ATTEMPTS:-5}
mkdir -p logs/paper
note() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# Match by process name as well: the powershell running this query has the
# patterns in its own command line and would otherwise count itself.
others_running() {
  powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { (\$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'train_new_work\.py|run_adaptive_combination\.py') -or (\$_.Name -eq 'bash.exe' -and \$_.CommandLine -match 'scripts/run_paper_experiments\.sh' -and \$_.CommandLine -notmatch ' -c ') }).Count" | tr -d '\r'
}

all_done() {
  for job in "$@"; do [ -f "logs/paper/$job.done" ] || return 1; done
}

note "watchdog started for: $* (pid $$)"
attempt=0
while ! all_done "$@"; do
  while [ "$(others_running)" != "0" ]; do sleep 120; done
  all_done "$@" && break
  attempt=$((attempt + 1))
  if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
    note "giving up after $MAX_ATTEMPTS attempts"; exit 1
  fi
  note "attempt $attempt: running queue for $*"
  bash scripts/run_paper_experiments.sh "$@"
  sleep 60
done
note "all done: $*"
