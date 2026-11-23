#!/usr/bin/env bash
# One exclusive device-0 window: stop production under the flock, run the page-group sweep and
# the dedup A/B, then restore production and verify health 200. Never leaves it stopped.
set -uo pipefail
LOCK=/tmp/ninfer-gpu-window.lock
LOGS=~/git/ninfer-all-paged/logs
SCRIPTS=~/git/ninfer-all-paged/scripts
mkdir -p "$LOGS"

(
  flock -x -w 120 9 || { echo "GPU window busy: another lane holds $LOCK"; exit 1; }
  echo "== window open: $(date) =="
  sudo -n systemctl stop ninfer-quant
  restore_production() {
    sudo -n systemctl start ninfer-quant
    local ok=0
    for _ in $(seq 1 90); do
      curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && { ok=1; break; }; sleep 2
    done
    if [ "$ok" = 1 ]; then
      echo "== production restored, health 200, $(date) =="
    else
      echo "== PRODUCTION RESTORE FAILED: health never went 200 =="
    fi
  }
  trap restore_production EXIT
  PORT=8084 bash "$SCRIPTS/paged-sweep.sh"
  PORT=8084 bash "$SCRIPTS/paged-dedup.sh"
  echo "== window measurements done: $(date) =="
) 9>"$LOCK"
