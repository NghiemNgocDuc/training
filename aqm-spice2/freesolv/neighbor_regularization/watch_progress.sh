#!/usr/bin/env bash
# Live progress bar for the sequential smoke/sweep launchers on the box.
# Parses the launcher's own log (smoke_v2_seq.log / sweep_v2_seq.log) for the
# `=== [i/N]` and `=== done [i/N]` markers, and the CURRENT run's per-epoch
# log for epoch-level progress.
#
# Usage:  bash watch_progress.sh smoke_v2_seq.log
#         bash watch_progress.sh sweep_v2_seq.log
# Ctrl+C to exit. Run in its own terminal (the launcher runs detached).

LOG="${1:-smoke_v2_seq.log}"
[ -f "$LOG" ] || { echo "log not found: $LOG"; exit 1; }

NR="aqm-spice2/freesolv/neighbor_regularization"
BARW=40

refresh() {
  local total done i name epoch ep_total label bar filled empty pct
  total=$(grep -oP '\[[0-9]+/[0-9]+\]' "$LOG" | tail -1)
  total="${total#\[}"; total="${total%\]}"
  done=$(grep -c '=== done \[' "$LOG" 2>/dev/null || true)
  i="${total%%/*}"; total="${total##*/}"

  # current run = most recent `=== [i/N] <out>` line that is NOT a done marker
  name=$(grep '=== \[' "$LOG" | grep -v 'done \[' | tail -1 \
    | sed -E 's/.*\] ([^ ]+).*/\1/' | xargs basename)
  # per-epoch log: prefer the launcher's own prefix (smoke_/sweep_) so stale
  # logs from a previous batch never shadow the current run
  PREFIX=smoke_
  case "$LOG" in *sweep*) PREFIX=sweep_;; esac
  RLOG="$NR/logs/$(ls "$NR/logs" 2>/dev/null | grep -E "^${PREFIX}${name}\.log$" | head -1)"
  epoch=""
  ep_total=""
  if [ -n "$RLOG" ] && [ -f "$RLOG" ]; then
    epoch=$(grep -oP 'ep\s+\K[0-9]+(?=/)' "$RLOG" | tail -1)
    ep_total=$(grep -oP 'ep\s+[0-9]+/\K[0-9]+' "$RLOG" | tail -1)
  fi

  if [ -z "$total" ] || [ "$total" = "0" ]; then
    echo "[setup] waiting for first run to start..."
    return
  fi

  # per-run weight: each run = 1 unit, epoch detail fills it
  if [ -n "$epoch" ] && [ -n "$ep_total" ] && [ "$ep_total" -gt 0 ]; then
    i=$((done + 1))   # current run is (done+1)th
    frac=$(awk -v d="$done" -v e="$epoch" -v t="$ep_total" \
      'BEGIN { printf "%.3f", d + e/t }')
  else
    frac="$done"
  fi

  pct=$(awk -v f="$frac" -v t="$total" \
    'BEGIN { printf "%d", (t>0 ? f/t*100 : 0) }')
  filled=$((pct * BARW / 100))
  bar=$(printf '%*s' "$filled" '' | tr ' ' '#')
  bar=$(printf '%s%*s' "$bar" $((BARW - filled)) '' | tr ' ' '.')

  if [ -n "$epoch" ] && [ -n "$ep_total" ]; then
    label="${name} (ep ${epoch}/${ep_total})"
  else
    label="$name"
  fi

  printf '\r[%s] %2d%%  %s/%s done | run %s | %s   ' \
    "$bar" "$pct" "$done" "$total" "$i" "$label"
}

# status line (also shows final state / failures)
status() {
  if grep -q 'ALL .* runs finished OK' "$LOG"; then
    echo "DONE: all runs OK"
  elif grep -q 'RUN FAILED' "$LOG"; then
    echo "FAILED run: $(grep 'RUN FAILED' "$LOG" | tail -1)"
  else
    echo "running: $(grep -c '=== done \[' "$LOG" 2>/dev/null) done"
  fi
}

echo "watching $LOG (Ctrl+C to exit)"
while true; do
  refresh
  status
  sleep 5
done