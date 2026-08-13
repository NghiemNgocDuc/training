#!/usr/bin/env bash
# Live progress bar for the smoke/sweep launchers on the box.
# Parses the launcher log for `=== [i/N]` launch markers and `=== done [pid]`
# completion markers, and lists EVERY active (launched, not finished) run with
# its latest epoch from its per-run log.
#
# Usage:  bash watch_progress.sh sweep_v2_seq.log   (run from the repo root)
# Ctrl+C to exit. Run in its own terminal (the launcher runs detached).

LOG="${1:-smoke_v2_seq.log}"
[ -f "$LOG" ] || { echo "log not found: $LOG"; exit 1; }

BARW=40

refresh() {
  local total done label bar filled empty pct a_idx a_log a_ep a_ep_total
  total=$(grep -oP '\[[0-9]+/[0-9]+\]' "$LOG" | tail -1)
  total="${total#\[}"; total="${total%\]}"
  done=$(grep -c '=== done \[' "$LOG" 2>/dev/null || true)

  # active runs = launch markers that never got a done marker; show each one's epoch
  label=""
  while IFS=' ' read -r a_idx a_log; do
    [ -n "$a_idx" ] || continue
    a_ep=""; a_ep_total=""
    if [ -f "$a_log" ]; then
      a_ep=$(grep -oP 'ep\s+\K[0-9]+(?=/)' "$a_log" | tail -1)
      a_ep_total=$(grep -oP 'ep\s+[0-9]+/\K[0-9]+' "$a_log" | tail -1)
    fi
    if [ -n "$a_ep" ]; then label="${label}${a_idx}:${a_ep}/${a_ep_total} "
    else label="${label}${a_idx}:init "; fi
  done < <(grep '=== \[' "$LOG" | grep -v 'done \[' \
    | sed -E 's/=== \[([0-9]+)\/[0-9]+\] [^ ]+ -> ([^ ]+) .*/\1 \2/')

  if [ -z "$total" ] || [ "$total" = "0" ]; then
    echo "[setup] waiting for first run to start..."
    return
  fi

  [ -n "$label" ] || label="all done"
  pct=$(awk -v d="$done" -v t="$total" 'BEGIN { printf "%d", (t>0 ? d/t*100 : 0) }')
  filled=$((pct * BARW / 100))
  bar=$(printf '%*s' "$filled" '' | tr ' ' '#')
  bar=$(printf '%s%*s' "$bar" $((BARW - filled)) '' | tr ' ' '.')

  printf '\r[%s] %2d%%  %s/%s done | %s   ' "$bar" "$pct" "$done" "$total" "$label"
}

status() {
  if grep -q 'ALL .* runs finished OK' "$LOG"; then
    echo "DONE: all runs OK"
  elif grep -q 'FAILED' "$LOG"; then
    echo "FAILED: $(grep '=== done .*FAILED' "$LOG" | tail -1)"
  fi
}

echo "watching $LOG (Ctrl+C to exit)"
while true; do
  refresh
  status
  sleep 5
done
