#!/bin/bash
# Remote queue for phase 2 of the clean rerun, in protocol order:
#   1. the matrix test pass   (screen "testev", started separately)
#   2. the baseline test pass (this script)
#   3. the grey-zone seed rows (this script)
#
# The order matters: the protocol forbids touching the test split while
# anything is still training, so both test passes finish before the seed queue
# starts.  And the baseline pass has to happen before the seed rows so the
# finished table-3 comparison is one single test pass over frozen checkpoints,
# not a test number against a validation number.
#
# Baseline rows: b1_roberta_mlp, b2_roberta_layout_mlp, b3_page_bigru -- the
# three rows table 3 reports.  B3 is scored twice (cap=48 as trained, cap=3000
# for the matched denominator); the two readings are recorded under separate
# keys and must never share a column.
#
# Seed rows (manifest group "seed_replication"): A2_no_g2 / A6_no_page_memory /
# A7_no_boundary_gate / A10_dense at seeds 43 and 44, 8 epochs, protocol
# unchanged.  Their seed-42 deltas are -2.22 / +1.69 / -2.80 / -1.32 pp of
# macro F1, all inside the 1-3 pp grey band of the pre-registered rule, so the
# paper cannot call any of them a real effect until they are repeated.
#
# Order inside the seeds: seed 43 for all four rows first, then seed 44, so
# stopping after the first four still leaves one complete extra-seed set.
#
# SLOTS=2.  The rented 4090 has 24.5 GB and every matrix row peaks at about
# 8.9 GB (measured, `peak_vram_mb` in each row's result.json), so two rows fit
# with roughly 6 GB to spare and the 128-core host is nowhere near CPU-bound.
# That halves the queue from about 86 h to about 45 h.  Concurrency cannot
# change any number: the rows are separate processes writing separate
# directories, each seeded independently, and the only shared artifact is the
# ViT feature cache, which training opens read-only (memmap).  A row is only
# started when the card actually has MIN_FREE_VRAM_MB free, so a transient
# spike can never be the reason a run dies.
#
# Disk: eight seed rows need about 26 GB of checkpoints (best_model.pt and
# last.pt are 1.6 GB each) plus a transient checkpoint_epoch5.pt, and the data
# disk holds the 37 GB corpus.  The guard below stops the queue instead of
# filling the disk, and each finished row drops its checkpoint_epoch5.pt
# because nothing of record reads it.
#
# Resumable: run_matrix.py skips a row already marked completed and retries a
# crashed one from its own last.pt, and the row loop below skips any row that
# already has a result.json, so re-running this script after an interruption
# continues where it stopped.
set -u
R=/root/autodl-tmp/hmsan-bsa
PY=/root/autodl-tmp/envs/hmsan/bin/python
LOG=/root/autodl-tmp/grey_seeds.log
MIN_FREE_GB=14
MIN_FREE_VRAM_MB=11000
SLOTS=2
DATA=/root/autodl-tmp/final_train

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1

say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

free_gb() { df -BG --output=avail /root/autodl-tmp | tail -1 | tr -dc 0-9; }
free_vram_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null |
    head -1 | tr -dc 0-9
}

while screen -ls 2>/dev/null | grep -q "\.testev\b"; do
  say "waiting for the matrix test pass (testev) to finish"
  sleep 300
done
say "matrix test pass finished"

# ---------------------------------------------------------------- baselines
BT=$R/outputs/path3_diag/test_eval_baselines.json
n_done() { if [ -f "$BT" ]; then grep -c '"key"' "$BT"; else echo 0; fi; }

score_baseline() {
  label=$1; shift
  cd "$R" || exit 1
  "$PY" scripts/paper/baselines_test_eval.py "$@" --data_dir "$DATA" >> "$LOG" 2>&1
  say "baseline test eval [$label] exit=$?"
}

if [ "$(n_done)" -ge 4 ]; then
  say "skip baseline test pass: $BT already has $(n_done) entries"
else
  say "scoring the table-3 baselines on the test split"
  score_baseline b1 --baseline b1 --row_dir outputs/path3_clean/b1_roberta_mlp
  score_baseline b2 --baseline b2 --row_dir outputs/path3_clean/b2_roberta_layout_mlp
  score_baseline b3cap48 --baseline b3 --row_dir outputs/path3_clean/b3_page_bigru --max_blocks_per_page 48
  score_baseline b3cap3000 --baseline b3 --row_dir outputs/path3_clean/b3_page_bigru --max_blocks_per_page 3000
  say "baseline test pass done -> $BT"
fi

# ------------------------------------------------------------------- seeds
ROWS="A2_no_g2_s43 A6_no_page_memory_s43 A7_no_boundary_gate_s43 A10_dense_s43 A2_no_g2_s44 A6_no_page_memory_s44 A7_no_boundary_gate_s44 A10_dense_s44"

pending=()
for row in $ROWS; do
  if [ -f "$R/outputs/path3_clean/$row/result.json" ]; then
    say "skip $row: result.json present"
  else
    pending+=("$row")
  fi
done
say "seed queue: ${#pending[@]} row(s) pending, ${SLOTS} at a time"

running=0
for row in "${pending[@]+"${pending[@]}"}"; do
  while [ "$running" -ge "$SLOTS" ]; do
    wait -n
    running=$((running - 1))
  done

  avail=$(free_gb)
  if [ "$avail" -lt "$MIN_FREE_GB" ]; then
    say "STOP: only ${avail} GB free on the data disk (guard is ${MIN_FREE_GB} GB)."
    say "      Free space and re-run this script; finished rows are skipped."
    break
  fi

  vram=$(free_vram_mb)
  while [ "$running" -gt 0 ] && [ "$vram" -lt "$MIN_FREE_VRAM_MB" ]; do
    say "waiting for VRAM before $row: ${vram} MB free (need ${MIN_FREE_VRAM_MB})"
    sleep 120
    vram=$(free_vram_mb)
  done

  say "=== $row: start (disk ${avail} GB free, vram ${vram} MB free) ==="
  (
    cd "$R" || exit 1
    "$PY" scripts/paper/run_matrix.py --only "$row" --data_dir "$DATA" \
      --num_workers 4 --max_attempts 2 --continue_on_failure \
      > "/root/autodl-tmp/grey_seeds_$row.log" 2>&1
    rc=$?
    say "=== $row: run_matrix exit=$rc ==="
    rm -f "$R/outputs/path3_clean/$row/checkpoint_epoch5.pt"
    say "=== $row: finished, $(free_gb) GB free ==="
  ) &
  running=$((running + 1))
done
wait

# run_matrix rewrites summary.json read-modify-write, so two rows finishing in
# the same instant could drop one entry.  The per-row result.json is the record
# of truth; this only reports whether the index lost anything.
"$PY" - <<'PYEOF' >> "$LOG" 2>&1
import json
import pathlib

root = pathlib.Path("/root/autodl-tmp/hmsan-bsa/outputs/path3_clean")
summary_path = root / "summary.json"
try:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
except (OSError, ValueError):
    summary = []
known = {row.get("id") for row in summary}
finished = sorted(p.parent.name for p in root.glob("*/result.json"))
missing = [name for name in finished if name not in known]
print(f"summary.json holds {len(summary)} row(s); {len(finished)} result.json present")
print("rows missing from summary.json:", ", ".join(missing) if missing else "none")
PYEOF

say "remote queue done"