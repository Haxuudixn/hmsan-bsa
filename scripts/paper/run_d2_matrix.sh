#!/bin/bash
# D2 matrix queue (2026-10-03): the 7 decided rows at the revised budget.
#
# What changed vs the 2026-09-28 matrix, and why.
#   D2  8 -> 10 epochs.  Measured on the 16-epoch extension of `delivered`
#       (docs/path3_overnight_2026-10-03.md): the macro-F1 ENVELOPE tops out at
#       epoch 10 (0.89177) and does not improve in the following six epochs,
#       while the 8-epoch envelope (0.8668) sits 2.5 pp below it - 2.4x the
#       1.04 pp median noise floor.  So 8 epochs systematically under-reports
#       macro F1 and 10 captures the plateau.
#   LR   --lr_patience 1 --lr_min_lr 1e-5.  train.py steps
#       ReduceLROnPlateau ONCE PER EPOCH on the epoch-end val accuracy, so
#       patience is counted in epochs.  With patience=3 the LR barely moved in
#       16 epochs (one halving at most) and two epochs collapsed late
#       (ep10 0.8416, ep16 0.9070) while the training loss kept falling.  On
#       the measured val series patience=1 halves the LR at ep7/8/9/10, which
#       is the intent; --lr_min_lr 1e-5 floors it so the schedule cannot run
#       away and freeze learning early.
#
# EVERY row gets the same protocol.  Comparing a patience=1 row against a
# patience=3 row is not a comparison, it is two experiments.
#
# Output root is NEW on purpose: outputs/path3_d2_10ep, not outputs/path3_clean.
# The 8-epoch runs still live in outputs/path3_clean/<row>/, and run_matrix.py
# auto-resumes any row that has a last.pt - pointing it at the old root would
# silently continue an 8-epoch checkpoint under a 10-epoch protocol.  The new
# root also keeps the 8-vs-10 epoch comparison readable for the thesis.
#
# The 7 rows are D3's core 6 (`--core`) plus `delivered`, which is the paper
# main model and is the same-config same-seed replicate of `full` - so this run
# also refreshes the noise floor at the new budget.
#
# Resumable: a row with result.json is skipped, a crashed row is retried from
# its own last.pt by run_matrix.py (--max_attempts 2).
set -u
R=/root/autodl-tmp/hmsan-bsa
PY=/root/autodl-tmp/envs/hmsan/bin/python
LOG=/root/autodl-tmp/d2_matrix.log
MIN_FREE_GB=14
MIN_FREE_VRAM_MB=11000
SLOTS=2
DATA=/root/autodl-tmp/final_train
OUT_ROOT=outputs/path3_d2_10ep
EPOCHS=10

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1

say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

free_gb() { df -BG --output=avail /root/autodl-tmp | tail -1 | tr -dc 0-9; }
free_vram_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null |
    head -1 | tr -dc 0-9
}

say "D2 matrix queue: ${EPOCHS} epochs, patience=1, min_lr=1e-5, ${SLOTS} slots"
"$PY" -c "import torch,sys;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" >> "$LOG" 2>&1

# Order is deliberate: the two `full`-config replicates first (they are the
# paper's main number and the noise-floor pair), then the two contested
# ablations, then the rest.  Stopping early still leaves the headline rows.
# Local/remote split (2026-10-03): the laptop 3060 runs A7_no_boundary_gate
# alone, the 4090 runs the other six.  Throughput is 0.44 row-epoch/h locally
# vs 1.67 on the 4090, so the split is 6+1 = 36 h instead of 42 h (all remote)
# or 45 h (2 local + 5 remote).  A7 is the row the thesis does not report
# (D8: no boundary concept), so it is the cheapest one to put on the machine
# that bluescreens.  Override with D2_ROWS="..." to re-run a subset.
ROWS="${D2_ROWS:-delivered full A6_no_page_memory A2_no_g2 A10_dense A11_no_gates A7_no_boundary_gate}"

pending=()
for row in $ROWS; do
  if [ -f "$R/$OUT_ROOT/$row/result.json" ]; then
    say "skip $row: result.json present"
  else
    pending+=("$row")
  fi
done
say "queue: ${#pending[@]} row(s) pending, ${SLOTS} at a time"

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
      --output_root "$OUT_ROOT" --epochs "$EPOCHS" \
      --lr_patience 1 --lr_min_lr 1e-5 \
      --num_workers 4 --max_attempts 2 --continue_on_failure \
      > "/root/autodl-tmp/d2_matrix_$row.log" 2>&1
    rc=$?
    say "=== $row: run_matrix exit=$rc ==="
    rm -f "$R/$OUT_ROOT/$row/checkpoint_epoch5.pt"
    say "=== $row: finished, $(free_gb) GB free ==="
  ) &
  running=$((running + 1))
done
wait

say "remote D2 matrix queue done"