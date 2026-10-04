#!/bin/bash
# Stage 5 (Linux port of scripts/paper/run_clean_eval.ps1)
#
# Re-score the clean matrix on the leak-free eval sets. Nothing is trained and
# no run directory is written to: the only outputs are the JSON report and the
# log next to it. The rule this driver exists to enforce is *one* test pass,
# after the matrix has stopped, over frozen checkpoints.
#
#   ./scripts/paper/run_clean_eval.sh --dry-run
#   ./scripts/paper/run_clean_eval.sh --gate A10_dense
#
# Exits non-zero if any pass fails.

set -u

REPO="${REPO:-/root/autodl-tmp/hmsan-bsa}"
PY="${PY:-/root/autodl-tmp/envs/hmsan/bin/python}"
GATE=""
POLL=60
SETTLE=120
MATRIX_ROOT="outputs/path3_clean"
OUT="outputs/path3_diag/leak_diag_clean.json"
INDEX="outputs/path3_diag/corpus_index.json"
LIMIT_SEGMENTS=0
SETS=""
DRY_RUN=0
FORCE=0
PASSES="all"          # all | matrix | delivered
BUILTIN=""            # empty => pass --no_builtin to the scorer

usage() {
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --gate ROW          wait until outputs/<matrix_root>/ROW/result.json says
                      status=completed and no train.py is running (default: no wait)
  --dry-run           resolve the work and print it, then stop before scoring
  --force             score even while a train.py is running
  --matrix-root DIR   matrix directory (default: outputs/path3_clean)
  --out FILE          report path
  --index FILE        corpus index (default: outputs/path3_diag/corpus_index.json)
  --limit-segments N  debug: score only the first N segments per set
  --sets LIST         comma-separated subset of eval sets
  --passes WHICH      all | matrix | delivered  (default: all)
  --poll SECONDS      gate poll interval (default: 60)
  --settle SECONDS    quiet period after the gate opens (default: 120)
  --with-builtin      also score the legacy hard-coded CHECKPOINTS list
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --gate)           GATE="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=1; shift ;;
        --force)          FORCE=1; shift ;;
        --matrix-root)    MATRIX_ROOT="$2"; shift 2 ;;
        --out)            OUT="$2"; shift 2 ;;
        --index)          INDEX="$2"; shift 2 ;;
        --limit-segments) LIMIT_SEGMENTS="$2"; shift 2 ;;
        --sets)           SETS="$2"; shift 2 ;;
        --passes)         PASSES="$2"; shift 2 ;;
        --poll)           POLL="$2"; shift 2 ;;
        --settle)         SETTLE="$2"; shift 2 ;;
        --with-builtin)   BUILTIN="yes"; shift ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

cd "$REPO" || { echo "no repo at $REPO" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")" "$(dirname "$INDEX")"
LOG="$(dirname "$OUT")/run_clean_eval.log"

log() { printf '[clean-eval] %s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }

training_count() {
    # Match the script name exactly; a bare 'train.py' pattern also matches the
    # ssh command line that invoked this driver.
    ps -eo args | grep -c '[t]rain[.]py'
}

status_of() {
    "$PY" - "$1" <<'PY' 2>/dev/null
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("status", ""))
except Exception:
    print("")
PY
}

if [ -n "$GATE" ]; then
    log "waiting for $MATRIX_ROOT/$GATE/result.json -> status=completed and no train.py"
    while :; do
        st="$(status_of "$MATRIX_ROOT/$GATE/result.json")"
        n="$(training_count)"
        if [ "$st" = "completed" ] && [ "$n" -eq 0 ]; then
            break
        fi
        if [ "$DRY_RUN" -eq 1 ]; then
            log "dry-run: gate status=$st train.py=$n -- not waiting further"
            break
        fi
        log "gate status='${st:-missing}' train.py=$n"
        sleep "$POLL"
    done
    log "gate open; settling ${SETTLE}s"
    [ "$DRY_RUN" -eq 1 ] || sleep "$SETTLE"
else
    running="$(training_count)"
    if [ "$running" -gt 0 ] && [ "$FORCE" -eq 0 ]; then
        log "REFUSING: $running train.py process(es) running; the protocol is one"
        log "pass over frozen checkpoints. Re-run with --force to override."
        exit 3
    fi
fi

common=(--index "$INDEX" --matrix_root "$MATRIX_ROOT" --out "$OUT")
[ -n "$BUILTIN" ] || common+=(--no_builtin)
[ "$LIMIT_SEGMENTS" -gt 0 ] && common+=(--limit_segments "$LIMIT_SEGMENTS")
[ -n "$SETS" ] && common+=(--sets "$SETS")

rc=0
run_pass() {
    local label="$1"; shift
    log "$label"
    if [ "$DRY_RUN" -eq 1 ]; then
        log "dry-run: would run  $PY scripts/paper/leak_diag_eval.py $*"
        return 0
    fi
    "$PY" scripts/paper/leak_diag_eval.py "$@" 2>&1 | tee -a "$LOG"
    local r="${PIPESTATUS[0]}"
    log "$label exit=$r"
    [ "$r" -eq 0 ] || rc=1
}

case "$PASSES" in
    all|matrix)
        run_pass "pass 1/2: every discovered row (best_model.pt + last.pt)" "${common[@]}"
        ;;
esac
case "$PASSES" in
    all|delivered)
        # Separate invocation: a preset-signature mismatch on the delivered
        # checkpoint must not be able to abort the matrix pass.
        run_pass "pass 2/2: delivered checkpoint" "${common[@]}" \
            --checkpoint delivered/best --checkpoint delivered/last
        ;;
esac

log "DONE -> $OUT (rc=$rc)"
exit "$rc"