"""Remote probe for scripts/paper/status.ps1.

Runs ON the rented box (piped in over ssh as base64) and prints one JSON blob
describing the D2 matrix rows plus the machine, so the dashboard needs exactly
one ssh round trip.  Read-only: it opens result.json / run_meta.json /
history.json / train.log and never writes inside a run directory.

    base64 -d <<< "$b64" | /root/autodl-tmp/envs/hmsan/bin/python - [root]
"""
import json
import os
import re
import subprocess
import sys
import time

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/hmsan-bsa/outputs/path3_d2_10ep"
QUEUE_LOG = "/root/autodl-tmp/d2_matrix.log"
ROWS = [
    "delivered",
    "full",
    "A6_no_page_memory",
    "A2_no_g2",
    "A10_dense",
    "A11_no_gates",
    "A7_no_boundary_gate",
]
EVIDENCE = ("train.log", "history.json", "last.pt", "best_model.pt")

RE_PROG = re.compile(
    r"Training:\s+\d+%\s*\|.*\|\s*(\d+)/(\d+)\s*\[([0-9:]+)<([0-9:]+),\s*([0-9.]+)s/it"
)
RE_EPOCH = re.compile(r"--- Epoch (\d+)/(\d+) ---")
RE_MID = re.compile(r"\[mid ep(\d+) step (\d+)\]\s*acc=([0-9.]+)\s+macro_f1=([0-9.]+)")


def read_text(path):
    try:
        with open(path, "rb") as handle:
            return handle.read().decode("utf-8", "replace")
    except Exception:
        return ""


def val_of(record, key):
    return (record.get("val") or {}).get(key)


def freshest_age_min(directory):
    """Minutes since the newest artifact, in the set that training actually
    touches.  train.log alone is not a liveness signal: a run started under the
    older run_matrix.py block-buffers the child, so the log can sit stale for
    minutes while history.json and last.pt are current."""
    ages = []
    for name in EVIDENCE:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            ages.append((time.time() - os.path.getmtime(path)) / 60.0)
    return round(min(ages), 1) if ages else None


def probe_row(rid, running_text):
    directory = os.path.join(ROOT, rid)
    state = {
        "id": rid,
        "host": "remote",
        "exists": os.path.isdir(directory),
        "running": ("--output_dir " + directory) in running_text,
    }
    if not state["exists"]:
        return state

    result_path = os.path.join(directory, "result.json")
    meta_path = os.path.join(directory, "run_meta.json")
    history_path = os.path.join(directory, "history.json")
    log_path = os.path.join(directory, "train.log")

    if os.path.exists(result_path):
        try:
            state["result_status"] = json.load(open(result_path)).get("status")
        except Exception:
            state["result_status"] = "unreadable"
    if os.path.exists(meta_path):
        try:
            state["started_at"] = json.load(open(meta_path)).get("started_at")
        except Exception:
            pass

    if os.path.exists(history_path):
        try:
            history = json.load(open(history_path))
            # One record per epoch end (no "phase" key) plus three mid-epoch
            # probes per epoch ("phase": "mid_epoch").
            ends = [r for r in history if r.get("phase") is None]
            mids = [r for r in history if r.get("phase") == "mid_epoch"]
            state["epochs_done"] = len(ends)
            state["curve"] = [
                {"epoch": r.get("epoch"), "acc": val_of(r, "accuracy"), "f1": val_of(r, "macro_f1")}
                for r in ends
            ]
            if ends:
                best = max(ends, key=lambda r: val_of(r, "accuracy") or -1.0)
                last = ends[-1]
                state["best_epoch"] = best.get("epoch")
                state["best_acc"] = val_of(best, "accuracy")
                state["best_f1"] = val_of(best, "macro_f1")
                state["last_epoch"] = last.get("epoch")
                state["last_acc"] = val_of(last, "accuracy")
                state["last_f1"] = val_of(last, "macro_f1")
                state["peak_vram_mb"] = last.get("peak_vram_mb")
            if mids:
                mid = mids[-1]
                state["mid"] = {
                    "epoch": mid.get("epoch"),
                    "step": mid.get("step"),
                    "acc": val_of(mid, "accuracy"),
                    "macro_f1": val_of(mid, "macro_f1"),
                }
        except Exception as exc:
            state["history_error"] = str(exc)

    log_text = read_text(log_path)
    if log_text:
        state["oom_count"] = log_text.count("[oom]")
        epochs = RE_EPOCH.findall(log_text)
        if epochs:
            state["epoch_index"] = int(epochs[-1][0])
            state["epochs_planned"] = int(epochs[-1][1])
        progress = RE_PROG.findall(log_text)
        if progress:
            state["phase"] = "Training"
            state["step_done"] = int(progress[-1][0])
            state["step_total"] = int(progress[-1][1])
            state["s_per_it"] = float(progress[-1][4])
        tail = [ln for ln in log_text.splitlines() if ln.strip()]
        if tail and "Validation:" in tail[-1]:
            state["phase"] = "Validation"

    mids = RE_MID.findall(log_text) if log_text else []
    if mids:
        state["mid_log"] = {
            "epoch": int(mids[-1][0]),
            "step": int(mids[-1][1]),
            "acc": float(mids[-1][2]),
            "macro_f1": float(mids[-1][3]),
        }

    state["fresh_min"] = freshest_age_min(directory)
    state["has_last_pt"] = os.path.exists(os.path.join(directory, "last.pt"))
    state["has_best_pt"] = os.path.exists(os.path.join(directory, "best_model.pt"))
    return state


def probe_machine():
    machine = {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        parts = [p.strip() for p in out.split(",")]
        machine["gpu_util"] = int(parts[0])
        machine["gpu_mem_used"] = int(parts[1])
        machine["gpu_mem_total"] = int(parts[2])
        machine["gpu_temp"] = int(parts[3])
    except Exception:
        pass
    try:
        stat = os.statvfs("/root/autodl-tmp")
        machine["disk_free_gb"] = round(stat.f_bavail * stat.f_frsize / 1e9, 1)
    except Exception:
        pass
    try:
        machine["queue_log_tail"] = [ln for ln in read_text(QUEUE_LOG).splitlines() if ln.strip()][-8:]
    except Exception:
        pass
    try:
        machine["screen"] = subprocess.run(
            ["screen", "-ls"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        pass
    return machine


def main():
    try:
        running_text = subprocess.run(
            ["pgrep", "-af", "train.py"], capture_output=True, text=True, timeout=15
        ).stdout
    except Exception:
        running_text = ""
    payload = {
        "ok": True,
        "rows": [probe_row(rid, running_text) for rid in ROWS],
        "machine": probe_machine(),
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()