"""Run the path-3 clean matrix on Linux or Windows.

Cross-platform replacement for ``run_comparison.ps1``.  The protocol of record
is ``experiments/path3_clean_manifest.json``; this script turns each row into
the exact ``train.py`` argv that ``run_train_final.ps1`` would have produced
(compare against ``run_train_final.ps1 -PrintOnly``), runs the rows serially,
and writes a ``result.json`` in the same shape the PowerShell runner used.

Why a second runner exists
--------------------------
* ``run_comparison.ps1`` is PowerShell, and the clean rerun moved to a rented
  Linux GPU (the local machine blue-screened mid-epoch on 2026-09-26).
* It aggregates *every* ``history.json`` entry.  That silently changed meaning
  when F5/F6 started appending ``phase="mid_epoch"`` records: a mid-epoch check
  can out-score every epoch-end value, and the frozen report convention (D6)
  keys on the *last epoch*.  This runner reads epoch-boundary records only.

Only one training runs at a time, matching the 12 GB constraint.

Usage
-----
    # gate: 1-epoch smoke, and calibrate the loader workers while you are there
    python scripts/paper/run_matrix.py --only stage1_smoke_full --num_workers 4

    # pilot: sets the epoch budget (D2)
    python scripts/paper/run_matrix.py --only stage2_pilot_full_seed42

    # matrix: after D2, run every core row at the chosen budget
    python scripts/paper/run_matrix.py --core --epochs 4

    # baselines
    python scripts/paper/run_matrix.py --group baselines

    python scripts/paper/run_matrix.py --core --dry_run   # print argv only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MID_EPOCH = "mid_epoch"


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def read_json(path: Path, default):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def epoch_records(history: list) -> list:
    """Only true epoch boundaries.

    F5/F6 append ``phase="mid_epoch"`` entries inside the epoch.  Counting them
    as epochs inflates ``epochs_completed`` and lets a mid-epoch check become
    the reported "final" metric.
    """
    return [h for h in history if h.get("phase") != MID_EPOCH]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="experiments/path3_clean_manifest.json")
    p.add_argument("--data_dir", default=None,
                   help="corpus directory; defaults to the manifest protocol")
    p.add_argument("--output_root", default="outputs/path3_clean")
    p.add_argument("--vit_cache_dir", default=None,
                   help="defaults to the manifest's cache.default_dir when the "
                        "directory exists; pass '' to disable the cache entirely")
    p.add_argument("--label_config", default="configs/labels.yaml")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--device", default="cuda")
    p.add_argument("--only", nargs="+", default=None, metavar="ID")
    p.add_argument("--group", default=None)
    p.add_argument("--core", action="store_true",
                   help="only the rows flagged core:true")
    p.add_argument("--epochs", type=int, default=None,
                   help="budget for rows whose manifest entry leaves epochs null")
    p.add_argument("--epochs_all", type=int, default=None,
                   help="force this budget on every selected row")
    p.add_argument("--num_workers", type=int, default=None,
                   help="DataLoader workers; overrides the manifest")
    # D2 follow-up (2026-10-03): train.py steps the LR scheduler ONCE PER EPOCH,
    # on the epoch-end val accuracy, so --lr_patience is measured in epochs.
    # Both default to None = the flag is not passed = train.py keeps its own
    # default (patience 3, no floor), i.e. the historical protocol.
    p.add_argument("--lr_patience", type=int, default=None,
                   help="pass through to train.py --lr_patience")
    p.add_argument("--lr_min_lr", type=float, default=None,
                   help="pass through to train.py --lr_min_lr")
    p.add_argument("--omp_threads", type=int, default=8)
    p.add_argument("--max_attempts", type=int, default=2)
    p.add_argument("--retry_delay", type=float, default=60.0)
    p.add_argument("--force", action="store_true",
                   help="re-run rows whose result.json already says completed")
    p.add_argument("--continue_on_failure", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def select_runs(manifest: dict, args: argparse.Namespace) -> list:
    runs = [r for r in manifest["runs"] if r.get("enabled", True)]
    if args.only:
        known = {r["id"] for r in runs}
        missing = [i for i in args.only if i not in known]
        if missing:
            raise SystemExit(f"unknown run id(s): {missing}\navailable: {sorted(known)}")
        wanted = set(args.only)
        runs = [r for r in runs if r["id"] in wanted]
    elif args.group:
        runs = [r for r in runs if r.get("group") == args.group]
    elif args.core:
        runs = [r for r in runs if r.get("core")]
    if not runs:
        raise SystemExit("no rows selected")
    return sorted(runs, key=lambda r: r.get("order", 999))


def resolve_epochs(run: dict, protocol: dict, args: argparse.Namespace) -> int:
    if args.epochs_all is not None:
        value = args.epochs_all
    elif run.get("epochs") is not None:
        value = int(run["epochs"])
    elif args.epochs is not None:
        value = args.epochs
    elif protocol.get("epochs") is not None:
        value = int(protocol["epochs"])
    else:
        raise SystemExit(
            f"run '{run['id']}' has no epoch budget and the protocol leaves "
            f"D2 open.\nThe stage-2 pilot decides it; until then pass --epochs N "
            f"(the smoke and pilot rows carry explicit budgets of their own)."
        )
    floor = run.get("epochs_min")
    if floor and value < int(floor):
        print(f"  WARNING: '{run['id']}' asks for at least {floor} epochs "
              f"(long-schedule paper row); --epochs {value} is below that.")
    return value


def build_train_argv(run, protocol, args, out_dir: Path, epochs: int,
                     cache_dir: str | None) -> list:
    a = [
        args.python, "train.py",
        "--data_dir", args.data_dir,
        "--output_dir", str(out_dir),
        "--label_config", args.label_config,
        "--device", args.device,
        "--epochs", str(epochs),
        "--encoder_lr", "2e-5",
        "--head_lr", "1e-4",
        "--weight_decay", "1e-4",
        "--grad_clip", "1.0",
        "--scheduler", "plateau",
        "--class_weight_mode", protocol["class_weight_mode"],
        "--val_every_steps", str(protocol["val_every_steps"]),
        "--early_stop_patience", str(protocol["early_stop_patience"]),
        "--ckpt_every_steps", str(protocol["ckpt_every_steps"]),
        "--max_blocks_per_sample", str(protocol["max_blocks_per_sample"]),
        "--max_pages_per_sample", str(protocol["max_pages_per_sample"]),
        "--train_ratio", str(protocol["split"]["train_ratio"]),
        "--val_ratio", str(protocol["split"]["val_ratio"]),
        "--seed", str(run.get("seed", protocol["split"]["seed"])),
        "--num_workers", str(args.num_workers if args.num_workers is not None
                             else protocol.get("num_workers", 0)),
    ]
    if args.lr_patience is not None:
        a += ["--lr_patience", str(args.lr_patience)]
    if args.lr_min_lr is not None:
        a += ["--lr_min_lr", str(args.lr_min_lr)]
    # train.py uses BooleanOptionalAction, so the freeze state is always explicit
    # and a row can never silently inherit a default.
    text_freeze = bool(run.get("text_freeze", protocol["text_freeze"]))
    image_freeze = bool(run.get("image_freeze", protocol["image_freeze"]))
    a += ["--text_freeze" if text_freeze else "--no-text_freeze"]
    a += ["--image_freeze" if image_freeze else "--no-image_freeze"]
    preset = run.get("ablation_preset")
    if preset:
        a += ["--ablation_preset", preset]
    if cache_dir:
        a += ["--vit_cache_dir", cache_dir]
    # D1 = plan A: --init_from is never passed. Resume only continues a row that
    # this same protocol started.
    last = out_dir / "last.pt"
    if last.exists():
        a += ["--resume", str(last)]
    return a


def build_baseline_argv(run, protocol, args, out_dir: Path, epochs: int) -> list:
    a = [
        args.python, "scripts/paper/baselines.py",
        "--baseline", run["baseline"],
        "--output_dir", str(out_dir),
        "--data_dir", args.data_dir,
        "--label_config", args.label_config,
        "--device", args.device,
        "--epochs", str(epochs),
        "--seed", str(run.get("seed", protocol["split"]["seed"])),
        "--train_ratio", str(protocol["split"]["train_ratio"]),
        "--val_ratio", str(protocol["split"]["val_ratio"]),
    ]
    return a


def child_env(args: argparse.Namespace) -> dict:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = str(args.omp_threads)
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # Allocator fragmentation was half of the 2026-09-20 OOM loop; PyTorch
    # itself recommends this setting for a 12 GB card.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def summarize(out_dir: Path, run: dict, started: str, ended: str,
              exit_code: int, attempts: int, epochs_requested: int,
              cache_dir: str | None) -> dict:
    history = read_json(out_dir / "history.json", [])
    meta = read_json(out_dir / "run_meta.json", {})
    epochs = epoch_records(history)

    best_acc, best_f1, best_epoch = -1.0, 0.0, 0
    for entry in epochs:
        val = entry.get("val") or {}
        acc = float(val.get("accuracy") or 0.0)
        if acc > best_acc:
            best_acc, best_f1 = acc, float(val.get("macro_f1") or 0.0)
            best_epoch = int(entry.get("epoch") or 0)
    final_val = (epochs[-1].get("val") or {}) if epochs else {}

    completed = str(meta.get("status", "")) == "completed"
    if completed and exit_code != 0:
        status = "completed-nonzero-exit"
    elif completed:
        status = "completed"
    else:
        status = "crashed"

    return {
        "id": run["id"],
        "group": run.get("group", ""),
        "note": run.get("note", ""),
        "status": status,
        "exit_code": exit_code,
        "attempts": attempts,
        "started_at": started,
        "ended_at": ended,
        "ablation_preset": run.get("ablation_preset", ""),
        "ablation_signature": meta.get("ablation_signature", ""),
        "text_freeze": meta.get("text_freeze"),
        "image_freeze": meta.get("image_freeze"),
        "epochs_requested": epochs_requested,
        "epochs_completed": meta.get("epochs_completed", len(epochs)),
        "seed": run.get("seed"),
        "params_total": meta.get("params_total"),
        "params_trainable": meta.get("params_trainable"),
        "vit_cache": cache_dir or "",
        "val_best_accuracy": best_acc,
        "val_best_macro_f1": best_f1,
        "val_best_epoch": best_epoch,
        "val_selection": ("epoch with the highest val accuracy (= best_model.pt); "
                          "macro F1 read from the same epoch; reads epoch "
                          "boundaries only, never mid-epoch checks"),
        "val_last_accuracy": final_val.get("accuracy"),
        "val_last_macro_f1": final_val.get("macro_f1"),
        "mid_epoch_checks": len(history) - len(epochs),
        "peak_vram_mb": max((e.get("peak_vram_mb", 0.0) for e in epochs),
                            default=0.0),
        "history": str(out_dir / "history.json") if history else "",
    }


def main() -> int:
    args = parse_args()
    manifest = read_json(REPO / args.manifest, None)
    if manifest is None:
        raise SystemExit(f"cannot read manifest {args.manifest}")
    protocol = manifest["protocol"]
    if args.data_dir is None:
        args.data_dir = protocol["dataset"]

    cache_dir = args.vit_cache_dir
    if cache_dir is None:
        candidate = REPO / manifest["cache"]["default_dir"]
        cache_dir = str(candidate) if candidate.is_dir() else None
        if cache_dir is None:
            print("NOTE: no ViT cache found; every step will run ViT live "
                  "(~48 min slower per epoch). Build it with "
                  "scripts/paper/build_vit_cache.py or ship the directory.")

    runs = select_runs(manifest, args)
    output_root = REPO / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    summary = read_json(summary_path, [])
    done = {r["id"] for r in summary if r.get("status", "").startswith("completed")}

    print(f"manifest   {args.manifest}  ({manifest['name']})")
    print(f"data       {args.data_dir}")
    print(f"outputs    {output_root}")
    print(f"cache      {cache_dir or '(none)'}")
    print(f"workers    {args.num_workers if args.num_workers is not None else protocol.get('num_workers', 0)}")
    print(f"rows       {len(runs)}")
    for run in runs:
        tag = "  skip (completed)" if (run["id"] in done and not args.force) else ""
        print(f"  - {run['id']:<28} {run.get('group',''):<16}{tag}")
    print()

    results = []
    failures = 0
    for run in runs:
        if run["id"] in done and not args.force:
            print(f"skip {run['id']}: already completed (pass --force to redo)")
            continue

        epochs = resolve_epochs(run, protocol, args)
        out_dir = output_root / run["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        is_baseline = "baseline" in run
        build = build_baseline_argv if is_baseline else build_train_argv
        argv = (build(run, protocol, args, out_dir, epochs)
                if is_baseline else build(run, protocol, args, out_dir, epochs, cache_dir))

        if args.dry_run:
            print(f"[dry-run] {run['id']} (epochs={epochs})")
            print("  " + " ".join(argv))
            continue

        started = now_iso()
        t0 = time.time()
        exit_code = 1
        for attempt in range(1, args.max_attempts + 1):
            # Rebuild so a crashed attempt that left last.pt gets --resume.
            argv = (build(run, protocol, args, out_dir, epochs)
                    if is_baseline else build(run, protocol, args, out_dir, epochs, cache_dir))
            resuming = "--resume" in argv
            banner = (f"\n{'='*70}\n{run['id']}  attempt {attempt}/{args.max_attempts}  "
                      f"epochs={epochs}  {'RESUME' if resuming else 'COLD START'}\n{'='*70}")
            print(banner, flush=True)
            with open(out_dir / "train.log", "a", encoding="utf-8") as log:
                log.write(banner + "\n")
                log.write("[run] " + " ".join(argv) + "\n")
                log.flush()
                proc = subprocess.Popen(argv, cwd=str(REPO), env=child_env(args),
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True,
                                        encoding="utf-8", errors="replace",
                                        bufsize=1)
                # Tee the child LIVE.  Before 2026-10-03 this was
                # subprocess.run(stdout=PIPE), which buffered everything and
                # only wrote train.log after the child exited: a run is 10-22
                # hours long, so that made progress unwatchable and cost the
                # whole log whenever a row crashed.  history.json is still the
                # authoritative record; this is for monitoring.
                tail = []
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    tail.append(line)
                    if len(tail) > 200:
                        del tail[:100]
                proc.stdout.close()
                proc.wait()
                tail_text = "".join(tail)
            exit_code = proc.returncode
            print(tail_text[-4000:], flush=True)
            if exit_code == 0:
                break
            if attempt < args.max_attempts:
                print(f"exit={exit_code}; retrying in {args.retry_delay:.0f}s "
                      f"(a crash that left last.pt will resume from it)",
                      flush=True)
                time.sleep(args.retry_delay)

        ended = now_iso()
        result = summarize(out_dir, run, started, ended, exit_code, attempt,
                           epochs, cache_dir)
        result["wall_hours"] = round((time.time() - t0) / 3600.0, 3)
        with open(out_dir / "result.json", "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=1)

        summary = [r for r in summary if r.get("id") != run["id"]]
        summary.append(result)
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=1)

        print(f"done {run['id']}  status={result['status']} "
              f"attempts={attempt} wall={result['wall_hours']}h "
              f"last_acc={result['val_last_accuracy']} "
              f"best_acc={result['val_best_accuracy']}", flush=True)
        results.append(result)

        if result["status"] not in ("completed", "completed-nonzero-exit"):
            failures += 1
            if not args.continue_on_failure:
                print(f"stopping: {run['id']} did not complete "
                      f"(pass --continue_on_failure to keep going)")
                break

    if args.dry_run:
        return 0

    if summary:
        with open(output_root / "summary.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    print(f"\n{'id':<28}{'status':<22}{'epochs':>7}{'last_acc':>10}{'last_f1':>10}{'peakMB':>9}")
    for r in summary:
        print(f"{r['id']:<28}{r['status']:<22}{str(r.get('epochs_completed')):>7}"
              f"{_fmt(r.get('val_last_accuracy')):>10}"
              f"{_fmt(r.get('val_last_macro_f1')):>10}"
              f"{_fmt(r.get('peak_vram_mb')):>9}")
    print(f"\nsummary -> {summary_path}")
    print(f"summary -> {output_root / 'summary.csv'}")
    return 1 if failures else 0


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
