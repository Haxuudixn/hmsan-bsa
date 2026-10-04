"""Shrink finished matrix-run output directories (Phase 0, item W4).

A finished row keeps three large files, and the matrix has 15 rows:

  * ``last.pt``          model + optimizer state, needed to resume a crash
  * ``best_model.pt``    model + optimizer state, but evaluation only ever
                         reads ``model_state_dict``
  * ``checkpoint_epochN.pt`` every 5th epoch, pure redundancy for a 3-epoch row

Measured sizes for a 104.78M-trainable model: ~1.60 GB per checkpoint, of which
~0.83 GB is AdamW state.  Pruning ``best_model.pt`` alone frees ~12 GB over the
matrix; dropping periodic checkpoints and the ``.tmp`` leftover frees the rest.

Refuses to touch a directory whose ``run_meta.json`` says the run is not
``completed`` unless ``--force`` is given, so a crashed row stays resumable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run_dir", required=True)
    parser.add_argument(
        "--keep_last_optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep last.pt fully intact so a finished row can still be resumed",
    )
    parser.add_argument(
        "--keep_periodic",
        action="store_true",
        help="keep checkpoint_epochN.pt files (default: delete them)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"not a directory: {run_dir}")

    meta_path = run_dir / "run_meta.json"
    status = "unknown"
    if meta_path.is_file():
        try:
            status = json.loads(meta_path.read_text(encoding="utf-8")).get("status", "unknown")
        except Exception:
            status = "unreadable"
    if status != "completed" and not args.force:
        print(f"skip {run_dir.name}: run_meta.json status={status!r} (use --force to override)")
        return 0

    before = dir_size_bytes(run_dir)
    freed = 0
    actions = []

    for tmp in run_dir.glob("*.tmp"):
        freed += tmp.stat().st_size
        actions.append(f"delete {tmp.name}")
        if not args.dry_run:
            tmp.unlink()

    if not args.keep_periodic:
        for ckpt in sorted(run_dir.glob("checkpoint_epoch*.pt")):
            freed += ckpt.stat().st_size
            actions.append(f"delete {ckpt.name}")
            if not args.dry_run:
                ckpt.unlink()

    targets = [run_dir / "best_model.pt"]
    if not args.keep_last_optimizer:
        targets.append(run_dir / "last.pt")
    for path in targets:
        if not path.is_file():
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "optimizer_state_dict" not in checkpoint:
            continue
        size_before = path.stat().st_size
        checkpoint.pop("optimizer_state_dict", None)
        checkpoint["optimizer_state_pruned"] = True
        if not args.dry_run:
            tmp_path = path.with_suffix(path.suffix + ".prune.tmp")
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, path)
            size_after = path.stat().st_size
        else:
            size_after = size_before
        freed += size_before - size_after
        actions.append(
            f"prune optimizer state from {path.name} "
            f"({size_before / 2**30:.2f} -> {size_after / 2**30:.2f} GiB)"
        )
        del checkpoint

    after = dir_size_bytes(run_dir) if not args.dry_run else before
    print(f"[prune] {run_dir.name}: status={status}")
    for action in actions:
        print(f"  - {action}")
    if not actions:
        print("  (nothing to do)")
    print(
        f"[prune] {before / 2**30:.2f} GiB -> {after / 2**30:.2f} GiB "
        f"(freed {freed / 2**30:.2f} GiB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())