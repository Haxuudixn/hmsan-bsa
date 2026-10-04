"""Fingerprint the training corpus so a 37 GB transfer can be verified.

``scripts/verify_split_clean.py`` publishes ``zips_sha256_16``, but that hash is
built from each ZIP's *name and size* - it catches a missing or renamed export,
not a truncated or corrupted one.  37 GB moving to a rented machine deserves a
content check, so this writes a per-file SHA-256 manifest that can be produced
independently on the sending and receiving side and compared.

Usage
-----
    # on the machine that has the data
    python scripts/paper/data_manifest.py --data_dir "<final_train>" \
        --out data_manifest_local.json

    # on the rented machine, after the upload
    python scripts/paper/data_manifest.py --data_dir /data/final_train \
        --out data_manifest_cloud.json

    # then compare
    python scripts/paper/data_manifest.py --compare \
        data_manifest_local.json data_manifest_cloud.json

The file list, the total byte count and the aggregate digest must all match.
Reading 37 GB takes a few minutes; that is cheap next to one epoch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PATTERN = "training_data_*.zip"
CHUNK = 8 << 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--out", default="data_manifest.json")
    p.add_argument("--compare", nargs=2, default=None, metavar=("A", "B"))
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build(data_dir: Path, quiet: bool) -> dict:
    files = sorted(zip_path for zip_path in data_dir.glob(PATTERN))
    if not files:
        raise SystemExit(f"no {PATTERN} under {data_dir}")
    entries = []
    aggregate = hashlib.sha256()
    total = 0
    for i, path in enumerate(files, 1):
        size = path.stat().st_size
        digest = sha256_file(path)
        total += size
        aggregate.update(path.name.encode("utf-8"))
        aggregate.update(digest.encode("utf-8"))
        entries.append({"name": path.name, "bytes": size, "sha256": digest})
        if not quiet:
            print(f"  [{i:>2}/{len(files)}] {path.name}  {size/2**20:8.1f} MiB  "
                  f"{digest[:16]}", flush=True)
    return {
        "schema": 1,
        "data_dir": str(data_dir),
        "pattern": PATTERN,
        "file_count": len(files),
        "total_bytes": total,
        "total_gib": round(total / 2 ** 30, 3),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": entries,
    }


def compare(path_a: str, path_b: str) -> int:
    a = json.loads(Path(path_a).read_text(encoding="utf-8"))
    b = json.loads(Path(path_b).read_text(encoding="utf-8"))
    ok = True
    for key in ("file_count", "total_bytes", "aggregate_sha256"):
        same = a[key] == b[key]
        ok = ok and same
        print(f"{'OK  ' if same else 'DIFF'} {key:<18} {a[key]} vs {b[key]}")
    by_name_a = {f["name"]: f["sha256"] for f in a["files"]}
    by_name_b = {f["name"]: f["sha256"] for f in b["files"]}
    for name in sorted(set(by_name_a) | set(by_name_b)):
        if name not in by_name_a:
            ok = False
            print(f"DIFF only on the right: {name}")
        elif name not in by_name_b:
            ok = False
            print(f"DIFF only on the left:  {name}")
        elif by_name_a[name] != by_name_b[name]:
            ok = False
            print(f"DIFF content differs:   {name}")
    print("\nIDENTICAL" if ok else "\nMISMATCH - do not train on this copy")
    if ok:
        print("next: python scripts/verify_split_clean.py --data_dir <DATA_DIR> "
              "and check zips_sha256_16 == aa3cb23bcddcc621")
    return 0 if ok else 1


def main() -> int:
    args = parse_args()
    if args.compare:
        return compare(*args.compare)
    if not args.data_dir:
        raise SystemExit("--data_dir is required unless --compare is used")
    manifest = build(Path(args.data_dir), args.quiet)
    out = Path(args.out)
    out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n{manifest['file_count']} files, {manifest['total_gib']} GiB")
    print(f"aggregate sha256: {manifest['aggregate_sha256']}")
    print(f"manifest -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
