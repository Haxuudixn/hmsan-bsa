"""Summarise the path-3 comparison matrix (Phase 0, item W3 / E0.5).

Reads the per-row ``result.json`` / ``history.json`` / ``run_meta.json`` written
by ``run_comparison.ps1`` plus the GPU telemetry CSV, and emits the paper's
ablation table, the freeze-2x2 parameter-efficiency table, a per-class F1 table
and a protocol audit.  Nothing here trains or evaluates: it only aggregates.

    python scripts\\paper\\summarize_comparison.py
    python scripts\\paper\\summarize_comparison.py --matrix_dir outputs\\path3_matrix

Outputs (default ``docs\\paper_materials``):

    path3_summary.json    machine-readable, one record per row
    path3_summary.md      the report, including the protocol audit
    path3_tables.tex      LaTeX for table 2 (ablations) and table 4 (freezes)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

FREEZE_CELLS = {
    ("frozen_both"): ("TT", "both frozen"),
    ("full_3ep"): ("FT", "text tuned, image frozen"),
    ("image_unfrozen"): ("TF", "text frozen, image tuned"),
    ("both_unfrozen"): ("FF", "both tuned"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--matrix_dir", default=str(REPO_ROOT / "outputs" / "path3_matrix")
    )
    parser.add_argument(
        "--manifest", default=str(REPO_ROOT / "experiments" / "path3_manifest.json")
    )
    parser.add_argument("--out_dir", default=str(REPO_ROOT / "docs" / "paper_materials"))
    parser.add_argument("--reference", default="full_3ep", help="row used for delta columns")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 - a broken row must not stop the report
        print(f"  WARNING: cannot read {path}: {exc}")
        return None


def load_manifest(path: Path) -> dict[str, dict]:
    data = read_json(path) or {}
    return {entry["id"]: entry for entry in data.get("runs", [])}


def load_rows(matrix_dir: Path, manifest: dict[str, dict]) -> list[dict]:
    rows = []
    for result_path in sorted(matrix_dir.glob("*/result.json")):
        result = read_json(result_path)
        if not result:
            continue
        run_dir = result_path.parent
        result["run_dir"] = str(run_dir)
        result["manifest"] = manifest.get(result["id"], {})
        history = read_json(run_dir / "history.json") or []
        result["history"] = history if isinstance(history, list) else [history]
        result["run_meta"] = read_json(run_dir / "run_meta.json") or {}
        result["order"] = result["manifest"].get("order", 999)
        rows.append(result)
    rows.sort(key=lambda item: (item["order"], item["id"]))
    return rows


def monitor_peaks(csv_path: Path, started: str, ended: str) -> dict:
    """Peak VRAM / temperature / power inside one row's time window."""
    if not csv_path.is_file() or not started or not ended:
        return {}
    peak = {"vram_mb": 0.0, "temp_c": 0.0, "power_w": 0.0, "samples": 0}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            stamp = (record.get("timestamp") or "").strip()
            if not stamp or stamp < started or stamp > ended:
                continue
            peak["samples"] += 1
            for key, column in (
                ("vram_mb", "used_mb"),
                ("temp_c", "temp_c"),
                ("power_w", "power_w"),
            ):
                try:
                    peak[key] = max(peak[key], float(record[column]))
                except (KeyError, TypeError, ValueError):
                    pass
    return peak


def pp(value) -> str:
    return "-" if value is None else f"{value * 100:.2f}"


def delta(row_value, ref_value):
    if row_value is None or ref_value is None:
        return None
    return row_value - ref_value


def label_for(row: dict) -> str:
    return row["manifest"].get("note") or row.get("ablation_preset") or row["id"]


def build_records(rows: list[dict], monitor_csv: Path, reference_id: str):
    reference = next((row for row in rows if row["id"] == reference_id), None)
    records = []
    for row in rows:
        history = row["history"]
        last_val = history[-1]["val"] if history else {}
        peaks = monitor_peaks(monitor_csv, row.get("started_at"), row.get("ended_at"))
        records.append({
            "id": row["id"],
            "group": row.get("group", ""),
            "note": row.get("note", ""),
            "label": label_for(row),
            "status": row.get("status", "unknown"),
            "order": row["order"],
            "ablation_preset": row.get("ablation_preset"),
            "ablation_signature": row.get("ablation_signature"),
            "text_freeze": row.get("text_freeze"),
            "image_freeze": row.get("image_freeze"),
            "seed": row.get("seed"),
            "epochs_requested": row.get("epochs_requested"),
            "epochs_completed": row.get("epochs_completed"),
            "params_total": row.get("params_total"),
            "params_trainable": row.get("params_trainable"),
            "vit_cache": row.get("vit_cache") or None,
            "val_best_accuracy": row.get("val_best_accuracy"),
            "val_best_macro_f1": row.get("val_best_macro_f1"),
            "val_final_accuracy": row.get("val_final_accuracy"),
            "val_final_macro_f1": row.get("val_final_macro_f1"),
            "val_final_per_class": {
                str(c): last_val.get(f"f1_{c}") for c in range(19)
            },
            "val_support": {str(c): last_val.get(f"support_{c}") for c in range(19)},
            "peak_vram_mb": row.get("peak_vram_mb"),
            "monitor_peak_vram_mb": peaks.get("vram_mb"),
            "monitor_peak_temp_c": peaks.get("temp_c"),
            "monitor_peak_power_w": peaks.get("power_w"),
            "monitor_samples": peaks.get("samples"),
            "wall_hours": row.get("wall_hours"),
            "est_hours": row.get("est_hours"),
            "attempts": row.get("attempts"),
            "exit_code": row.get("exit_code"),
            "run_dir": row["run_dir"],
            "delta_accuracy_vs_reference": delta(
                row.get("val_best_accuracy"),
                reference and reference.get("val_best_accuracy"),
            ),
            "delta_macro_f1_vs_reference": delta(
                row.get("val_best_macro_f1"),
                reference and reference.get("val_best_macro_f1"),
            ),
        })
    return records


def audit(rows: list[dict], manifest: dict[str, dict], records: list[dict]) -> list[str]:
    """Protocol checks that must hold before the numbers mean anything."""
    warnings = []
    metas = {row["id"]: row["run_meta"] for row in rows if row["run_meta"]}

    for field in ("split", "caps", "optimizer"):
        seen = {}
        for run_id, meta in metas.items():
            key = json.dumps(meta.get(field), sort_keys=True)
            seen.setdefault(key, []).append(run_id)
        if len(seen) > 1:
            warnings.append(f"{field} differs between rows: {seen}")

    signatures = {}
    for record in records:
        signatures.setdefault(record["ablation_signature"] or "(missing)", []).append(record["id"])
    for signature, ids in signatures.items():
        if len(ids) > 1:
            warnings.append(
                f"ablation signature {signature!r} used by {len(ids)} rows: {ids} "
                "(expected only for the planned tied presets A1/A2, A3, A4, A9/A10)"
            )

    expected = set(manifest)
    missing = sorted(expected - {record["id"] for record in records})
    if missing:
        warnings.append(f"{len(missing)} manifest row(s) have no result yet: {missing}")

    failed = [record["id"] for record in records if record["status"] != "completed"]
    if failed:
        warnings.append(f"{len(failed)} row(s) not completed: {failed}")

    partial = [
        record["id"] for record in records
        if record["epochs_completed"] is not None
        and record["epochs_requested"] is not None
        and record["epochs_completed"] < record["epochs_requested"]
    ]
    if partial:
        warnings.append(f"row(s) with fewer epochs than requested: {partial}")

    seeds = {record["seed"] for record in records if record["seed"] is not None}
    if len(seeds) > 1:
        warnings.append(f"rows were trained with different seeds: {sorted(seeds)}")

    warnings.append(
        "single seed per row: the table carries no error bars.  Differences below "
        "the epoch-to-epoch swing of one run are not interpretable; report this in "
        "the limitations section or add seeds 43/44 later."
    )
    return warnings


def markdown_table(headers: list[str], body: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_markdown(records, warnings, class_names, reference_id, matrix_dir) -> str:
    lines = [
        "# 路径 3 消融矩阵汇总",
        "",
        f"- 来源：`{matrix_dir}`",
        f"- 行数：{len(records)}（完成 {sum(1 for r in records if r['status'] == 'completed')}）",
        f"- 参照行：`{reference_id}`",
        "- 说明：所有指标取自 **val**（测试集在矩阵期间不使用，见协议锁定）。",
        "",
        "## 0 协议审计",
        "",
    ]
    for warning in warnings:
        lines.append(f"- [warn] {warning}")
    if not warnings:
        lines.append("- 无异常。")

    lines += ["", "## 1 表 2 · 架构消融（val，3 epoch，budget-matched）", ""]
    body = []
    for record in [r for r in records if r["group"] == "architecture"]:
        body.append([
            f"`{record['id']}`",
            record["label"].replace("|", "/"),
            f"{record['params_trainable']:,}" if record["params_trainable"] else "-",
            pp(record["val_best_accuracy"]),
            pp(record["val_best_macro_f1"]),
            f"{record['delta_accuracy_vs_reference'] * 100:+.2f}"
            if record["delta_accuracy_vs_reference"] is not None else "-",
            f"{record['delta_macro_f1_vs_reference'] * 100:+.2f}"
            if record["delta_macro_f1_vs_reference"] is not None else "-",
            f"{record['wall_hours']:.2f}" if record["wall_hours"] else "-",
            f"{record['peak_vram_mb']:.0f}" if record["peak_vram_mb"] else "-",
        ])
    lines += markdown_table(
        ["配置", "消融内容", "可训练参数", "val acc (%)", "val macro F1 (%)",
         "Δacc (pp)", "ΔF1 (pp)", "wall (h)", "峰值显存 (MB)"],
        body,
    )

    lines += ["", "## 2 表 4 · 冻结 2×2 参数效率", ""]
    body = []
    for record in [r for r in records if r["group"] == "freeze_2x2"]:
        cell, description = FREEZE_CELLS.get(record["id"], ("?", record["id"]))
        body.append([
            cell,
            description,
            f"`{record['id']}`",
            f"{record['params_trainable']:,}" if record["params_trainable"] else "-",
            pp(record["val_best_accuracy"]),
            pp(record["val_best_macro_f1"]),
            f"{record['wall_hours']:.2f}" if record["wall_hours"] else "-",
        ])
    body.sort(key=lambda item: ["TT", "FT", "TF", "FF"].index(item[0]) if item[0] in
              ["TT", "FT", "TF", "FF"] else 9)
    lines += markdown_table(
        ["格", "含义", "配置", "可训练参数", "val acc (%)", "val macro F1 (%)", "wall (h)"],
        body,
    )

    lines += ["", "## 3 逐类 F1（val，最后一轮）", ""]
    completed = [r for r in records if r["status"] == "completed"]
    headers = ["class"] + [r["id"] for r in completed]
    body = []
    for class_id in range(len(class_names)):
        name = class_names[class_id]
        support = next(
            (r["val_support"].get(str(class_id)) for r in completed
             if r["val_support"].get(str(class_id))), None)
        row = [f"{class_id} {name}" + (f" (n={support})" if support else "")]
        for record in completed:
            value = record["val_final_per_class"].get(str(class_id))
            row.append(f"{value:.3f}" if isinstance(value, (int, float)) else "-")
        body.append(row)
    lines += markdown_table(headers, body)

    lines += ["", "## 4 GPU 遥测（monitor.csv 窗口内峰值）", ""]
    body = []
    for record in records:
        body.append([
            f"`{record['id']}`",
            f"{record['monitor_peak_vram_mb']:.0f}" if record["monitor_peak_vram_mb"] else "-",
            f"{record['monitor_peak_temp_c']:.0f}" if record["monitor_peak_temp_c"] else "-",
            f"{record['monitor_peak_power_w']:.0f}" if record["monitor_peak_power_w"] else "-",
            str(record["monitor_samples"] or 0),
            f"{record['wall_hours']:.2f}" if record["wall_hours"] else "-",
            f"{record['est_hours']}" if record["est_hours"] else "-",
            str(record["attempts"] or "-"),
        ])
    lines += markdown_table(
        ["配置", "峰值显存 (MB)", "峰值温度 (℃)", "峰值功耗 (W)", "采样数", "实测 wall (h)",
         "预估 (h)", "尝试次数"],
        body,
    )
    lines += ["", "## 5 使用说明", "",
              "- 本文件由 `scripts\\paper\\summarize_comparison.py` 生成，请勿手改。",
              "- 表中 `Δ` 一律相对参照行 `" + reference_id + "`，单位 pp。",
              "- 提交前需在论文中声明：测试集未参与任何配置或检查点选择。"]
    return "\n".join(lines) + "\n"


def render_latex(records, reference_id) -> str:
    def esc(text: str) -> str:
        return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    lines = [
        "% Generated by scripts/paper/summarize_comparison.py - do not edit by hand.",
        "% Delta columns are percentage points against the budget-matched reference.",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Ablation study on the validation split (3 epochs, budget-matched, "
        r"text encoder fine-tuned, image encoder frozen). $\Delta$ is in percentage "
        r"points against \texttt{" + esc(reference_id) + r"}.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Row & Ablation & Trainable & Acc & Macro-F1 & $\Delta$Acc & $\Delta$F1 & Wall (h) \\",
        r"\midrule",
    ]
    for record in [r for r in records if r["group"] == "architecture"]:
        lines.append(" & ".join([
            r"\texttt{" + esc(record["id"]) + "}",
            esc(record["label"]),
            f"{record['params_trainable']:,}" if record["params_trainable"] else "--",
            pp(record["val_best_accuracy"]),
            pp(record["val_best_macro_f1"]),
            f"{record['delta_accuracy_vs_reference'] * 100:+.2f}"
            if record["delta_accuracy_vs_reference"] is not None else "--",
            f"{record['delta_macro_f1_vs_reference'] * 100:+.2f}"
            if record["delta_macro_f1_vs_reference"] is not None else "--",
            f"{record['wall_hours']:.2f}" if record["wall_hours"] else "--",
        ]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "", r"\begin{table}[t]",
              r"\centering", r"\small",
              r"\caption{Freeze matrix: trainable parameters against validation quality. "
              r"TT/FT/TF/FF = image frozen or tuned $\times$ text frozen or tuned.}",
              r"\label{tab:freeze}", r"\begin{tabular}{llrrr}", r"\toprule",
              r"Cell & Configuration & Trainable & Acc & Macro-F1 \\", r"\midrule"]
    ordered = [r for r in records if r["group"] == "freeze_2x2"]
    ordered.sort(key=lambda item: ["TT", "FT", "TF", "FF"].index(
        FREEZE_CELLS.get(item["id"], ("?", ""))[0]))
    for record in ordered:
        cell, description = FREEZE_CELLS.get(record["id"], ("?", record["id"]))
        lines.append(" & ".join([
            cell, esc(description), r"\texttt{" + esc(record["id"]) + "}",
            f"{record['params_trainable']:,}" if record["params_trainable"] else "--",
            pp(record["val_best_accuracy"]), pp(record["val_best_macro_f1"]),
        ]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def class_names() -> list[str]:
    try:
        from bid_slicing.data.label_schema import LabelSchema

        schema = LabelSchema.from_yaml(str(REPO_ROOT / "configs" / "labels.yaml"))
        return [schema.decode_class(c) or f"class_{c}" for c in range(schema.num_classes)]
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: cannot load the label schema ({exc}); using ids")
        return [f"class_{c}" for c in range(19)]


def main() -> int:
    args = parse_args()
    matrix_dir = Path(args.matrix_dir)
    monitor_csv = matrix_dir / "monitor.csv"
    manifest = load_manifest(Path(args.manifest))
    rows = load_rows(matrix_dir, manifest)
    if not rows:
        raise SystemExit(f"no */result.json under {matrix_dir}")

    records = build_records(rows, monitor_csv, args.reference)
    warnings = audit(rows, manifest, records)
    names = class_names()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "path3_summary.json").write_text(
        json.dumps({
            "matrix_dir": str(matrix_dir),
            "reference": args.reference,
            "rows": records,
            "warnings": warnings,
        }, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    (out_dir / "path3_summary.md").write_text(
        render_markdown(records, warnings, names, args.reference, matrix_dir),
        encoding="utf-8",
    )
    (out_dir / "path3_tables.tex").write_text(
        render_latex(records, args.reference), encoding="utf-8"
    )

    if not args.quiet:
        print(f"{'row':<24}{'status':<12}{'trainable':>12}{'val_acc':>9}{'val_f1':>9}{'Δacc':>8}{'wall_h':>8}")
        for record in records:
            trainable = f"{record['params_trainable']:,}" if record["params_trainable"] else "-"
            delta_acc = (
                f"{record['delta_accuracy_vs_reference'] * 100:+.2f}"
                if record["delta_accuracy_vs_reference"] is not None else "-"
            )
            wall = f"{record['wall_hours']:.2f}" if record["wall_hours"] else "-"
            print(f"{record['id']:<24}{record['status']:<12}{trainable:>12}"
                  f"{pp(record['val_best_accuracy']):>9}{pp(record['val_best_macro_f1']):>9}"
                  f"{delta_acc:>8}{wall:>8}")
        print()
        for warning in warnings:
            print(f"  [warn] {warning}")
    print(f"\nwrote {out_dir / 'path3_summary.json'}")
    print(f"wrote {out_dir / 'path3_summary.md'}")
    print(f"wrote {out_dir / 'path3_tables.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())