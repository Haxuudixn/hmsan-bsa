"""Check that the E0.3 ablation plumbing is actually connected end to end.

Covers three things the model-level verifier (verify_ablation.py) does not:
  1. train.py resolves --ablation_preset / --ablation_set into AblationFlags,
  2. those flags are handed to HMSAN_BSA (a resolved preset really does change
     the module layout),
  3. experiments.py maps every experiment name onto a distinct preset.

Usage:
    python scripts/paper/check_ablation_plumbing.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import types

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from bid_slicing.models.hmsan_bsa import HMSAN_BSA  # noqa: E402

src = (REPO_ROOT / "train.py").read_text(encoding="utf-8").split("def main()")[0]
train_head = types.ModuleType("train_head")
train_head.__file__ = str(REPO_ROOT / "train.py")
exec(compile(src, str(REPO_ROOT / "train.py"), "exec"), train_head.__dict__)


def ns(preset=None, aset=None):
    return argparse.Namespace(ablation_preset=preset, ablation_set=aset)


def main():
    resolve = train_head.resolve_ablation_flags

    print("1) CLI -> AblationFlags")
    cases = [
        ("default", ns()),
        ("preset A11", ns(preset="A11_no_gates")),
        ("override only", ns(aset=["g2_value_gate=false", "local_window=7", "gated_ffn=off"])),
        ("preset+override", ns(preset="A6_no_page_memory", aset=["gated_ffn=off"])),
    ]
    for label, args in cases:
        print(f"   {label:<16} -> {resolve(args).signature()}")

    assert resolve(ns()).is_default()
    assert not resolve(ns(preset="A11_no_gates")).is_default()
    g2_off = resolve(ns(aset=["g2_value_gate=false"]))
    assert g2_off.g2_value_gate is False, "override did not take effect"
    assert g2_off.g1_output_gate is True, "unrelated switch should be untouched"
    assert g2_off.attention_mode == "gsa"
    assert g2_off.is_default() is False

    print("2) invalid input is rejected")
    for bad, label in (("bogus=1", "unknown key"), ("no_equals", "bad syntax")):
        try:
            resolve(ns(aset=[bad]))
        except SystemExit as exc:
            print(f"   {label:<16} -> {exc}")
        else:
            raise AssertionError(f"{bad!r} should have been rejected")

    print("3) resolved flags reach the model")
    flags = resolve(ns(preset="A7_no_boundary_gate"))
    model = HMSAN_BSA(text_freeze=True, image_freeze=True, flags=flags)
    assert model.flags.signature() == flags.signature()
    assert not hasattr(model.section_memory, "boundary_gate")
    print("   A7 -> section_memory has no boundary_gate")

    print("4) experiments.py maps every experiment to a distinct preset")
    import experiments as exp_mod

    signatures = {}
    for name, cfg in exp_mod.EXPERIMENTS.items():
        signature = cfg["flags"].signature()
        print(f"   {name:<18} preset={cfg['preset']:<20} {signature}")
        assert signature not in signatures, (
            f"{name} and {signatures[signature]} resolve to the same architecture"
        )
        signatures[signature] = name
    assert len(signatures) == len(exp_mod.EXPERIMENTS)

    print("\nOK - ablation plumbing is connected end to end")


if __name__ == "__main__":
    main()
