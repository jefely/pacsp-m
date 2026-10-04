"""Measure the packaging footprint and inspect the L4 timestamp proof.

Two numbers decide whether a one-click bootstrap is realistic, and both were being guessed
at: how large a bundled runtime would be, and whether the L4 timestamp is an actual
independent anchor or a local artefact.

The L4 question is the important one. A proof that was generated locally and never
submitted to a calendar proves nothing to a third party, and the record's own anchor field
says pending_bitcoin_confirmation. This reads the stamp to see what is actually there.
"""

import importlib.util
import json
import sys
from pathlib import Path

ID = Path(r"D:\myproject\PACSP-ID")
M = Path(r"D:\myproject\PACSP-M")


def dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main():
    print("=== dependency footprint ===")
    pkgs = {}
    for name in ("torch", "transformers", "sentence_transformers", "numpy",
                 "huggingface_hub", "tokenizers", "safetensors", "scipy",
                 "sklearn", "PIL"):
        spec = importlib.util.find_spec(name)
        if not spec or not spec.origin:
            pkgs[name] = None
            continue
        d = Path(spec.origin).parent
        if name == "torch":
            d = d.parent
        pkgs[name] = dir_size(d)
    total = 0
    for k, v in pkgs.items():
        if v is None:
            print(f"  {k:<24} not installed")
        else:
            total += v
            print(f"  {k:<24} {v / 2**20:>9.1f} MB")
    print(f"  {'TOTAL':<24} {total / 2**20:>9.1f} MB")

    print("\n=== embedding models available for bundling ===")
    for base in (M / "_hf_home" / "hub", ID / "_hf_home" / "hub"):
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            sz = dir_size(d)
            print(f"  {d.name.replace('models--', ''):<40} {sz / 2**20:>8.1f} MB"
                  f"   ({base.parent.parent.name})")

    print("\n=== L4 timestamp proof: is it an independent anchor? ===")
    recs = sorted((ID / "records").glob("*.pacsp"))
    if not recs:
        print("  no records found")
        return 0
    rec = json.loads(recs[0].read_text(encoding="utf-8"))
    l4 = rec.get("integrity", {}).get("L4", {})
    print(f"  record : {recs[0].name}")
    if not l4:
        print("  no integrity.L4 present")
        return 0
    for k, v in l4.items():
        if isinstance(v, str) and len(v) > 120:
            print(f"  {k:<18}: <{len(v)} chars> {v[:80]!r}...")
        else:
            print(f"  {k:<18}: {v}")

    print("\n  reading the stored proof bytes:")
    for key in ("proof", "ots_proof", "stamp", "proof_hex"):
        if key in l4:
            raw = l4[key]
            print(f"    key {key!r} present, {len(str(raw))} chars")
    # show the start of any base64 or hex blob
    for k, v in l4.items():
        if isinstance(v, str) and len(v) > 200:
            print(f"    {k} first 60: {v[:60]}")
            head = v[:16]
            if head.startswith("AE") or head.startswith("Ot"):
                print("      looks like an OpenTimestamps detached proof header")
            else:
                print(f"      header {head!r} is not the OTS magic")
    print("\n  OTS magic is b'\\x00OpenTimestamps\\x00\\x00' at the start of a .ots file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
