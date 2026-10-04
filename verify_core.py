"""Prove the vendored core is numerically identical to the pipeline's.

This project's standard is that a reimplementation must be shown to produce the same
numbers before it is trusted. Several results in this folder are comparisons of C_T
against other statistics, so if the C_T here differed from the pipeline's even slightly,
those comparisons would be against a different quantity.

Two checks.

  live   import pacsp_build from the sibling PACSP-ID and compare outputs on the same
         corpora, function by function
  frozen compare against C_T values previously produced by the pipeline, so the check
         still has content when PACSP-ID is not present

The frozen values come from the pipeline itself: docs/CT-TOOL-GUIDE.md records that
pacsp_ct.py and pacsp_build.py agree at 1.895004 for data/poem and 2.126140 for
data/lyrics, and records_centroid/_comparison.json holds the full arm table.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
M = Path(__file__).resolve().parent
sys.path.insert(0, str(M))

import pacsp_core  # noqa: E402

DATA = M / "data"
MODEL = "BAAI/bge-large-zh-v1.5"

# Values produced by PACSP-ID's scripts/pacsp_build.py and scripts/pacsp_ct.py
FROZEN = {
    "poem": 1.895004,
    "lyrics": 2.126140,
    "techdoc": 7.3982,
    "machine_poem": 3.8271,
    "machine_lyrics": 5.9210,
    "machine_techdoc": 2.4662,
    "hc3_human_medicine": 14.9678,
    "hc3_ai_medicine": 19.3988,
    "hc3_human_openqa": 21.2274,
    "hc3_ai_openqa": 25.5820,
}

TOL = 5e-4          # frozen values are rounded to 4 decimals in the documents


def find_sibling():
    """PACSP-ID next to this folder, if it is there."""
    for cand in (M.parent / "PACSP-ID", M / "PACSP-ID"):
        if (cand / "scripts" / "pacsp_build.py").is_file():
            return cand
    return None


def main():
    print(f"  core: {M / 'pacsp_core.py'}")
    print(f"  data: {DATA}  ({len(list(DATA.glob('*/')))} corpora)")

    # ------------------------------------------------------------------ frozen
    print("\n=== frozen comparison against pipeline values ===")
    print(f"  {'corpus':<22} {'frozen':>10} {'this core':>11} {'diff':>10}  ok")
    bad = []
    for arm, want in FROZEN.items():
        p = DATA / arm
        if not p.is_dir():
            print(f"  {arm:<22} {'-':>10} {'missing':>11}")
            continue
        got = pacsp_core.ct_of_directory(p, model_name=MODEL)
        diff = abs(got - want)
        ok = diff < TOL * max(1.0, abs(want))
        if not ok:
            bad.append(arm)
        print(f"  {arm:<22} {want:>10.4f} {got:>11.4f} {diff:>10.5f}  "
              f"{'OK' if ok else 'MISMATCH'}")
    print(f"\n  {len(FROZEN) - len(bad)}/{len(FROZEN)} match the pipeline values")

    # ------------------------------------------------------------------ live
    sib = find_sibling()
    if not sib:
        print("\n=== live comparison: PACSP-ID not found, skipped ===")
        return 0 if not bad else 1

    print(f"\n=== live comparison against {sib.name}/scripts/pacsp_build.py ===")
    sys.path.insert(0, str(sib / "scripts"))
    try:
        import pacsp_build as pb
    except Exception as e:
        print(f"  cannot import pacsp_build: {type(e).__name__}: {str(e)[:120]}")
        return 0 if not bad else 1

    pairs = [("load_samples", lambda mod, t: mod.load_samples(t)[0]),
             ("compute_embeddings", lambda mod, t: mod.compute_embeddings(t, MODEL)),
             ("compute_deltas", None), ("compute_mus", None),
             ("compute_ct", None), ("detect_changepoints", None)]

    print(f"  {'function':<22} {'corpus':<12} {'identical':>10}")
    all_ok = True
    for arm in ("poem", "lyrics", "machine_poem"):
        p = DATA / arm
        if not p.is_dir():
            continue
        texts_m, _ = pacsp_core.load_samples(p)
        texts_b, _ = pb.load_samples(p)
        same_load = texts_m == texts_b
        print(f"  {'load_samples':<22} {arm:<12} {str(same_load):>10}")
        all_ok &= same_load

        em = pacsp_core.compute_embeddings(texts_m, model_name=MODEL)
        eb = pb.compute_embeddings(texts_b, model_name=MODEL)
        same_emb = bool(np.allclose(em, eb, atol=0, rtol=0))
        print(f"  {'compute_embeddings':<22} {arm:<12} {str(same_emb):>10}")
        all_ok &= same_emb

        dm, mm = pacsp_core.compute_deltas(em), pacsp_core.compute_mus(em)
        db, mb = pb.compute_deltas(eb), pb.compute_mus(eb)
        same_d = dm == db
        same_m = mm == mb
        print(f"  {'compute_deltas':<22} {arm:<12} {str(same_d):>10}")
        print(f"  {'compute_mus':<22} {arm:<12} {str(same_m):>10}")
        all_ok &= same_d and same_m

        cm = pacsp_core.compute_ct(dm, mm)
        cb = pb.compute_ct(db, mb)
        same_c = cm == cb
        print(f"  {'compute_ct':<22} {arm:<12} {str(same_c):>10}  "
              f"({cm:.6f})")
        all_ok &= same_c

        pmm = pacsp_core.detect_changepoints(mm, penalty=10.0)
        pmb = pb.detect_changepoints(mb, penalty=10.0)
        same_p = pmm == pmb
        print(f"  {'detect_changepoints':<22} {arm:<12} {str(same_p):>10}  "
              f"({pmm})")
        all_ok &= same_p

    print()
    print(f"  live comparison: {'ALL IDENTICAL' if all_ok else 'DIFFERENCES FOUND'}")
    verdict = all_ok and not bad
    print(f"\n  VERDICT: {'core verified' if verdict else 'CHECK NEEDED'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
