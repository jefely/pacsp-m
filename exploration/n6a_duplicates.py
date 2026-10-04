"""N6a: is the techdoc reversal an artefact of duplicate texts?

Inspecting the corpus showed techdoc/001.txt and techdoc/002.txt have identical opening
sentences, and the arm's byte distribution is 3276 to 65281 against machine_techdoc's 690
to 3101, a factor of 8.3 in the means. Both facts threaten the section 4.5 reversal.

Duplicates matter specifically for D. D is the mean of all pairwise distances, and the
distance between two copies of the same text is near zero, so repeated items pull the
mean down. If the human arm is more repetitive than the machine arm, then the human arm
having the higher D in techdoc cannot be read as "more dispersed" without first removing
the effect of exact repeats.

This measures: how many exact duplicates each arm contains, what D is with duplicates
collapsed to one copy, and whether the reversal survives.
"""

import hashlib
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
M = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(M))
import pacsp_core  # noqa: E402

DATA = M / "data"
MODEL = "BAAI/bge-large-zh-v1.5"
ARMS = ["techdoc", "machine_techdoc", "poem", "machine_poem",
        "lyrics", "machine_lyrics"]


def sq(A, B):
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def mpd(E):
    n = len(E)
    if n < 2:
        return float("nan")
    iu = np.triu_indices(n, k=1)
    return float(np.sqrt(sq(E, E)[iu]).mean())


def near_duplicate_groups(texts, threshold=0.98):
    """Group texts whose character 3-gram sets overlap above the threshold."""
    grams = [set(t[i:i + 3] for i in range(max(1, len(t) - 2)))
             for t in texts]
    n = len(texts)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if not grams[i] or not grams[j]:
                continue
            inter = len(grams[i] & grams[j])
            jac = inter / (len(grams[i]) + len(grams[j]) - inter)
            if jac >= threshold:
                union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1], groups


def main():
    out = {}
    print("=== duplicate structure ===")
    print(f"  {'arm':<20} {'n':>4} {'exact dup':>10} {'near-dup groups':>16} "
          f"{'largest group':>14}")
    for arm in ARMS:
        p = DATA / arm
        if not p.is_dir():
            continue
        texts, _ = pacsp_core.load_samples(p)
        exact = Counter(hashlib.sha256(t.strip().encode()).hexdigest()
                        for t in texts)
        n_exact_dup = sum(c - 1 for c in exact.values() if c > 1)
        groups, allg = near_duplicate_groups(texts)
        largest = max((len(g) for g in groups), default=1)
        print(f"  {arm:<20} {len(texts):>4} {n_exact_dup:>10} {len(groups):>16} "
              f"{largest:>14}")
        out.setdefault("duplicates", {})[arm] = {
            "n": len(texts), "exact_duplicate_extras": n_exact_dup,
            "near_dup_groups": len(groups), "largest_group": largest,
        }

    # ------------------------------------------------------------------ effect on D
    print()
    print("=== effect of collapsing duplicates on D ===")
    print(f"  {'arm':<20} {'D all':>10} {'D dedup':>10} {'change':>9} "
          f"{'n dedup':>8}")
    for arm in ARMS:
        p = DATA / arm
        if not p.is_dir():
            continue
        texts, _ = pacsp_core.load_samples(p)
        E_all = pacsp_core.compute_embeddings(texts, model_name=MODEL)
        d_all = mpd(E_all)
        # collapse: keep one representative per near-duplicate group
        _, allg = near_duplicate_groups(texts)
        keep = sorted(rep[0] for rep in allg.values())
        E_dd = E_all[keep]
        d_dd = mpd(E_dd)
        ch = d_dd / d_all if d_all else float("nan")
        print(f"  {arm:<20} {d_all:>10.4f} {d_dd:>10.4f} {ch:>9.4f} {len(keep):>8}")
        out.setdefault("dedup_effect", {})[arm] = {
            "D_all": round(d_all, 4), "D_dedup": round(d_dd, 4),
            "change": round(float(ch), 4), "n_dedup": len(keep)}

    # ------------------------------------------------------------------ verdict
    print()
    print("=== does the techdoc reversal survive? ===")
    de = out.get("dedup_effect", {})
    if "techdoc" in de and "machine_techdoc" in de:
        for label, key in (("all", "D_all"), ("dedup", "D_dedup")):
            ratio = de["techdoc"][key] / de["machine_techdoc"][key]
            print(f"  techdoc / machine_techdoc  ({label:<5}): "
                  f"{de['techdoc'][key]:.4f} / {de['machine_techdoc'][key]:.4f} "
                  f"= {ratio:.4f}")
        r_all = de["techdoc"]["D_all"] / de["machine_techdoc"]["D_all"]
        r_dd = de["techdoc"]["D_dedup"] / de["machine_techdoc"]["D_dedup"]
        out["verdict"] = {
            "ratio_all": round(r_all, 4), "ratio_dedup": round(r_dd, 4),
            "reversal_survives": bool(r_dd > 1),
            "shift": round(abs(r_dd / r_all - 1), 4)}
        print(f"  反转是否存活（去重后）: {out['verdict']['reversal_survives']}")
        print(f"  比值偏移: {out['verdict']['shift']:.1%}")

    f = M / "records_centroid" / "n6a_duplicates.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
