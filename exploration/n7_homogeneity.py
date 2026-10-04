"""N7: is machine text more homogeneous than human text?

Section 4.5 records a hint: the machine arms span 0.65 to 1.20 against the human arms'
0.54 to 1.12, so machine text looks less dispersed. With five corpora per side that is
five observations against five, far too few to conclude anything. The test has to drop to
the level of individual texts, which gives 155 against 155.

Two per-text statistics are computed inside each corpus, so both are defined relative to
the corpus the text belongs to and neither compares across frames:

  centroid distance   ||v - mean(v in own corpus)||: how far a text sits from its own
                      group's centre
  mean neighbour dist the text's mean distance to the other texts of its own corpus

The hypothesis makes two predictions that are tested separately, because they can come
apart: machine text may sit closer to its own centre (lower centroid distance) or may be
more evenly spread (lower dispersion of the per-text values), or both.

Length is a candidate confound and is controlled by trimming every text to a common
budget before embedding. Both unadjusted and adjusted comparisons are reported.
"""

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
M = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(M))
import pacsp_core  # noqa: E402

DATA = M / "data"
MODEL = "BAAI/bge-large-zh-v1.5"
NBOOT = 4000

# (human arm, machine arm, domain label)
PAIRS = [
    ("poem", "machine_poem", "poem"),
    ("lyrics", "machine_lyrics", "lyrics"),
    ("techdoc", "machine_techdoc", "techdoc-orig"),
    ("techdoc", "machine_techdoc2", "techdoc-topic"),
    ("hc3_human_medicine", "hc3_ai_medicine", "medicine"),
    ("hc3_human_openqa", "hc3_ai_openqa", "openqa"),
]


def sq(A, B):
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def per_text_stats(E):
    """Two per-text values, both defined inside this corpus."""
    C = E.mean(axis=0, keepdims=True)
    cent = np.sqrt(sq(E, C)[:, 0])
    n = len(E)
    D = np.sqrt(sq(E, E))
    np.fill_diagonal(D, np.nan)
    nbr = np.nanmean(D, axis=1)
    return cent, nbr


def trim(text, target):
    t = " ".join(text.split())
    if len(t) <= target:
        return t
    parts = [p for p in re.split(r"(?<=[。！？；!?;])", t) if p.strip()]
    acc = ""
    for p in parts:
        if len(acc) + len(p) > target:
            break
        acc += p
    return acc if len(acc) >= target * 0.4 else t[:target]


def boot_ratio(a, b, rng, nboot=NBOOT):
    """Bootstrap the ratio of means, and also the ratio of standard deviations."""
    ma, mb = [], []
    for _ in range(nboot):
        ma.append(rng.choice(a, size=len(a), replace=True).mean())
        mb.append(rng.choice(b, size=len(b), replace=True).mean())
    ma, mb = np.asarray(ma), np.asarray(mb)
    r = ma / mb
    sa, sb = [], []
    for _ in range(nboot):
        sa.append(rng.choice(a, size=len(a), replace=True).std())
        sb.append(rng.choice(b, size=len(b), replace=True).std())
    sr = np.asarray(sa) / np.asarray(sb)
    return (float(r.mean()), [float(np.percentile(r, 2.5)),
                              float(np.percentile(r, 97.5))],
            bool(np.percentile(r, 2.5) > 1 or np.percentile(r, 97.5) < 1),
            float(sr.mean()), [float(np.percentile(sr, 2.5)),
                               float(np.percentile(sr, 97.5))])


def main():
    rng = np.random.default_rng(23)
    out = {}

    for adjusted in (False, True):
        tag = "length-controlled" if adjusted else "unadjusted"
        print("=" * 76)
        print(f"  {tag}")
        print("=" * 76)

        # gather all texts, decide the common budget from the smallest median
        cache = {}
        budgets = []
        for h, m, dom in PAIRS:
            for arm in (h, m):
                p = DATA / arm
                if p.is_dir() and arm not in cache:
                    t, _ = pacsp_core.load_samples(p)
                    cache[arm] = t
                    if len(t) > 3:
                        budgets.append(int(np.median([len(x) for x in t])))
        target = min(budgets) if budgets else 200
        if adjusted:
            print(f"  common character budget: {target}")

        rows = {}
        for h, m, dom in PAIRS:
            if h not in cache or m not in cache:
                continue
            th = [trim(x, target) for x in cache[h]] if adjusted else cache[h]
            tm = [trim(x, target) for x in cache[m]] if adjusted else cache[m]
            Eh = pacsp_core.compute_embeddings(th, model_name=MODEL)
            Em = pacsp_core.compute_embeddings(tm, model_name=MODEL)
            ch, nh = per_text_stats(Eh)
            cm, nm = per_text_stats(Em)
            rows[dom] = {"cent": (ch, cm), "nbr": (nh, nm),
                         "n_human": len(ch), "n_machine": len(cm)}

        print(f"\n  {'domain':<14} {'cent H/M':>10} {'cent ratio':>11} {'excl1':>6} "
              f"{'nbr H/M':>10} {'nbr ratio':>10} {'excl1':>6}")
        summary = {}
        for dom, r in rows.items():
            ch, cm = r["cent"]
            nh, nm = r["nbr"]
            cr_mean, cr_ci, cr_ex, csr_mean, csr_ci = boot_ratio(ch, cm, rng)
            nr_mean, nr_ci, nr_ex, nsr_mean, nsr_ci = boot_ratio(nh, nm, rng)
            print(f"  {dom:<14} {ch.mean():>4.3f}/{cm.mean():<4.3f} "
                  f"{cr_mean:>11.4f} {str(cr_ex):>6} "
                  f"{nh.mean():>4.3f}/{nm.mean():<4.3f} {nr_mean:>10.4f} "
                  f"{str(nr_ex):>6}")
            summary[dom] = {
                "cent_mean_human": round(float(ch.mean()), 4),
                "cent_mean_machine": round(float(cm.mean()), 4),
                "cent_ratio": round(cr_mean, 4), "cent_ci": [round(x, 4) for x in cr_ci],
                "cent_excl1": cr_ex,
                "cent_sd_ratio": round(csr_mean, 4),
                "nbr_mean_human": round(float(nh.mean()), 4),
                "nbr_mean_machine": round(float(nm.mean()), 4),
                "nbr_ratio": round(nr_mean, 4), "nbr_ci": [round(x, 4) for x in nr_ci],
                "nbr_excl1": nr_ex,
            }
        out[tag] = summary

        # pooled across domains, lengths differ so standardise per domain first
        allc_h = np.concatenate([r["cent"][0] / r["cent"][0].mean()
                                 for r in rows.values()])
        allc_m = np.concatenate([r["cent"][1] / r["cent"][1].mean()
                                 for r in rows.values()])
        alln_h = np.concatenate([r["nbr"][0] / r["nbr"][0].mean()
                                 for r in rows.values()])
        alln_m = np.concatenate([r["nbr"][1] / r["nbr"][1].mean()
                                 for r in rows.values()])
        print(f"\n  pooled (each domain normalised to its own mean first):")
        for label, a, b in (("centroid distance", allc_h, allc_m),
                            ("neighbour distance", alln_h, alln_m)):
            mr, ci, ex, sr, sci = boot_ratio(a, b, rng)
            print(f"    {label:<20} n={len(a)}/{len(b)}  ratio {mr:.4f} "
                  f"CI[{ci[0]:.4f},{ci[1]:.4f}]  excl1={ex}  "
                  f"sd ratio {sr:.4f} CI[{sci[0]:.4f},{sci[1]:.4f}]")
            out[tag][f"pooled_{label.split()[0]}"] = {
                "n_human": int(len(a)), "n_machine": int(len(b)),
                "mean_ratio": round(mr, 4),
                "mean_ci": [round(x, 4) for x in ci], "mean_excl1": ex,
                "sd_ratio": round(sr, 4),
                "sd_ci": [round(x, 4) for x in sci],
                "sd_excl1": bool(sci[0] > 1 or sci[1] < 1)}
        print()

    # ------------------------------------------------------------------ verdict
    print("=" * 76)
    print("  verdict")
    print("=" * 76)
    for tag in out:
        ph = out[tag].get("pooled_centroid", {})
        pn = out[tag].get("pooled_neighbour", {})
        print(f"  {tag}:")
        print(f"    centroid  mean ratio {ph.get('mean_ratio')} "
              f"excl1={ph.get('mean_excl1')}   sd ratio {ph.get('sd_ratio')} "
              f"excl1={ph.get('sd_excl1')}")
        print(f"    neighbour mean ratio {pn.get('mean_ratio')} "
              f"excl1={pn.get('mean_excl1')}   sd ratio {pn.get('sd_ratio')} "
              f"excl1={pn.get('sd_excl1')}")
        n_cent = sum(1 for k, v in out[tag].items()
                     if isinstance(v, dict) and v.get("cent_ratio", 1) < 1)
        n_nbr = sum(1 for k, v in out[tag].items()
                    if isinstance(v, dict) and v.get("nbr_ratio", 1) < 1)
        print(f"    domains where machine is closer to centre: {n_cent}")
        print(f"    domains where machine has shorter neighbour distance: {n_nbr}")

    f = M / "records_centroid" / "n7_homogeneity.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
