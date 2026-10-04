"""N7b: is the homogeneity result robust to the truncation budget?

N7 pooled per-text statistics across domains after normalising each domain to its own
mean, and found that machine text shows no difference in mean compactness but larger
dispersion. The length-controlled run used a 36-character budget, inherited from the
shortest arm's median, which truncates lyrics from a 407-character median to 36. That is
aggressive enough to be worth perturbing.

Because the pooled comparison divides each domain by its own mean first, a domain-wide
truncation effect largely cancels. What would not cancel is truncation changing the two
arms of a domain by different proportions. This sweeps the budget and reports whether the
mean and standard-deviation ratios move.
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
NBOOT = 2000
BUDGETS = [36, 80, 150, 300, 600]
PAIRS = [("poem", "machine_poem", "poem"),
         ("lyrics", "machine_lyrics", "lyrics"),
         ("techdoc", "machine_techdoc2", "techdoc"),
         ("hc3_human_medicine", "hc3_ai_medicine", "medicine"),
         ("hc3_human_openqa", "hc3_ai_openqa", "openqa")]


def sq(A, B):
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def per_text(E):
    C = E.mean(axis=0, keepdims=True)
    cent = np.sqrt(sq(E, C)[:, 0])
    D = np.sqrt(sq(E, E))
    np.fill_diagonal(D, np.nan)
    return cent, np.nanmean(D, axis=1)


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


def main():
    rng = np.random.default_rng(31)
    cache = {}
    for h, m, _ in PAIRS:
        for arm in (h, m):
            p = DATA / arm
            if p.is_dir() and arm not in cache:
                cache[arm] = pacsp_core.load_samples(p)[0]

    out = {}
    print(f"  {'budget':>8} {'n':>6} {'cent mean r':>13} {'excl1':>6} "
          f"{'cent sd r':>11} {'excl1':>6} {'nbr mean r':>12} {'excl1':>6} "
          f"{'nbr sd r':>10} {'excl1':>6}")
    for budget in BUDGETS:
        ch_all, cm_all, nh_all, nm_all = [], [], [], []
        for h, m, dom in PAIRS:
            th = [trim(x, budget) for x in cache[h]]
            tm = [trim(x, budget) for x in cache[m]]
            Eh = pacsp_core.compute_embeddings(th, model_name=MODEL)
            Em = pacsp_core.compute_embeddings(tm, model_name=MODEL)
            ch, nh = per_text(Eh)
            cm, nm = per_text(Em)
            ch_all.append(ch / ch.mean()); cm_all.append(cm / cm.mean())
            nh_all.append(nh / nh.mean()); nm_all.append(nm / nm.mean())
        ch_all = np.concatenate(ch_all); cm_all = np.concatenate(cm_all)
        nh_all = np.concatenate(nh_all); nm_all = np.concatenate(nm_all)

        def boot(a, b):
            mr, sr = [], []
            for _ in range(NBOOT):
                x = rng.choice(a, len(a), replace=True)
                y = rng.choice(b, len(b), replace=True)
                mr.append(x.mean() / y.mean())
                sr.append(x.std() / y.std())
            mr, sr = np.asarray(mr), np.asarray(sr)
            mci = [float(np.percentile(mr, 2.5)), float(np.percentile(mr, 97.5))]
            sci = [float(np.percentile(sr, 2.5)), float(np.percentile(sr, 97.5))]
            return (float(mr.mean()), mci, mci[0] > 1 or mci[1] < 1,
                    float(sr.mean()), sci, sci[0] > 1 or sci[1] < 1)

        cm_, cmci, cmex, csd, csci, csex = boot(ch_all, cm_all)
        nm_, nmci, nmex, nsd, nsci, nsex = boot(nh_all, nm_all)
        print(f"  {budget:>8} {len(ch_all):>6} {cm_:>13.4f} {str(cmex):>6} "
              f"{csd:>11.4f} {str(csex):>6} {nm_:>12.4f} {str(nmex):>6} "
              f"{nsd:>10.4f} {str(nsex):>6}")
        out[str(budget)] = {
            "n_per_side": int(len(ch_all)),
            "cent_mean_ratio": round(cm_, 4), "cent_mean_ci": [round(x, 4) for x in cmci],
            "cent_mean_excl1": cmex,
            "cent_sd_ratio": round(csd, 4), "cent_sd_ci": [round(x, 4) for x in csci],
            "cent_sd_excl1": csex,
            "nbr_mean_ratio": round(nm_, 4), "nbr_mean_ci": [round(x, 4) for x in nmci],
            "nbr_mean_excl1": nmex,
            "nbr_sd_ratio": round(nsd, 4), "nbr_sd_ci": [round(x, 4) for x in nsci],
            "nbr_sd_excl1": nsex}

    print()
    print("=== robustness ===")
    means_c = [v["cent_mean_ratio"] for v in out.values()]
    sds_c = [v["cent_sd_ratio"] for v in out.values()]
    means_n = [v["nbr_mean_ratio"] for v in out.values()]
    sds_n = [v["nbr_sd_ratio"] for v in out.values()]
    print(f"  centroid mean ratio across budgets: "
          f"{min(means_c):.4f} - {max(means_c):.4f}")
    print(f"  centroid sd ratio   across budgets: "
          f"{min(sds_c):.4f} - {max(sds_c):.4f}")
    print(f"  neighbour mean ratio across budgets: "
          f"{min(means_n):.4f} - {max(means_n):.4f}")
    print(f"  neighbour sd ratio  across budgets: "
          f"{min(sds_n):.4f} - {max(sds_n):.4f}")
    n_mean_excl = sum(1 for v in out.values()
                      if v["cent_mean_excl1"] or v["nbr_mean_excl1"])
    n_sd_excl = sum(1 for v in out.values()
                    if v["cent_sd_excl1"] and v["nbr_sd_excl1"])
    print(f"  budgets where a mean ratio excludes 1: {n_mean_excl}/{len(out)}")
    print(f"  budgets where both sd ratios exceed 1: {n_sd_excl}/{len(out)}")
    out["_robustness"] = {
        "cent_mean_range": [round(min(means_c), 4), round(max(means_c), 4)],
        "cent_sd_range": [round(min(sds_c), 4), round(max(sds_c), 4)],
        "nbr_mean_range": [round(min(means_n), 4), round(max(means_n), 4)],
        "nbr_sd_range": [round(min(sds_n), 4), round(max(sds_n), 4)],
        "budgets_with_mean_excl1": n_mean_excl,
        "budgets_with_both_sd_excl1": n_sd_excl}

    f = M / "records_centroid" / "n7b_budget_sweep.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
