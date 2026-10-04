"""N4: how to handle sigma. Four candidate resolutions, tested.

Section 2.4 of the framework paper states that sigma must be declared and its sensitivity
reported. That is a holding position, not a solution. This tests four ways to get past it.

The observation that motivates them: for an RBF kernel, changing sigma multiplies Phi by a
positive constant at every point, because K(x,v; s) = phi(v)^(s0^2/s^2) with
phi(v) = exp(-d^2/2s0^2) > 0. The constant is 13.9x per unit of Phi norm at 1.6x, but it
is a constant. So any functional that first normalises Phi across its argument is exactly
invariant to it. The earlier measurement compared raw Phi as a function in L2, where that
constant is not cancelled, so what was measured was sigma sensitivity of an unnormalised
comparison, not of the representation.

Four candidates:

  A  barycentre      normalise Phi to a probability vector and compare the resulting
                     distributions; exactly scale-invariant in Phi
  B  integrated      average the normalised profiles over a range of sigma, so no single
                     value is chosen
  C  laplacian       replace the Gaussian kernel with exp(-d/s), which decays more
                     sharply and may make the profile shape less sigma-dependent
  D  scale-free      use the profile's own shape statistics (its entropy and its
                     concentration on the top-k entries) rather than the vector

Each is measured on the same corpora three ways: order sensitivity (as-filed against 200
permutations), cross-arm spread, and the human-versus-machine ratio where a pair exists.
"""

import json
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
NPERM = 200
PAPER_ARMS = ["poem", "machine_poem", "lyrics", "machine_lyrics", "techdoc",
              "machine_techdoc", "hc3_human_medicine", "hc3_ai_medicine",
              "hc3_human_openqa", "hc3_ai_openqa"]
PAIRS = [("poem", "machine_poem"), ("lyrics", "machine_lyrics"),
         ("techdoc", "machine_techdoc"),
         ("hc3_human_medicine", "hc3_ai_medicine"),
         ("hc3_human_openqa", "hc3_ai_openqa")]
SIGMA_MULTS = [0.5, 0.7, 1.0, 1.4, 2.0]


def sq(A, B):
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def median_dist(E):
    d2 = sq(E, E)
    off = d2[~np.eye(len(E), dtype=bool)]
    off = off[off > 0]
    return float(np.sqrt(np.median(off)))


def profile(E, sigma, kind="gaussian", frame=None):
    """Per-snapshot profile over the frame, as a row-stochastic matrix."""
    F = E if frame is None else frame
    d2 = sq(E, F)
    if kind == "gaussian":
        K = np.exp(-d2 / (2.0 * sigma ** 2))
    else:                                   # laplacian
        K = np.exp(-np.sqrt(d2) / sigma)
    s = K.sum(axis=1, keepdims=True)
    return K / np.maximum(s, 1e-300)


def barycentre(P):
    """Normalise the mean profile to a probability vector: exactly scale-invariant."""
    b = P.mean(axis=0)
    return b / max(b.sum(), 1e-300)


def integrated_profile(E, sigma0, kind="gaussian"):
    """Average the profiles over a range of sigma, so none is chosen."""
    acc = None
    for mult in SIGMA_MULTS:
        P = profile(E, sigma0 * mult, kind)
        acc = P if acc is None else acc + P
    return acc / len(SIGMA_MULTS)


def shape_stats(P):
    """Scale-free shape of the mean profile: entropy and top-k concentration."""
    b = barycentre(P)
    ent = float(-np.sum(b * np.log(b + 1e-300)))
    order = np.sort(b)[::-1]
    top1 = float(order[0])
    top5 = float(order[:5].sum())
    top20 = float(order[:20].sum())
    return np.array([ent, top1, top5, top20])


def measure(E, kind, sigma0, mode):
    """One scalar for the corpus under a given resolution."""
    if mode == "A":
        b = barycentre(profile(E, sigma0, kind))
        # distance from a uniform profile, which is scale-free in Phi
        uni = np.full_like(b, 1.0 / len(b))
        return float(np.linalg.norm(b - uni))
    if mode == "B":
        b = barycentre(integrated_profile(E, sigma0, kind))
        uni = np.full_like(b, 1.0 / len(b))
        return float(np.linalg.norm(b - uni))
    if mode == "C":
        b = barycentre(profile(E, sigma0, "laplacian"))
        uni = np.full_like(b, 1.0 / len(b))
        return float(np.linalg.norm(b - uni))
    if mode == "D":
        s = shape_stats(profile(E, sigma0, kind))
        # standardise so the four statistics are commensurate
        return float(np.linalg.norm(s / (np.abs(s).sum() + 1e-300)))
    raise ValueError(mode)


def main():
    embs = {}
    for arm in PAPER_ARMS:
        p = DATA / arm
        if not p.is_dir():
            continue
        texts, _ = pacsp_core.load_samples(p)
        embs[arm] = pacsp_core.compute_embeddings(texts, model_name=MODEL)
    print(f"  loaded {len(embs)} corpora")

    rng = np.random.default_rng(11)
    out = {"sigma_mults": SIGMA_MULTS}

    # ---------------------------------------------------------------- sigma sweep
    print("\n=== A: how much does each resolution move when sigma changes? ===")
    print("  (relative change of the corpus scalar at sigma x1.6 vs sigma x1.0)")
    rows = {}
    for mode, label in (("A", "A barycentre"), ("B", "B integrated"),
                        ("C", "C laplacian"), ("D", "D shape stats")):
        rels = []
        for arm, E in embs.items():
            s0 = 0.5 * median_dist(E)
            kind = "laplacian" if mode == "C" else "gaussian"
            v1 = measure(E, kind, s0, mode)
            v2 = measure(E, kind, s0 * 1.6, mode)
            rels.append(abs(v2 - v1) / abs(v1) if v1 else 0.0)
        rows[mode] = rels
        print(f"  {label:<16} mean {np.mean(rels):.4f}  max {np.max(rels):.4f}  "
              f"min {np.min(rels):.4f}")
    out["sigma_sensitivity"] = {k: {"mean": round(float(np.mean(v)), 4),
                                    "max": round(float(np.max(v)), 4)}
                                for k, v in rows.items()}

    # ---------------------------------------------------------------- order
    print("\n=== order sensitivity: as-filed vs 200 permutations ===")
    print(f"  {'resolution':<16} {'mean |bias|':>12} {'mean CV':>9}")
    order = {}
    for mode, label in (("A", "A barycentre"), ("B", "B integrated"),
                        ("C", "C laplacian"), ("D", "D shape stats")):
        biases, cvs = [], []
        for arm, E in embs.items():
            s0 = 0.5 * median_dist(E)
            kind = "laplacian" if mode == "C" else "gaussian"
            asis = measure(E, kind, s0, mode)
            vals = np.array([measure(E[rng.permutation(len(E))], kind, s0, mode)
                             for _ in range(NPERM)])
            if vals.mean():
                biases.append(abs(asis - vals.mean()) / vals.mean())
                cvs.append(vals.std() / vals.mean())
        order[mode] = {"mean_bias": round(float(np.mean(biases)), 6),
                       "mean_cv": round(float(np.mean(cvs)), 6)}
        print(f"  {label:<16} {np.mean(biases):>12.6f} {np.mean(cvs):>9.6f}")
    out["order_sensitivity"] = order

    # ---------------------------------------------------------------- contrast
    print("\n=== human / machine ratio, by resolution ===")
    print(f"  {'resolution':<16} " + "".join(f"{h.split('_')[-1][:9]:>11}" for h, _ in PAIRS))
    print(f"  {'mean_pair_dist':<16} " + "".join(
        f"{'':>11}" for _ in PAIRS) + "   (reference)")
    ratios = {}
    for mode, label in (("A", "A barycentre"), ("B", "B integrated"),
                        ("C", "C laplacian"), ("D", "D shape stats")):
        r = {}
        for h, m in PAIRS:
            if h not in embs or m not in embs:
                continue
            sh = 0.5 * median_dist(embs[h])
            sm = 0.5 * median_dist(embs[m])
            kind = "laplacian" if mode == "C" else "gaussian"
            vh = measure(embs[h], kind, sh, mode)
            vm = measure(embs[m], kind, sm, mode)
            r[h.split("_")[-1]] = round(vh / vm, 4) if vm else None
        ratios[mode] = r
        print(f"  {label:<16} " + "".join(
            f"{(r.get(h.split('_')[-1]) or float('nan')):>11.4f}" for h, _ in PAIRS))
    # reference: mean_pair_dist
    ref = {}
    for h, m in PAIRS:
        if h in embs and m in embs:
            dh = sq(embs[h], embs[h]); dm = sq(embs[m], embs[m])
            iu = np.triu_indices(len(dh), k=1)
            iu2 = np.triu_indices(len(dm), k=1)
            ref[h.split("_")[-1]] = round(
                float(np.sqrt(dh[iu]).mean() / np.sqrt(dm[iu2]).mean()), 4)
    print(f"  {'mean_pair_dist':<16} " + "".join(
        f"{ref.get(h.split('_')[-1], float('nan')):>11.4f}" for h, _ in PAIRS))
    out["ratios"] = ratios
    out["reference_mean_pair_dist"] = ref

    print()
    print("=== direction consistency per resolution ===")
    for mode, r in ratios.items():
        vals = [v for v in r.values() if v is not None]
        below = sum(1 for v in vals if v < 1)
        print(f"  {mode}: {below}/{len(vals)} below 1   "
              f"spread {max(vals)/min(vals):.2f}x")
    vals = list(ref.values())
    below = sum(1 for v in vals if v < 1)
    print(f"  mean_pair_dist: {below}/{len(vals)} below 1   "
          f"spread {max(vals)/min(vals):.2f}x")

    f = M / "records_centroid" / "n4_sigma_resolutions.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
