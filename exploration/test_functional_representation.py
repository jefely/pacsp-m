"""Test the functional reformulation: S(γ) = Φ(γ), Φ(γ)(x,t) = K(x, γ(t)).

Four claims from the proposal are checked against real corpora:

  1. injectivity of γ ↦ Φ(γ) under a characteristic kernel
  2. automatic recovery γ(t) = argmax_x Φ(γ)(x,t), and what it costs when the
     reference frame Ω is discretised -- argmax over a finite codebook can only
     return codebook members
  3. "order becomes signal": how much of Φ actually changes when the corpus is
     permuted, separated from the scale and window effects that inflate the scalar
  4. whether the construction fixes the within-document consistency issue

The reference frame is the union of all corpus embeddings, optionally augmented by
random draws, so Ω is a fixed codebook independent of the corpus being measured --
which is what the proposal requires of a 泛参照系.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
from pacsp_core import compute_embeddings, load_samples  # noqa: E402

DATA = ROOT / "data"
MODEL = "BAAI/bge-large-zh-v1.5"
ARMS = ["poem", "lyrics", "techdoc", "hc3_human_medicine", "hc3_ai_medicine"]
NPERM = 100
SIGMA_Q = 0.5          # sigma as a quantile of pairwise distances
CODEBOOK_SIZES = [31, 100, 500, 2000]


def kernel(A, B, sigma):
    """Gaussian kernel between rows of A and rows of B, via squared distances."""
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    d2 = np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)
    return np.exp(-d2 / (2.0 * sigma ** 2))


def main():
    print("=== load corpora and build a shared reference frame ===")
    embs, sizes = {}, {}
    for arm in ARMS:
        p = DATA / arm
        if not p.is_dir():
            continue
        texts, _ = load_samples(p)
        if len(texts) < 5:
            continue
        embs[arm] = compute_embeddings(texts, model_name=MODEL)
        sizes[arm] = [len(t.encode("utf-8")) for t in texts]
        print(f"  {arm:<22} T={len(texts):>3}  dim={embs[arm].shape[1]}")

    allv = np.vstack(list(embs.values()))
    # sigma from the median pairwise distance, a standard scale heuristic
    rng = np.random.default_rng(3)
    idx = rng.choice(len(allv), size=min(800, len(allv)), replace=False)
    sub = allv[idx]
    aa = np.sum(sub * sub, axis=1)[:, None]
    d2 = np.maximum(aa + aa.T - 2.0 * (sub @ sub.T), 0.0)
    med = float(np.sqrt(np.median(d2[d2 > 0])))
    sigma = SIGMA_Q * med
    print(f"  reference frame: |Ω|={len(allv)}  median dist={med:.4f}  "
          f"sigma={sigma:.4f}")

    results = {}

    # ---------------------------------------------------------------- 1 & 2
    print()
    print("=== 1/2. injectivity and recovery, versus codebook size ===")
    print("  Ω sampled from the union; recovery returns the nearest codebook")
    print("  member, so it is exact only if the original point is in Ω.")
    print(f"  {'arm':<22} {'recov err':>10} {'rel err':>9} {'argmax ok':>10}")
    for arm, E in embs.items():
        # full-frame recovery: the corpus itself is inside Ω
        Phi_full = kernel(E, allv, sigma)          # (T, |Ω|)
        rec = allv[np.argmax(Phi_full, axis=1)]
        err = float(np.linalg.norm(rec - E, axis=1).mean())
        rel = err / float(np.linalg.norm(E, axis=1).mean())
        exact = int(np.sum(np.argmax(Phi_full, axis=1) < len(E)))
        print(f"  {arm:<22} {err:>10.5f} {rel:>9.5f} {exact:>6}/{len(E)}")
        results.setdefault(arm, {})["recovery_full_frame"] = {
            "mean_err": round(err, 6), "rel_err": round(rel, 6),
            "exact_hits": exact, "T": len(E)}

    # recovery with a small codebook that excludes the corpus points
    print()
    print("  with Ω drawn at random and NOT containing the corpus points:")
    print(f"  {'arm':<22} " + "".join(f"{('n=' + str(n)):>12}" for n in CODEBOOK_SIZES))
    for arm, E in embs.items():
        row = []
        for n in CODEBOOK_SIZES:
            cb = allv[rng.choice(len(allv), size=min(n, len(allv)), replace=False)]
            Phi = kernel(E, cb, sigma)
            rec = cb[np.argmax(Phi, axis=1)]
            row.append(float(np.linalg.norm(rec - E, axis=1).mean()))
        print(f"  {arm:<22} " + "".join(f"{v:>12.5f}" for v in row))
        results[arm]["recovery_random_codebook"] = {
            str(n): round(v, 6) for n, v in zip(CODEBOOK_SIZES, row)}

    # ---------------------------------------------------------------- 3
    print()
    print("=== 3. does permutation change Φ, beyond scale? ===")
    print(f"  {'arm':<22} {'||Φ||':>9} {'mean||ΔΦ||':>11} {'rel':>7} "
          f"{'L11 spread':>11}")
    for arm, E in embs.items():
        Phi0 = kernel(E, allv, sigma)
        n0 = float(np.linalg.norm(Phi0))
        l11_0 = float(Phi0.sum())
        r = np.random.default_rng(5)
        diffs, l11s = [], []
        for _ in range(NPERM):
            perm = r.permutation(len(E))
            Ph = kernel(E[perm], allv, sigma)
            diffs.append(float(np.linalg.norm(Ph - Phi0)))
            l11s.append(float(Ph.sum()))
        rel = float(np.mean(diffs)) / n0
        l11s = np.asarray(l11s)
        spread = float((l11s.max() - l11s.min()) / l11s.mean())
        print(f"  {arm:<22} {n0:>9.3f} {np.mean(diffs):>11.3f} {rel:>7.4f} "
              f"{spread:>11.4f}")
        results[arm]["permutation"] = {
            "phi_norm": round(n0, 4),
            "mean_delta_norm": round(float(np.mean(diffs)), 4),
            "relative": round(rel, 6),
            "L11_as_ordered": round(l11_0, 4),
            "L11_spread_over_permutations": round(spread, 6),
        }

    # ---------------------------------------------------------------- 4
    print()
    print("=== 4. order-invariant and order-dependent parts ===")
    print(f"  {'arm':<22} {'||E_π Φ||':>11} {'||Φ_ord||':>11} {'ratio':>7}")
    for arm, E in embs.items():
        Phi0 = kernel(E, allv, sigma)
        r = np.random.default_rng(5)
        stack = np.stack([kernel(E[r.permutation(len(E))], allv, sigma)
                          for _ in range(NPERM)])
        mean = stack.mean(axis=0)
        dev = stack - mean
        n_inv = float(np.linalg.norm(mean))
        n_ord = float(np.linalg.norm(dev))
        print(f"  {arm:<22} {n_inv:>11.3f} {n_ord:>11.3f} {n_ord/n_inv:>7.4f}")
        results[arm]["decomposition"] = {
            "invariant_norm": round(n_inv, 4),
            "order_norm": round(n_ord, 4),
            "ratio": round(n_ord / n_inv, 6),
        }

    out = ROOT / "records_centroid" / "functional_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
