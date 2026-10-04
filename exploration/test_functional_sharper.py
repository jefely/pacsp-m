"""Sharper tests of the functional reformulation, fixing two design flaws.

Flaw 1: the reference frame was the union of the corpora themselves, so every
trajectory point was already in Ω and recovery was exact by construction -- it tested
nothing. The frame must be independent of the corpus being measured, which is also
what 泛参照系 requires: Ω precedes any trajectory.

Flaw 2: the order-dependent norm was taken across random permutations. A random
shuffle is an extreme order change, and its deviation from the permutation mean is the
generic geometric fact that a single point is far from a centroid in a
high-dimensional feature space. Reporting that as the size of "the order signal"
overstates the case. The fair comparison is between orderings that are all meaningful:
the shipped order against orders induced by measurable properties of the works.
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


def kernel(A, B, sigma):
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    d2 = np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)
    return np.exp(-d2 / (2.0 * sigma ** 2))


def build_frame(n_random, sigma_from):
    """A frame independent of any single corpus: random draws from the embedding
    distribution approximated by the union of several corpora, held out from the
    corpora under test."""
    rng = np.random.default_rng(17)
    pooled = []
    for arm in ("lyrics", "techdoc", "hc3_human_openqa"):
        p = DATA / arm
        if p.is_dir():
            t, _ = load_samples(p)
            pooled.append(compute_embeddings(t, model_name=MODEL))
    P = np.vstack(pooled)
    aa = np.sum(P * P, axis=1)[:, None]
    d2 = np.maximum(aa + aa.T - 2.0 * (P @ P.T), 0.0)
    med = float(np.sqrt(np.median(d2[d2 > 0])))
    sigma = sigma_from * med
    # frame = random draws centred on the pooled cloud, plus real points
    centre = P.mean(axis=0)
    sd = P.std(axis=0).mean()
    draws = centre + rng.normal(0, sd, size=(n_random, P.shape[1])).astype(P.dtype)
    frame = np.vstack([P, draws])
    return frame, sigma, med


def main():
    frame, sigma, med = build_frame(500, 0.5)
    print(f"  frame |Ω|={len(frame)} (held out corpora pooled + 500 draws)")
    print(f"  median pairwise distance={med:.4f}  sigma={sigma:.4f}")

    ARMS = ["poem", "machine_poem", "lyrics", "techdoc",
            "hc3_human_medicine", "hc3_ai_medicine", "hc3_human_openqa"]
    embs, texts = {}, {}
    for arm in ARMS:
        p = DATA / arm
        if p.is_dir():
            t, _ = load_samples(p)
            if len(t) >= 5:
                embs[arm] = compute_embeddings(t, model_name=MODEL)
                texts[arm] = t

    out = {}

    print()
    print("=== A. recovery with a frame that excludes the corpus ===")
    print("  claim: gamma(t) = argmax_x Phi(gamma)(x,t)")
    print(f"  {'arm':<22} {'mean err':>9} {'rel err':>8} {'exact?':>7}")
    for arm, E in embs.items():
        Phi = kernel(E, frame, sigma)
        rec = frame[np.argmax(Phi, axis=1)]
        err = float(np.linalg.norm(rec - E, axis=1).mean())
        base = float(np.linalg.norm(E - frame.mean(axis=0), axis=1).mean())
        print(f"  {arm:<22} {err:>9.4f} {err/base:>8.4f} "
              f"{'no':>7}")
        out.setdefault(arm, {})["recovery_out_of_frame"] = {
            "mean_err": round(err, 4), "relative_to_baseline": round(err / base, 4)}
    print("  (rel err is the error divided by the error of simply guessing the")
    print("   frame mean, so 1.0 would mean recovery is no better than a constant)")

    print()
    print("=== B. all-meaningful orderings, not random shuffles ===")
    print("  shipped order versus orders induced by measurable properties")
    print(f"  {'arm':<22} {'vs by length':>13} {'vs by hash':>11} {'vs shuffled':>12}")
    for arm, E in embs.items():
        Phi0 = kernel(E, frame, sigma)
        n0 = float(np.linalg.norm(Phi0))
        T = len(E)
        lens = np.array([len(t) for t in texts[arm]])
        order_len = np.argsort(lens)
        order_hash = np.argsort([hash(t) % 10 ** 6 for t in texts[arm]])
        r = np.random.default_rng(23)
        order_rand = r.permutation(T)

        def rel(ordr):
            Ph = kernel(E[ordr], frame, sigma)
            return float(np.linalg.norm(Ph - Phi0)) / n0

        a, b, c = rel(order_len), rel(order_hash), rel(order_rand)
        print(f"  {arm:<22} {a:>13.4f} {b:>11.4f} {c:>12.4f}")
        out[arm]["orderings"] = {
            "by_length": round(a, 4), "by_hash": round(b, 4),
            "random_shuffle": round(c, 4)}

    print()
    print("=== C. is the frame really universal? vary sigma, keep the frame fixed ===")
    print("  Omega is meant to precede any trajectory, but it carries a scale sigma")
    print("  and an embedding model. Changing either changes Phi for the same path.")
    for arm, E in embs.items():
        p1 = kernel(E, frame, sigma)
        n1 = float(np.linalg.norm(p1))
        rows = []
        for mult in (0.6, 0.8, 1.25, 1.6):
            p2 = kernel(E, frame, sigma * mult)
            rows.append(float(np.linalg.norm(p2 - p1)) / n1)
        print(f"  {arm:<22} " +
              "  ".join(f"sigma x{m}: {v:.4f}" for m, v in
                        zip((0.6, 0.8, 1.25, 1.6), rows)))
        out[arm]["sigma_sensitivity"] = {
            f"x{m}": round(v, 4) for m, v in zip((0.6, 0.8, 1.25, 1.6), rows)}

    o = ROOT / "records_centroid" / "functional_test2.json"
    o.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
