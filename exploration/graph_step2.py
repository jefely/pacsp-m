"""Step 2: does centring the embeddings reduce the sigma sensitivity?

Step 1 established two things. The graph representation removes order sensitivity
completely when edges are built from similarity rather than adjacency: the kNN edge
set and every graph invariant were unchanged under permutation, while the path graph's
invariants moved by 32 to 240 percent. But sigma sensitivity remained, with Phi moving
0.26 to 1.16 of its own norm when sigma moved by 0.6x to 1.6x.

The measured geometry explains why. The encoder returns L2-normalised vectors, so all
embeddings lie on the unit sphere, and their mean has norm 0.916 -- that is, 91.6
percent of a typical vector's length is a shared component present in every text. The
median pairwise distance is only 0.562 out of a maximum of sqrt(2) = 1.414, so the
corpus occupies a small cap.

For unit vectors, d^2 = 2(1 - cos theta), so the RBF kernel is
    K = exp( -(1 - cos theta) / sigma^2 )
and the ratio between two bandwidths is
    K'/K = exp( (1 - cos theta)(1/sigma^2 - 1/sigma'^2) )
which varies from pair to pair. A shared component inflates the cosine similarity of
everything, compressing (1 - cos theta) into a narrow band, so the kernel weights are
concentrated and sigma has an outsized effect.

Centring removes the shared component. This step tests whether that actually reduces
the sensitivity, and whether it damages anything else.
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
from pacsp_core import compute_embeddings, load_samples  # noqa: E402

DATA = ROOT / "data"
MODEL = "BAAI/bge-large-zh-v1.5"
STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
ARMS = ["poem", "lyrics", "techdoc", "machine_poem", "hc3_human_medicine"]


def sq(A, B):
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def median_dist(E):
    d2 = sq(E, E)
    off = d2[~np.eye(len(E), dtype=bool)]
    off = off[off > 0]
    return float(np.sqrt(np.median(off)))


def kernel(A, B, s):
    return np.exp(-sq(A, B) / (2.0 * s ** 2))


def knn_edges(E, k=3):
    d2 = sq(E, E)
    np.fill_diagonal(d2, np.inf)
    e = set()
    for i in range(len(E)):
        for j in np.argsort(d2[i])[:k]:
            e.add((min(i, int(j)), max(i, int(j))))
    return sorted(e)


def invariants(E, edges, s):
    w = np.array([float(np.exp(-sq(E[[i]], E[[j]])[0, 0] / (2 * s ** 2)))
                  for i, j in edges]) if edges else np.zeros(0)
    deg = np.zeros(len(E))
    for (i, j), wij in zip(edges, w):
        deg[i] += wij; deg[j] += wij
    return {"weight_sum": float(w.sum()), "degree_std": float(deg.std())}


def main():
    report = {}
    for arm in ARMS:
        p = DATA / arm
        if not p.is_dir():
            continue
        texts, _ = load_samples(p)
        if len(texts) < 6:
            continue
        E = compute_embeddings(texts, model_name=MODEL)
        Ec = E - E.mean(axis=0)              # centred
        out = {}
        print(f"\n{'=' * 72}\n{arm}\n{'=' * 72}")
        for label, X in (("raw", E), ("centred", Ec)):
            nrm = np.linalg.norm(X, axis=1)
            med = median_dist(X)
            s = 0.5 * med
            # spread of the distance distribution, which governs how much sigma bites
            d2 = sq(X, X)
            off = np.sqrt(d2[~np.eye(len(X), dtype=bool)])
            off = off[off > 0]
            cv = off.std() / off.mean()
            print(f"  {label:<8} norm {nrm.min():.3f}-{nrm.max():.3f}  "
                  f"median_d {med:.4f}  sigma {s:.4f}  "
                  f"pairdist CV {cv:.3f}  mean_norm_share "
                  f"{np.linalg.norm(X.mean(axis=0))/nrm.mean():.3f}")
            # sigma sensitivity of Phi
            base = kernel(X, X, s)
            nb = float(np.linalg.norm(base))
            sens = {}
            for k in (0.6, 0.8, 1.25, 1.6):
                sens[k] = float(np.linalg.norm(kernel(X, X, s * k) - base)) / nb
            # graph invariants and their sigma sensitivity
            e = knn_edges(X, 3)
            iv = invariants(X, e, s)
            gsens = {}
            for k in (0.6, 1.6):
                iv2 = invariants(X, e, s * k)
                gsens[k] = abs(iv2["weight_sum"] - iv["weight_sum"]) / iv["weight_sum"]
            print(f"           Phi sigma-sens: " +
                  "  ".join(f"x{k}:{v:.4f}" for k, v in sens.items()))
            print(f"           graph w_sum sigma-sens: " +
                  "  ".join(f"x{k}:{v:.4f}" for k, v in gsens.items()))
            out[label] = {
                "norm_min": round(float(nrm.min()), 4),
                "norm_max": round(float(nrm.max()), 4),
                "median_dist": round(med, 4),
                "sigma": round(s, 4),
                "pairdist_cv": round(float(cv), 4),
                "mean_norm_share": round(float(
                    np.linalg.norm(X.mean(axis=0)) / nrm.mean()), 4),
                "phi_sigma_sens": {str(k): round(v, 4) for k, v in sens.items()},
                "graph_sigma_sens": {str(k): round(v, 4) for k, v in gsens.items()},
            }
        # comparison
        r, c = out["raw"], out["centred"]
        print(f"  --> Phi sigma-sens  raw mean "
              f"{np.mean(list(r['phi_sigma_sens'].values())):.4f}  "
              f"centred mean {np.mean(list(c['phi_sigma_sens'].values())):.4f}")
        print(f"  --> distance CV     raw {r['pairdist_cv']:.3f}  "
              f"centred {c['pairdist_cv']:.3f}")
        out["verdict"] = {
            "phi_sens_raw_mean": round(float(np.mean(list(r["phi_sigma_sens"].values()))), 4),
            "phi_sens_centred_mean": round(float(np.mean(list(c["phi_sigma_sens"].values()))), 4),
            # a difference below 1e-3 is numerical noise, not an improvement; an
            # earlier version compared floats directly and reported IMPROVED for a
            # gap of 0.0000, which was wrong
            "improved": bool(
                np.mean(list(r["phi_sigma_sens"].values())) -
                np.mean(list(c["phi_sigma_sens"].values())) > 1e-3),
            "identical_within_tolerance": bool(abs(
                np.mean(list(r["phi_sigma_sens"].values())) -
                np.mean(list(c["phi_sigma_sens"].values()))) < 1e-3),
        }
        report[arm] = out

    f = ROOT / "records_centroid" / f"graph_step2_{STAMP}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {f}")

    print("\n=== summary: does centring reduce sigma sensitivity? ===")
    for arm, o in report.items():
        v = o.get("verdict", {})
        tag = ("NO CHANGE" if v.get("identical_within_tolerance")
               else ("IMPROVED" if v.get("improved") else "WORSE"))
        print(f"  {arm:<22} raw {v.get('phi_sens_raw_mean')}  "
              f"centred {v.get('phi_sens_centred_mean')}  {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
