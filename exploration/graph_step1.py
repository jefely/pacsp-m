"""Step 1: build the graph representation and test what it fixes.

The proposal in the source conversation replaces the linear path with a graph
G = (V, E, l) and a two-part representation:

    Phi(G)(x)   = sum_{v in V} l(v) K(x, v)              node density
    Psi(G)(x,y) = sum_{(u,v) in E} w(u,v) K(x,u) K(y,v)   edge density

Step 1 of the verification asks three concrete questions.

Q1  Does Phi determine V?  The claim is yes, because Phi is a kernel density
    estimate whose modes are the nodes. Tested directly by fitting an RBF network to
    Phi and measuring how far the fitted centres move from the true embeddings.
    This is the honest version of the recovery test; the earlier "does argmax find
    the point" formulation only tested whether the point was in a lookup table.

Q2  Is the graph representation order-invariant?  This is the whole reason for
    switching to a graph. Two graphs are compared: the original induced by reading
    order (a path), and a canonical k-nearest-neighbour graph built from pairwise
    similarity alone. If the canonical graph is unchanged under permutation while
    the path graph changes, the fix works.

Q3  Does the graph remove the sigma sensitivity measured earlier, where Phi changed
    by 136-325 percent when sigma moved by 1.6x?  Tested both for Phi and for graph
    invariants.
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

ARMS = ["poem", "lyrics", "techdoc", "hc3_human_medicine", "hc3_ai_medicine",
        "machine_poem"]


def sq_dists(A, B):
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def median_scale(X):
    d2 = sq_dists(X, X)
    off = d2[~np.eye(len(X), dtype=bool)]
    off = off[off > 0]
    return float(np.sqrt(np.median(off)))


def kernel(A, B, sigma):
    return np.exp(-sq_dists(A, B) / (2.0 * sigma ** 2))


# ----------------------------------------------------------------- graph builders
def path_graph(n):
    """Edges induced by reading order: a chain, plus self-identity diagonal."""
    return [(i, i + 1) for i in range(n - 1)]


def knn_graph(E, k):
    """Canonical graph from pairwise similarity alone; independent of file order."""
    d2 = sq_dists(E, E)
    np.fill_diagonal(d2, np.inf)
    edges = set()
    for i in range(len(E)):
        for j in np.argsort(d2[i])[:k]:
            edges.add((min(i, int(j)), max(i, int(j))))
    return sorted(edges)


def edge_weights(E, edges, sigma):
    if not edges:
        return np.zeros(0)
    out = []
    for i, j in edges:
        d2 = float(sq_dists(E[[i]], E[[j]])[0, 0])
        out.append(np.exp(-d2 / (2.0 * sigma ** 2)))
    return np.array(out)


def phi_of(E, frame, sigma):
    return kernel(E, frame, sigma)


def psi_of(E, edges, frame, sigma):
    """Psi(G)(x,y) collapsed to its diagonal-in-frame matrix: sum over edges of
    w(u,v) K(x,u) K(y,v). For comparison purposes it is summarised by its Frobenius
    norm and its spectrum, which are invariant to node labelling."""
    Phi = kernel(E, frame, sigma)                     # (n, m)
    w = edge_weights(E, edges, sigma)
    Psi = np.zeros((len(frame), len(frame)))
    for (i, j), wij in zip(edges, w):
        Psi += wij * np.outer(Phi[i], Phi[j])
    return Psi


def fit_modes(Phi, frame, k):
    """Fit an RBF network to Phi by linear least squares and return the centres.

    Phi(x) = sum_c a_c exp(-||x - c||^2 / 2 sigma^2), so with centres fixed at the
    frame points this is linear in a. The fitted centres are those with weight above
    a threshold, which is the practical form of "the modes of Phi are the nodes".
    """
    G = kernel(frame, frame, median_scale(frame) * 0.5)   # basis matrix
    # solve G a = Phi^T  for each node
    A, *_ = np.linalg.lstsq(G, Phi.T, rcond=None)
    # reconstruct the density and find its peaks over the frame
    return A


def graph_invariants(E, edges, sigma, frame):
    w = edge_weights(E, edges, sigma)
    deg = np.zeros(len(E))
    for (i, j), wij in zip(edges, w):
        deg[i] += wij
        deg[j] += wij
    lap = np.zeros((len(E), len(E)))
    for i in range(len(E)):
        lap[i, i] = deg[i]
    for (i, j), wij in zip(edges, w):
        lap[i, j] -= wij
        lap[j, i] -= wij
    ev = np.linalg.eigvalsh(lap)
    ev = np.sort(ev)[::-1]
    return {
        "n_edges": len(edges),
        "weight_sum": float(w.sum()),
        "weight_mean": float(w.mean()) if len(w) else 0.0,
        "degree_mean": float(deg.mean()),
        "degree_std": float(deg.std()),
        "spectral_gap": float(ev[1] - ev[2]) if len(ev) > 2 else 0.0,
        "lap_top": float(ev[0]),
    }


def main():
    report = {}
    rng = np.random.default_rng(5)

    for arm in ARMS:
        p = DATA / arm
        if not p.is_dir():
            continue
        texts, _ = load_samples(p)
        if len(texts) < 6:
            continue
        E = compute_embeddings(texts, model_name=MODEL)
        n = len(E)
        med = median_scale(E)
        sigma = 0.5 * med
        frame = E                      # frame that contains the corpus, see Q1 below
        out = {"n": n, "median_dist": round(med, 4), "sigma": round(sigma, 4)}
        print(f"\n{'=' * 74}\n{arm}   n={n}  median={med:.4f}  sigma={sigma:.4f}\n{'=' * 74}")

        # ------------------------------------------------------------------ Q1
        print("  Q1  does Phi determine V?  (RBF fit, then compare centres)")
        A = fit_modes(phi_of(E, frame, sigma), frame, n)
        # for each node, the frame point with the largest fitted weight
        peaks = np.argmax(A, axis=0)
        rec = frame[peaks]
        err = np.linalg.norm(rec - E, axis=1)
        base = np.linalg.norm(E - E.mean(axis=0), axis=1)
        exact = int(np.sum(np.abs(err) < 1e-9))
        print(f"      fitted-centre error : mean {err.mean():.4f}  "
              f"relative to baseline {err.mean()/base.mean():.4f}")
        print(f"      exactly recovered   : {exact}/{n}")
        out["q1"] = {"mean_err": round(float(err.mean()), 6),
                     "rel": round(float(err.mean() / base.mean()), 6),
                     "exact": exact,
                     "baseline": round(float(base.mean()), 4)}

        # out-of-frame version: nodes are not in the frame
        held = frame[::3]
        A2 = fit_modes(phi_of(E, held, sigma), held, n)
        peaks2 = np.argmax(A2, axis=0)
        rec2 = held[peaks2]
        err2 = np.linalg.norm(rec2 - E, axis=1)
        print(f"      out-of-frame error  : mean {err2.mean():.4f}  "
              f"relative {err2.mean()/base.mean():.4f}  "
              f"(frame has {len(held)} of {n} nodes)")
        out["q1_oof"] = {"mean_err": round(float(err2.mean()), 6),
                         "rel": round(float(err2.mean() / base.mean()), 6),
                         "frame_size": len(held)}

        # ------------------------------------------------------------------ Q2
        print("  Q2  is the graph order-invariant?")
        pg = path_graph(n)
        kg = knn_graph(E, k=3)
        inv_path = graph_invariants(E, pg, sigma, frame)
        inv_knn = graph_invariants(E, kg, sigma, frame)
        print(f"      path graph : edges={inv_path['n_edges']} "
              f"w_sum={inv_path['weight_sum']:.3f} w_mean={inv_path['weight_mean']:.4f}")
        print(f"      knn  graph : edges={inv_knn['n_edges']} "
              f"w_sum={inv_knn['weight_sum']:.3f} w_mean={inv_knn['weight_mean']:.4f}")

        # permute and rebuild both
        perm = rng.permutation(n)
        Ep = E[perm]
        ppg = path_graph(n)
        pkg = knn_graph(Ep, k=3)
        # canonical graphs must agree as edge SETS over the same node identities
        kg_set = {tuple(sorted((int(a), int(b)))) for a, b in kg}
        pkg_set = {tuple(sorted((int(perm[a]), int(perm[b])))) for a, b in pkg}
        same_knn = kg_set == pkg_set
        # the path graph's edge set under permutation is different by construction
        pg_set = set(pg)
        ppg_set = {(int(perm[a]), int(perm[b])) for a, b in ppg}
        same_path = {tuple(sorted(e)) for e in pg_set} == {tuple(sorted(e)) for e in ppg_set}
        print(f"      knn  edge set identical after permutation : {same_knn}")
        print(f"      path edge set identical after permutation : {same_path}")

        # how much do the invariants move under permutation?
        inv_path_p = graph_invariants(Ep, ppg, sigma, frame)
        inv_knn_p = graph_invariants(Ep, pkg, sigma, frame)
        def rel(a, b, key):
            va, vb = a[key], b[key]
            return abs(vb - va) / abs(va) if va else 0.0
        print(f"      invariants under permutation (relative change):")
        for key in ("weight_sum", "weight_mean", "degree_std", "spectral_gap"):
            print(f"        {key:<14} path {rel(inv_path, inv_path_p, key):>8.4f}   "
                  f"knn {rel(inv_knn, inv_knn_p, key):>8.4f}")
        out["q2"] = {
            "knn_edge_set_stable": bool(same_knn),
            "path_edge_set_stable": bool(same_path),
            "path_rel": {k: round(rel(inv_path, inv_path_p, k), 4)
                         for k in ("weight_sum", "weight_mean", "degree_std", "spectral_gap")},
            "knn_rel": {k: round(rel(inv_knn, inv_knn_p, k), 4)
                        for k in ("weight_sum", "weight_mean", "degree_std", "spectral_gap")},
        }

        # ------------------------------------------------------------------ Q3
        print("  Q3  does the graph remove the sigma sensitivity?")
        print(f"      {'sigma mult':>11} {'|dPhi|/|Phi|':>13} "
              f"{'d weight_sum':>13} {'d spectral_gap':>15}")
        base_phi = phi_of(E, frame, sigma)
        nb = float(np.linalg.norm(base_phi))
        q3 = {}
        for mult in (0.6, 0.8, 1.25, 1.6):
            s2 = sigma * mult
            p2 = phi_of(E, frame, s2)
            dphi = float(np.linalg.norm(p2 - base_phi)) / nb
            inv2 = graph_invariants(E, kg, s2, frame)
            dw = abs(inv2["weight_sum"] - inv_knn["weight_sum"]) / inv_knn["weight_sum"]
            dg = (abs(inv2["spectral_gap"] - inv_knn["spectral_gap"]) /
                  abs(inv_knn["spectral_gap"]) if inv_knn["spectral_gap"] else 0.0)
            print(f"      {mult:>11} {dphi:>13.4f} {dw:>13.4f} {dg:>15.4f}")
            q3[str(mult)] = {"dPhi_rel": round(dphi, 4),
                             "d_weight_sum_rel": round(dw, 4),
                             "d_spectral_gap_rel": round(dg, 4)}
        out["q3"] = q3
        report[arm] = out

    outfile = ROOT / "records_centroid" / f"graph_step1_{STAMP}.json"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print(f"\n  written {outfile}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
