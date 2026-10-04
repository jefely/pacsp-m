"""E5b: put the candidate representations on the same footing as the baseline.

E5 established that mean_pair_dist, a zero-parameter order-free statistic, separates
human from machine more consistently than C_T. Any replacement representation has to
beat that baseline, not merely have fewer problems. This adds the emotion-tree
features, built from real model logits, and the kernel embedding features, next to the
baseline measures so the comparison is direct.

Emotion-tree features, per corpus:
    node volume      sum over items and emotion words of the probe probabilities
    edge volume      sum of weights over the induced hierarchy edges
    hierarchy size   number of induced edges
    tree depth       longest path in the induced DAG
    top hub share    fraction of edge weight concentrated on the single largest sink

All but the last are order-free by construction; the probe matrix is a set-level
statistic.

Kernel embedding features, per corpus, from the same embeddings the baseline uses:
    phi_norm         Frobenius norm of Phi at the protocol sigma
    sigma_sensitivity  relative change in Phi when sigma moves by 1.6x
"""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
from pacsp_core import compute_embeddings, load_samples  # noqa: E402

DATA = ROOT / "data"
EMB_MODEL = "BAAI/bge-large-zh-v1.5"
LM = "Qwen/Qwen2.5-7B-Instruct"
STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
LIMIT = 12

ARMS = ["poem", "machine_poem", "lyrics", "machine_lyrics",
        "techdoc", "hc3_ai_medicine"]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from emotion_tree_probe import (  # noqa: E402
    EMOTIONS, PROBE, build_hierarchy, emotion_token_ids, load_model, probe_matrix,
)


def sq(A, B):
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def depth_of(edges, V):
    """Longest path length in the induced DAG, by repeated relaxation."""
    if not edges:
        return 0
    adj = {}
    for a, b, _ in edges:
        adj.setdefault(a, []).append(b)
    depth = {i: 0 for i in range(V)}
    for _ in range(V):
        changed = False
        for a, bs in adj.items():
            for b in bs:
                if depth[b] < depth[a] + 1:
                    depth[b] = depth[a] + 1
                    changed = True
        if not changed:
            break
    return int(max(depth.values()))


def phi_features(E):
    n = len(E)
    d2 = sq(E, E)
    iu = np.triu_indices(n, k=1)
    med = float(np.median(np.sqrt(d2[iu])))
    sigma = 0.5 * med
    K = np.exp(-d2 / (2 * sigma ** 2))
    nrm = float(np.linalg.norm(K))
    K2 = np.exp(-d2 / (2 * (sigma * 1.6) ** 2))
    return {
        "phi_frobenius": round(nrm, 4),
        "phi_sigma_sens_1.6": round(float(np.linalg.norm(K2 - K)) / nrm, 4),
    }


def main():
    os.environ.setdefault("HF_HOME", str(ROOT / "_hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "_hf_home" / "hub"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    print(f"  loading {LM}")
    tok, mdl, torch = load_model(LM)
    ids = emotion_token_ids(tok, EMOTIONS)
    print(f"  emotion words as single tokens: {len(ids)}")

    report = {}
    for arm in ARMS:
        p = DATA / arm
        if not p.is_dir():
            continue
        texts, _ = load_samples(p)
        texts = texts[:LIMIT]
        print(f"\n=== {arm}  n={len(texts)} ===")
        Y, words, _ = probe_matrix(texts, tok, mdl, torch, ids)
        C = Y.T @ Y
        edges, rs, cs = build_hierarchy(C)
        V = len(words)
        ew = np.array([w for _, _, w in edges]) if edges else np.zeros(1)

        # hub share: weight landing on the single largest sink
        sink = {}
        for a, b, w in edges:
            sink[b] = sink.get(b, 0.0) + w
        hub = max(sink.values()) / ew.sum() if sink and ew.sum() else 0.0

        # order invariance, verified not assumed
        perm = np.random.default_rng(1).permutation(len(texts))
        C_perm = Y[perm].T @ Y[perm]

        feat = {
            "hierarchy_size": len(edges),
            "node_volume": round(float(Y.sum()), 6),
            "edge_volume": round(float(ew.sum()), 6),
            "tree_depth": depth_of(edges, V),
            "hub_share": round(float(hub), 4),
            "C_permutation_invariant": bool(np.allclose(C, C_perm, atol=1e-12)),
            "Y_row_sum_min": round(float(Y.sum(axis=1).min()), 6),
            "Y_row_sum_max": round(float(Y.sum(axis=1).max()), 6),
            "top_sink": words[max(sink, key=sink.get)] if sink else None,
        }
        E = compute_embeddings(texts, model_name=EMB_MODEL)
        feat.update(phi_features(E))
        print(f"    {feat}")
        report[arm] = feat

    print()
    print("=== cross-arm comparison ===")
    keys = ["hierarchy_size", "node_volume", "edge_volume", "tree_depth",
            "hub_share", "phi_frobenius", "phi_sigma_sens_1.6"]
    print(f"  {'arm':<20} " + "".join(f"{k[:11]:>13}" for k in keys))
    for arm, f in report.items():
        print(f"  {arm:<20} " + "".join(f"{f.get(k, 0):>13.4f}" for k in keys))

    print()
    print("=== spread over arms (higher means better separation) ===")
    for k in keys:
        v = np.array([f.get(k, 0.0) for f in report.values()], dtype=float)
        if v.min() > 0:
            print(f"  {k:<22} min {v.min():>10.4f}  max {v.max():>10.4f}  "
                  f"ratio {v.max()/v.min():>7.2f}")
        else:
            print(f"  {k:<22} min {v.min():>10.4f}  max {v.max():>10.4f}  "
                  f"ratio n/a (zero present)")

    out = ROOT / "records_centroid" / f"e5b_candidates_{STAMP}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
