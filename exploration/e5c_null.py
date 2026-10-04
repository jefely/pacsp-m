"""Two corrections: a fair null model, and paired ratios for the new features.

Correction 1. The earlier rank-1 comparison was not fair. The real C = Y^T Y with 12
items has rank at most 12 and is sparse; an outer product r c^T is full rank and dense,
and a denser matrix produces edges more easily. So the high overlap of 73 to 89 percent
may reflect the density difference rather than the marginals alone. The correct null is
the Chung-Lu model, which preserves the row and column sums and keeps the same sparsity
pattern by thresholding on the product of the marginals:

    E_ab = r_a c_b / W      (the expected count if a and b were independent)

The graph is then rebuilt on this null and compared with the real graph, so what is
left is structure that the marginals do not explain.

Correction 2. The e5b features are reported as raw values. What matters for the paper
is the same-domain human-versus-machine contrast, which needs paired ratios.

The residual idea is also implemented: the null-expected edge weight is subtracted from
the real one, and node and edge volume are recomputed on what remains.
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pacsp_core import load_samples  # noqa: E402

DATA = ROOT / "data"
STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
LM = "Qwen/Qwen2.5-7B-Instruct"
LIMIT = 31
T = 0.02

ARMS = [("poem", "poem", "human"), ("machine_poem", "poem", "machine"),
        ("lyrics", "lyrics", "human"), ("machine_lyrics", "lyrics", "machine"),
        ("techdoc", "techdoc", "human"),
        ("hc3_human_medicine", "medicine", "human"),
        ("hc3_ai_medicine", "medicine", "machine"),
        ("hc3_human_openqa", "openqa", "human"),
        ("hc3_ai_openqa", "openqa", "machine")]

from emotion_tree_probe import (  # noqa: E402
    EMOTIONS, build_hierarchy, emotion_token_ids, load_model, probe_matrix,
)


def null_matrix(C):
    """Chung-Lu: expected value under independence, same margins, same total."""
    r = C.sum(axis=1, keepdims=True)
    c = C.sum(axis=0, keepdims=True)
    W = C.sum()
    return (r @ c) / W


def graph_from(C, words, t=T):
    """Apply the described rule, restricted to positive entries."""
    V = C.shape[0]
    rs = C.sum(axis=1)
    cs = C.sum(axis=0)
    edges = []
    for a in range(V):
        for b in range(V):
            if a == b or rs[a] <= 0 or cs[b] <= 0 or C[a, b] <= 0:
                continue
            if C[a, b] / rs[a] > t and rs[a] < cs[b]:
                edges.append((a, b, float(C[a, b])))
    return edges


def volumes(edges, Y):
    node = float(Y.sum())
    edge = float(sum(w for _, _, w in edges))
    return node, edge


def main():
    os.environ.setdefault("HF_HOME", str(ROOT / "_hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "_hf_home" / "hub"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    tok, mdl, torch = load_model(LM)
    ids = emotion_token_ids(tok, EMOTIONS)
    print(f"  emotion words as single tokens: {len(ids)}")

    rep = {}
    for arm, dom, kind in ARMS:
        p = DATA / arm
        if not p.is_dir():
            continue
        texts, _ = load_samples(p)
        texts = texts[:LIMIT]
        print(f"\n=== {arm}  n={len(texts)} ===")
        Y, words, _ = probe_matrix(texts, tok, mdl, torch, ids)
        C = Y.T @ Y
        V = len(words)

        real = graph_from(C, words)
        # the null: same margins, no interaction
        Cn = null_matrix(C)
        # keep only entries that are positive in the real matrix, so sparsity matches
        mask = C > 0
        Cn_masked = np.where(mask, Cn, 0.0)
        null = graph_from(Cn_masked, words)

        rs_set = {(a, b) for a, b, _ in real}
        ns_set = {(a, b) for a, b, _ in null}
        inter = len(rs_set & ns_set)
        denom = max(1, len(rs_set))
        node_v, edge_v = volumes(real, Y)

        # how dense is each matrix, which is what made the earlier test unfair
        dens_real = float((C > 0).mean())
        dens_null = float((Cn_masked > 0).mean())

        # residual volumes: real minus null expectation
        Cn_full = Cn_masked
        resid = C - Cn_full
        resid_edges = [(a, b, max(0.0, float(resid[a, b])))
                       for a, b, _ in real]
        resid_edge_v = float(sum(w for _, _, w in resid_edges))
        resid_node_v = float(np.maximum(Y - Y.mean(axis=0, keepdims=True), 0).sum())

        rep[arm] = {
            "domain": dom, "kind": kind, "n": len(texts),
            "real_edges": len(real), "null_edges": len(null),
            "edge_overlap": inter,
            "overlap_share_of_real": round(inter / denom, 4),
            "density_real": round(dens_real, 4),
            "density_null": round(dens_null, 4),
            "node_volume": round(node_v, 4),
            "edge_volume": round(edge_v, 6),
            "resid_node_volume": round(resid_node_v, 4),
            "resid_edge_volume": round(resid_edge_v, 6),
        }
        d = rep[arm]
        print(f"    real {d['real_edges']:>4} edges  null {d['null_edges']:>4}  "
              f"overlap {d['edge_overlap']:>4} ({d['overlap_share_of_real']:.1%})  "
              f"density real {d['density_real']:.3f} null {d['density_null']:.3f}")
        print(f"    node_v {d['node_volume']:>8.4f}   edge_v {d['edge_volume']:.6f}   "
              f"resid_node {d['resid_node_volume']:>8.4f}   "
              f"resid_edge {d['resid_edge_volume']:.6f}")

    print()
    print("=== same-domain human / machine ratios ===")
    print(f"  {'domain':<10} " + "".join(f"{k:>14}" for k in
          ("node_volume", "edge_volume", "resid_node", "resid_edge")))
    pairs = {}
    for dom in sorted({v["domain"] for v in rep.values()}):
        hs = [a for a, v in rep.items() if v["domain"] == dom and v["kind"] == "human"]
        ms = [a for a, v in rep.items() if v["domain"] == dom and v["kind"] == "machine"]
        if not (hs and ms):
            continue
        h, m = rep[hs[0]], rep[ms[0]]
        row = []
        for k in ("node_volume", "edge_volume"):
            row.append(h[k] / m[k] if m[k] else float("nan"))
        for k in ("resid_node_volume", "resid_edge_volume"):
            row.append(h[k] / m[k] if m[k] else float("nan"))
        pairs[dom] = row
        print(f"  {dom:<10} " + "".join(f"{v:>14.4f}" for v in row))

    print()
    print("  (values below 1 mean the human arm scores lower)")
    print()
    print("=== how consistent is each measure across the five domains? ===")
    for i, k in enumerate(("node_volume", "edge_volume",
                           "resid_node_volume", "resid_edge_volume")):
        vals = np.array([pairs[d][i] for d in pairs if not np.isnan(pairs[d][i])])
        if len(vals) == 0:
            continue
        same_dir = np.all(vals < 1) or np.all(vals > 1)
        print(f"  {k:<20} min {vals.min():.4f}  max {vals.max():.4f}  "
              f"ratio {vals.max()/vals.min() if vals.min() else float('inf'):.2f}  "
              f"direction consistent: {same_dir}")

    print()
    print("=== corrected verdict on the null comparison ===")
    shares = [v["overlap_share_of_real"] for v in rep.values()]
    print(f"  overlap share of real edges: min {min(shares):.1%}  max {max(shares):.1%}")
    print(f"  density real: {min(v['density_real'] for v in rep.values()):.3f}"
          f"-{max(v['density_real'] for v in rep.values()):.3f}")
    print(f"  density null: {min(v['density_null'] for v in rep.values()):.3f}"
          f"-{max(v['density_null'] for v in rep.values()):.3f}")

    out = ROOT / "records_centroid" / f"e5c_null_{STAMP}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
