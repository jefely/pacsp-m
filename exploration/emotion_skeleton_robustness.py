"""Cross-skeleton robustness of the sediment-concentration direction.

The 13-cluster skeleton is a DECLARED choice, not a derived structure. To check whether
the "machine output deposits more concentratedly" direction is a property of the data or
an artifact of that particular skeleton, this re-aggregates the same per-document
activation onto two coarser skeletons:

    * valence-3: 正性 / 负性 / 中性
    * valence-2: 正性 / 负性 (惊讶-中性 merged into 负性 is avoided; instead 中性 is
                 folded into whichever of 正/负 is larger, i.e. a 2-way split)

Concentration under each skeleton is the share of sedimentation mass held by the single
largest node; the human/machine ratio and its bootstrap interval are reported per pair.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from pacsp_emotion import (  # noqa: E402
    _cluster_index, document_region, tree_index, word_embeddings,
)
from pacsp_stream import segment_document  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODEL = "BAAI/bge-large-zh-v1.5"
PAIRS = [
    ("poem", "machine_poem"),
    ("lyrics", "machine_lyrics"),
    ("techdoc", "machine_techdoc2"),
]


def concentration_by(sed_by_cluster, valence_of_cluster):
    """Aggregate per-cluster mass onto a grouping and return the max node share."""
    agg = {}
    for c, mass in sed_by_cluster.items():
        agg[valence_of_cluster[c]] = agg.get(valence_of_cluster[c], 0.0) + mass
    total = sum(agg.values()) or 1.0
    return max(agg.values()) / total


def ratio_ci(a, b, seed=7):
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(2000):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        if sb.mean() == 0:
            continue
        ratios.append(sa.mean() / sb.mean())
    ratios = np.asarray(ratios)
    return (float(np.mean(a) / np.mean(b)),
            float(np.percentile(ratios, 2.5)),
            float(np.percentile(ratios, 97.5)))


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)
    embed = lambda t: model.encode(list(t), batch_size=8, normalize_embeddings=False)
    words, cluster, valence = tree_index()
    W = word_embeddings(embed, words)

    # valence of each cluster prototype
    valence_of_cluster = {p: v for p, v, _ in __import__('pacsp_emotion').EMOTION_CLUSTERS}

    # skeleton 2: 正性/负性 only (中性 folded into 正性? no -- fold into whichever of
    # 正/负 is larger per document is data-dependent; instead use a fixed rule: 中性->负性
    # is arbitrary, so keep valence-2 as 正性 vs (负性+中性), a defensible coarse split)
    def valence2(v):
        return "正性" if v == "正性" else "负性/中性"

    print(f"  pair                          skeleton      human   machine  ratio   interval")
    rows = []
    for arm_a, arm_b in PAIRS:
        for skel_name, grouper in [("13-cluster", None),
                                   ("valence-3", lambda c: valence_of_cluster[c]),
                                   ("valence-2", valence2)]:
            concs = {}
            for arm in (arm_a, arm_b):
                vals = []
                for f in sorted((DATA / arm).glob("*.txt")):
                    r = document_region(f.read_text(encoding="utf-8"), embed, W,
                                        temperature=0.1)
                    if r.get("n_segments", 0) == 0:
                        continue
                    if grouper is None:
                        vals.append(r["sediment_concentration"])
                    else:
                        grouped = {}
                        for c, mass in r["cluster_sedimentation"].items():
                            g = grouper(c)
                            grouped[g] = grouped.get(g, 0.0) + mass
                        total = sum(grouped.values()) or 1.0
                        vals.append(max(grouped.values()) / total)
                concs[arm] = np.asarray(vals)
            ratio, lo, hi = ratio_ci(concs[arm_a], concs[arm_b])
            excl = "排除1" if (lo > 1 or hi < 1) else "含1"
            rows.append({"pair": f"{arm_a}/{arm_b}", "skeleton": skel_name,
                         "human": round(float(concs[arm_a].mean()), 4),
                         "machine": round(float(concs[arm_b].mean()), 4),
                         "ratio": round(ratio, 4), "ci_low": round(lo, 4),
                         "ci_high": round(hi, 4), "excludes_1": excl == "排除1"})
            print(f"  {arm_a+'/'+arm_b:<30} {skel_name:<13} "
                  f"{concs[arm_a].mean():.4f}  {concs[arm_b].mean():.4f}  "
                  f"{ratio:.4f}  [{lo:.4f}, {hi:.4f}]  {excl}")

    out = ROOT / "results" / "emotion_skeleton_robustness.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
