"""Bootstrap intervals for the emotion-tree sediment-concentration human/machine ratio.

The point estimates in EMOTION-TREE-DYNAMIC-REGION.md §5.3 show machine output deposits
emotional activation MORE concentratedly (3/3 pairs). Per the framework's P1' rule, a
difference is only real if its interval excludes 1. This computes that interval by
bootstrapping over documents.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from pacsp_emotion import document_region, tree_index, word_embeddings  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODEL = "BAAI/bge-large-zh-v1.5"
PAIRS = [
    ("poem", "machine_poem"),
    ("lyrics", "machine_lyrics"),
    ("techdoc", "machine_techdoc2"),
]
BOOT = 2000
SEED = 7


def ratio_ci(a, b):
    rng = np.random.default_rng(SEED)
    ratios = []
    for _ in range(BOOT):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        if sb.mean() == 0:
            continue
        ratios.append(sa.mean() / sb.mean())
    ratios = np.asarray(ratios)
    return (float(np.mean(a) / np.mean(b)),
            float(np.percentile(ratios, 2.5)),
            float(np.percentile(ratios, 97.5)),
            float(ratios.std() / ratios.mean()))


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)
    embed = lambda t: model.encode(list(t), batch_size=8, normalize_embeddings=False)
    words, cluster, valence = tree_index()
    W = word_embeddings(embed, words)

    rows = []
    for arm_a, arm_b in PAIRS:
        concs = {}
        for arm in (arm_a, arm_b):
            vals = []
            for f in sorted((DATA / arm).glob("*.txt")):
                r = document_region(f.read_text(encoding="utf-8"), embed, W,
                                    temperature=0.1)
                if r.get("n_segments", 0) > 0:
                    vals.append(r["sediment_concentration"])
            concs[arm] = np.asarray(vals)
        r, lo, hi, cv = ratio_ci(concs[arm_a], concs[arm_b])
        row = {
            "pair": f"{arm_a}/{arm_b}",
            "human_concentration": round(float(concs[arm_a].mean()), 4),
            "machine_concentration": round(float(concs[arm_b].mean()), 4),
            "ratio": round(r, 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "ratio_cv": round(cv, 4),
        }
        rows.append(row)
        excl = "排除 1" if (lo > 1 or hi < 1) else "含 1"
        print(f"  {row['pair']:<32} human={row['human_concentration']:.4f} "
              f"machine={row['machine_concentration']:.4f} ratio={row['ratio']:.4f} "
              f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]  {excl}")

    out = ROOT / "results" / "emotion_region_intervals.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
