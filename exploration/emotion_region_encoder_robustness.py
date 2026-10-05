"""Encoder robustness for the emotion-tree sediment-concentration ratio (§4.9 style).

The 3/3 interval-excluding concentration result was measured with bge-large only. This
checks whether the human/machine direction survives switching encoder (family), the same
check the paper applies to D (where cross-family degraded the effect).
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
MODELS = [
    "BAAI/bge-large-zh-v1.5",
    "BAAI/bge-small-zh-v1.5",
    "shibing624/text2vec-base-chinese",
]
PAIRS = [
    ("poem", "machine_poem"),
    ("lyrics", "machine_lyrics"),
    ("techdoc", "machine_techdoc2"),
]


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
    rows = []
    for model_name in MODELS:
        model = SentenceTransformer(model_name)
        embed = lambda t: model.encode(list(t), batch_size=8, normalize_embeddings=False)
        words, cluster, valence = tree_index()
        W = word_embeddings(embed, words)
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
            ratio, lo, hi = ratio_ci(concs[arm_a], concs[arm_b])
            excl = "排除1" if (lo > 1 or hi < 1) else "含1"
            rows.append({"model": model_name, "pair": f"{arm_a}/{arm_b}",
                         "ratio": round(ratio, 4), "ci_low": round(lo, 4),
                         "ci_high": round(hi, 4)})
            print(f"  {model_name:<34} {arm_a+'/'+arm_b:<28} "
                  f"ratio={ratio:.4f} [{lo:.4f}, {hi:.4f}]  {excl}")

    out = ROOT / "results" / "emotion_region_encoder_robustness.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
