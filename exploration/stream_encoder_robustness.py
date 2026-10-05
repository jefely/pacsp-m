"""Encoder robustness for S_sed (mirrors §4.9 of the paper for D).

S_sed is "zero-parameter" only relative to a given reference frame Ω, and the encoder
is the one undeclared parameter that matters. This checks whether the length-matched
techdoc finding (human S_sed > machine S_sed) survives switching encoder family, the
same check the paper applies to D.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from pacsp_core import compute_embeddings  # noqa: E402
from pacsp_stream import segment_document  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = [
    "BAAI/bge-large-zh-v1.5",
    "BAAI/bge-small-zh-v1.5",
    "shibing624/text2vec-base-chinese",
]
K = 10


def sed_for_prefix(emb, k):
    e = emb[:k]
    n = e.shape[0]
    steps = np.linalg.norm(e[1:] - e[:-1], axis=1)
    sflow = float(steps.mean())
    recur = np.empty(n - 1, dtype=float)
    for i in range(1, n):
        recur[i - 1] = np.linalg.norm(e[:i] - e[i], axis=1).min()
    return float(recur.mean()) / sflow


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
    # pre-segment once
    docs = {"techdoc": [], "machine_techdoc2": []}
    for arm in docs:
        for f in sorted((DATA / arm).glob("*.txt")):
            units = segment_document(f.read_text(encoding="utf-8"))
            if len(units) >= K:
                docs[arm].append(units)

    rows = []
    for model in MODELS:
        vals = {}
        for arm, unit_lists in docs.items():
            allu = [u for us in unit_lists for u in us]
            emb = compute_embeddings(allu, model_name=model)
            idx = 0
            sed = []
            for us in unit_lists:
                e = emb[idx:idx + len(us)]
                idx += len(us)
                sed.append(sed_for_prefix(e, K))
            vals[arm] = np.asarray(sed)
        r, lo, hi = ratio_ci(vals["techdoc"], vals["machine_techdoc2"])
        row = {"model": model, "S_sed_human": round(float(vals["techdoc"].mean()), 4),
               "S_sed_machine": round(float(vals["machine_techdoc2"].mean()), 4),
               "ratio": round(r, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)}
        rows.append(row)
        print(f"  {model:<34} human={row['S_sed_human']:.4f} "
              f"machine={row['S_sed_machine']:.4f} ratio={row['ratio']:.4f} "
              f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]")

    out = ROOT / "results" / "stream_encoder_robustness.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
