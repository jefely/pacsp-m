"""Locate the emotion-tree region a document activates, and check the pipeline.

Answers three questions from the latest direction (message 57 of the source chat):

    1. Interpretability — does the activation region match the text content?
    2. Sensitivity    — does the result survive changing the temperature (the one
                        declared parameter)?
    3. Discriminability — does human vs agent output deposit on different regions?

The projection is embedding similarity (no probe phrase, no next-token model), so the
only heavy dependency is the bge-large embedder already used everywhere else.
"""

import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from pacsp_emotion import (  # noqa: E402
    PROTOTYPES, _cluster_index, document_region, project_segment,
    segment_activations, tree_index, word_embeddings,
)
from pacsp_stream import segment_document  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODEL = "BAAI/bge-large-zh-v1.5"

ARMS = ["poem", "machine_poem", "lyrics", "machine_lyrics",
        "techdoc", "machine_techdoc2"]

TEMPS = [0.02, 0.05, 0.1, 0.2, 0.5]
DEFAULT_TEMP = 0.1


def dominant_clusters(unit_lists, emb_all, W, cluster, cidx, temp):
    """Return the dominant cluster per document at a given temperature."""
    doms = []
    idx = 0
    for us in unit_lists:
        e = emb_all[idx:idx + len(us)]
        idx += len(us)
        if len(us) == 0:
            continue
        A = segment_activations(np.asarray(e), W, temperature=temp)
        sed = A.sum(axis=0)
        c_sed = {c: float(sed[ix].sum()) for c, ix in cidx.items()}
        doms.append(max(c_sed, key=c_sed.get))
    return doms


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)
    embed = lambda texts: model.encode(list(texts), batch_size=8,
                                       normalize_embeddings=False)

    words, cluster, valence = tree_index()
    cidx = _cluster_index(cluster)
    W = word_embeddings(embed, words)
    print(f"emotion tree: {len(PROTOTYPES)} clusters, {len(words)} words\n")

    # --- 1. interpretability smoke test ------------------------------------------
    print("== 1. interpretability (hand-written segment -> top activated words) ==")
    probes = [
        "我很难过，失去了一切，心里空落落的。",
        "今天特别开心，多年的梦想终于实现了。",
        "他对这种不公感到愤怒和怨恨。",
        "深夜独自一人，感到深深的孤独与寂寞。",
        "实验结果令人震惊，完全出乎意料。",
    ]
    for seg in probes:
        v = embed([seg])[0]
        a = project_segment(v, W, temperature=DEFAULT_TEMP)
        top = [words[i] for i in np.argsort(-a)[:5]]
        print(f"  「{seg}」\n      -> {top}")

    # --- 2. per-corpus aggregation ----------------------------------------------
    print("\n== 2. dominant region + sediment concentration per corpus ==")
    rows = {}
    for arm in ARMS:
        files = sorted((DATA / arm).glob("*.txt"))
        dominant = Counter()
        concs = []
        for f in files:
            text = f.read_text(encoding="utf-8")
            r = document_region(text, embed, W, temperature=DEFAULT_TEMP)
            if r.get("n_segments", 0) == 0:
                continue
            dominant[r["dominant_cluster"]] += 1
            concs.append(r["sediment_concentration"])
        rows[arm] = {
            "n_docs": len(files),
            "dominant_cluster_counts": dict(dominant.most_common()),
            "mean_sediment_concentration": round(float(np.mean(concs)), 4)
            if concs else None,
        }
        top3 = dominant.most_common(3)
        print(f"  {arm:<20} n={len(files):>2}  top={top3}  "
              f"concentration={rows[arm]['mean_sediment_concentration']}")

    # --- 3. temperature sensitivity ---------------------------------------------
    print(f"\n== 3. temperature sensitivity (dominant-region agreement vs "
          f"temp={DEFAULT_TEMP}) ==")
    for arm in ["poem", "machine_poem", "lyrics", "machine_lyrics"]:
        files = sorted((DATA / arm).glob("*.txt"))
        unit_lists = [segment_document(f.read_text(encoding="utf-8")) for f in files]
        all_units = [u for us in unit_lists for u in us]
        emb_all = embed(all_units)

        base = dominant_clusters(unit_lists, emb_all, W, cluster, cidx, DEFAULT_TEMP)
        agreement = []
        for temp in TEMPS:
            doms = dominant_clusters(unit_lists, emb_all, W, cluster, cidx, temp)
            ag = np.mean([1 if a == b else 0 for a, b in zip(base, doms)])
            agreement.append(round(float(ag), 3))
        print(f"  {arm:<18} temps={[str(t) for t in TEMPS]}")
        print(f"  {'':<18} agree={agreement}")

    out = ROOT / "results" / "emotion_region.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
