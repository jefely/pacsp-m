"""F6: is the probe measuring emotional structure, or just emotion-word density?

The phrase sensitivity found in F1 is large enough to explain away the discriminative
result: changing the probe wording moves the human-over-machine ratio by a factor of 826
and reverses its sign twice. That raises a specific question. A question-answer phrasing
may push the model to restate emotion words found in the passage, in which case the probe
mass is tracking how many emotion words the text contains, not how its emotions relate.

Three tests.

A  Correlation. Count emotion-word occurrences per text directly, and correlate with the
   probe mass for the same text. A high correlation means the probe largely reports word
   frequency.

B  Frequency-matched null. Build a null co-occurrence matrix in which every item keeps
   its own distribution over emotion words but item identity is broken, by averaging over
   a class-preserving shuffle of the item order across the probe dimensions. If the graph
   survives this null, the structure is not simply item-level word mixing.

C  Phrase comparison. Count emotion words in the text under each probe phrasing's
   behaviour, and check whether the phrasings that reverse the sign are the ones whose
   mass correlates most strongly with raw word counts.

If A is high and C separates the phrasings by their correlation, the probe is a
lexical-frequency instrument and the hierarchy built on it inherits that.
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
CACHE = ROOT / "records_centroid" / "probe_cache"
STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
LIMIT = 31

ARMS = ["poem", "machine_poem", "lyrics", "machine_lyrics",
        "techdoc", "hc3_human_medicine", "hc3_ai_medicine",
        "hc3_human_openqa", "hc3_ai_openqa"]

PROBE_LABELS = {0: "这段素材的核心情绪是", 1: "这段文字表达的情绪是",
                2: "这段素材主要体现什么情绪？答：", 3: "情绪："}

from emotion_tree_probe import (  # noqa: E402
    EMOTIONS, emotion_token_ids, load_model,
)


def count_emotion_words(texts, words):
    """Raw occurrence count of each emotion word in each text."""
    M = np.zeros((len(texts), len(words)), dtype=np.float64)
    for i, t in enumerate(texts):
        for j, w in enumerate(words):
            M[i, j] = t.count(w)
    return M


def main():
    os.environ.setdefault("HF_HOME", str(ROOT / "_hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "_hf_home" / "hub"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    tok, _, _ = load_model("Qwen/Qwen2.5-7B-Instruct")
    ids = emotion_token_ids(tok, EMOTIONS)
    words = list(ids.keys())

    out = {"vocab": len(words)}

    # ------------------------------------------------------------------ A and C
    print("=== A/C: probe mass versus raw emotion-word counts ===")
    print(f"  {'arm':<22} {'words':>7} {'sum Y':>9} " +
          "".join(f"{'r_p' + str(p):>9}" for p in PROBE_LABELS))
    rows = {}
    for arm in ARMS:
        p = DATA / arm
        if not p.is_dir():
            continue
        texts, _ = load_samples(p)
        texts = texts[:LIMIT]
        M = count_emotion_words(texts, words)
        wc = M.sum(axis=1)                      # emotion words per item
        rec = {"total_emotion_words": float(wc.sum()),
               "words_per_item": round(float(wc.mean()), 2)}
        cors = {}
        for pi in PROBE_LABELS:
            f = CACHE / f"{arm}__p{pi}.npy"
            if not f.exists():
                continue
            Y = np.load(f)
            y = Y.sum(axis=1)
            # Pearson r between per-item word count and per-item probe mass
            if wc.std() > 0 and y.std() > 0:
                r = float(np.corrcoef(wc, y)[0, 1])
            else:
                r = float("nan")
            cors[pi] = round(r, 4)
            rec[f"r_p{pi}"] = None if np.isnan(r) else round(r, 4)
        rec["correlations"] = cors
        rows[arm] = rec
        print(f"  {arm:<22} {int(wc.sum()):>7} "
              f"{sum(np.load(CACHE / f'{arm}__p{pi}.npy').sum() for pi in PROBE_LABELS if (CACHE / f'{arm}__p{pi}.npy').exists()):>9.2f} " +
              "".join(f"{cors.get(p, float('nan')):>9.3f}" for p in PROBE_LABELS))
    out["arms"] = rows

    # correlation pooled across arms, per phrasing
    print()
    print("  pooled correlation by phrasing:")
    pooled = {}
    for pi in PROBE_LABELS:
        xs, ys = [], []
        for arm in rows:
            f = CACHE / f"{arm}__p{pi}.npy"
            if not f.exists():
                continue
            texts, _ = load_samples(DATA / arm)
            texts = texts[:LIMIT]
            wc = count_emotion_words(texts, words).sum(axis=1)
            Y = np.load(f)
            # normalise within arm, since arms differ in scale
            if wc.std() > 0 and Y.sum(axis=1).std() > 0:
                xs.append((wc - wc.mean()) / wc.std())
                y = Y.sum(axis=1)
                ys.append((y - y.mean()) / y.std())
        if xs:
            x = np.concatenate(xs); y = np.concatenate(ys)
            r = float(np.corrcoef(x, y)[0, 1])
            pooled[pi] = round(r, 4)
            print(f"    {PROBE_LABELS[pi]:<28} r = {r:>7.4f}  (n={len(x)})")
    out["pooled_correlation"] = pooled

    # ------------------------------------------------------------------ B
    print()
    print("=== B: class-preserving shuffle null ===")
    print("  the null keeps each item's marginal over emotions and each emotion's total,")
    print("  but breaks the item-to-emotion assignment jointly")
    rng = np.random.default_rng(3)
    b_out = {}
    for arm in ARMS:
        f = CACHE / f"{arm}__p0.npy"
        if not f.exists():
            continue
        Y = np.load(f)
        # column-wise shuffle within each item: preserves the item's total mass but
        # redistributes it across emotions without regard to the item's content
        Yn = np.empty_like(Y)
        for i in range(Y.shape[0]):
            Yn[i] = rng.permutation(Y[i])
        # Sinkhorn-style rescale so column sums are also preserved
        for _ in range(60):
            rs = Y.sum(axis=1, keepdims=True) / np.maximum(Yn.sum(axis=1, keepdims=True), 1e-12)
            Yn *= rs
            cs = Y.sum(axis=0, keepdims=True) / np.maximum(Yn.sum(axis=0, keepdims=True), 1e-12)
            Yn *= cs
        C = Y.T @ Y
        Cn = Yn.T @ Yn
        keep = C > 0
        Cn = np.where(keep, Cn, 0.0)
        # compare the top edges of each
        k = 200
        top_real = set(map(tuple, np.dstack(np.unravel_index(
            np.argsort(C.ravel())[::-1][:k], C.shape))[0].tolist()))
        top_null = set(map(tuple, np.dstack(np.unravel_index(
            np.argsort(Cn.ravel())[::-1][:k], Cn.shape))[0].tolist()))
        inter = len(top_real & top_null)
        b_out[arm] = {"top{k}_overlap".replace("{k}", str(k)): inter,
                      "top_share": round(inter / k, 4),
                      "row_sum_rel_err": round(float(
                          np.abs(Yn.sum(axis=1) - Y.sum(axis=1)).max() /
                          Y.sum(axis=1).mean()), 4),
                      "col_sum_rel_err": round(float(
                          np.abs(Yn.sum(axis=0) - Y.sum(axis=0)).max() /
                          Y.sum(axis=0).mean()), 4)}
        print(f"  {arm:<22} top-{k} overlap {inter:>4} ({inter/k:.1%})  "
              f"row-sum err {b_out[arm]['row_sum_rel_err']:.4f}  "
              f"col-sum err {b_out[arm]['col_sum_rel_err']:.4f}")
    out["shuffle_null"] = b_out

    print()
    print("=== interpretation ===")
    pr = [v for v in pooled.values()]
    if pr:
        print(f"  pooled correlation ranges {min(pr):.3f} to {max(pr):.3f}")
    shares = [v["top_share"] for v in b_out.values()]
    if shares:
        print(f"  shuffle-null top-edge overlap {min(shares):.1%} to {max(shares):.1%}")
        print("  a high overlap means the co-occurrence pattern follows from the")
        print("  item marginals alone, with content order irrelevant")

    f = ROOT / "records_centroid" / f"f6_mechanism_{STAMP}.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
