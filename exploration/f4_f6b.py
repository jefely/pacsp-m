"""F6b: free-generation control, and F4: head-to-head against the zero-parameter baseline.

F6a refuted the word-frequency explanation: machine_poem contains no emotion word at all
yet has the second highest probe mass, while techdoc contains 314 and has almost none.
So the probe mass is not counting literal emotion words. What it does track is
unresolved, and the phrase sensitivity of a factor of 826 makes that a problem.

F6b asks the model to name the emotion without any vocabulary supplied, so the answer is
the model's own choice rather than a restricted readout of my word list. If machine text
yields more explicit emotion naming under free generation, the effect is the model's
judgement and not an artefact of the vocabulary; if not, the restricted readout is doing
the work.

F4 compares node_volume against mean_pair_dist and the other order-free statistics on
identical items, which is the comparison that decides whether the extra dependencies --
language model, vocabulary, threshold, probe phrase -- buy anything.
"""

import json
import os
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pacsp_core import compute_embeddings, load_samples  # noqa: E402

DATA = ROOT / "data"
STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
LM = "Qwen/Qwen2.5-7B-Instruct"
EMB = "BAAI/bge-large-zh-v1.5"
LIMIT = 31
CACHE = ROOT / "records_centroid" / "probe_cache"

PAIRS = [("poem", "machine_poem"), ("lyrics", "machine_lyrics"),
         ("hc3_human_medicine", "hc3_ai_medicine"),
         ("hc3_human_openqa", "hc3_ai_openqa")]

FREE_PROMPT = ("{text}\n\n请用一个词概括这段文字的核心情绪。只输出这个词，"
               "不要解释。")

from emotion_tree_probe import (  # noqa: E402
    EMOTIONS, emotion_token_ids, load_model,
)


def free_generate(texts, tok, mdl, torch, max_new=12):
    """Let the model choose its own emotion word; return the raw answers."""
    answers = []
    for t in texts:
        prompt = FREE_PROMPT.format(text=t.strip().replace("\n", " ")[:400])
        msgs = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        enc = {k: v.to("cuda") for k, v in enc.items()}
        with torch.no_grad():
            out = mdl.generate(**enc, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0][enc["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip()
        answers.append(ans)
    return answers


def sq(A, B):
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def baseline_features(E):
    n = len(E)
    d2 = sq(E, E)
    iu = np.triu_indices(n, k=1)
    off = np.sqrt(d2[iu])
    return {
        "mean_pair_dist": float(off.mean()),
        "dist_cv": float(off.std() / off.mean()),
    }


def main():
    os.environ.setdefault("HF_HOME", str(ROOT / "_hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "_hf_home" / "hub"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    tok, mdl, torch = load_model(LM)
    ids = emotion_token_ids(tok, EMOTIONS)
    in_vocab = set(ids.keys())

    # ------------------------------------------------------------------ F6b
    print("=== F6b: free generation, no vocabulary supplied ===")
    free = {}
    for arm in [a for p in PAIRS for a in p]:
        p = DATA / arm
        if not p.is_dir():
            continue
        texts, _ = load_samples(p)
        texts = texts[:LIMIT]
        t0 = time.time()
        ans = free_generate(texts, tok, mdl, torch)
        n_hit = sum(1 for a in ans if any(w in a for w in in_vocab))
        n_refuse = sum(1 for a in ans if len(a) > 12 or not a)
        free[arm] = {
            "answers_sample": ans[:8],
            "hit_rate": round(n_hit / len(ans), 4),
            "long_or_empty": n_refuse,
            "mean_len": round(float(np.mean([len(a) for a in ans])), 2),
        }
        print(f"  {arm:<22} in-vocabulary {n_hit}/{len(ans)}  "
              f"({n_hit/len(ans):.0%})  mean len {free[arm]['mean_len']:.1f}  "
              f"[{time.time()-t0:.0f}s]")
        print(f"      e.g. {ans[:5]}")

    print()
    print("  paired hit-rate, human over machine:")
    f6b_ratios = {}
    for h, m in PAIRS:
        if h in free and m in free:
            rr = (free[h]["hit_rate"] / free[m]["hit_rate"]
                  if free[m]["hit_rate"] else float("nan"))
            f6b_ratios[h.split("_")[-1]] = round(rr, 4)
            print(f"    {h.split('_')[-1]:<10} human {free[h]['hit_rate']:.3f}  "
                  f"machine {free[m]['hit_rate']:.3f}  ratio {rr:.3f}")

    # ------------------------------------------------------------------ F4
    print("\n=== F4: node_volume versus the zero-parameter baseline ===")
    print(f"  {'domain':<10} {'mean_pair_dist':>16} {'node_volume':>13} "
          f"{'better spread':>15}")
    rows = {}
    for h, m in PAIRS:
        d = h.split("_")[-1]
        Eh = compute_embeddings(load_samples(DATA / h)[0][:LIMIT], model_name=EMB)
        Em = compute_embeddings(load_samples(DATA / m)[0][:LIMIT], model_name=EMB)
        bh = baseline_features(Eh)
        bm = baseline_features(Em)
        mpd = bh["mean_pair_dist"] / bm["mean_pair_dist"]
        fh = CACHE / f"{h}__p0.npy"
        fm = CACHE / f"{m}__p0.npy"
        nv = None
        if fh.exists() and fm.exists():
            nv = float(np.load(fh).sum() / np.load(fm).sum())
        rows[d] = {"mean_pair_dist_ratio": round(mpd, 4),
                   "node_volume_ratio": round(nv, 4) if nv else None}
        print(f"  {d:<10} {mpd:>16.4f} {nv if nv else float('nan'):>13.4f}")

    print()
    print("  spread of each measure across the four domains:")
    for key in ("mean_pair_dist_ratio", "node_volume_ratio"):
        v = np.array([rows[d][key] for d in rows if rows[d][key] is not None])
        if len(v):
            print(f"    {key:<24} min {v.min():.4f}  max {v.max():.4f}  "
                  f"spread {v.max()/v.min():.2f}x  "
                  f"all below 1: {bool(np.all(v < 1))}")

    out = {"free_generation": free, "f6b_ratios": f6b_ratios,
           "f4_comparison": rows}
    f = ROOT / "records_centroid" / f"f4f6b_{STAMP}.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
