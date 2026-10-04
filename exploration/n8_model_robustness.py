"""N8: does the D direction survive a change of embedding model?

The concern is that the findings rest on one encoder, BAAI/bge-large-zh-v1.5 (326M,
1024-dim). Different conclusions could plausibly split along a line:

  structural   order sensitivity, the failure of mu localisation, the sigma problem.
               These follow from the definition of a sum along a sequence or from the
               algebra of an RBF kernel. They should not depend on the encoder.
  empirical    the 5-of-5 direction of D, and the size of the ratios. These are measured
               properties of a particular embedding space and could move.

This tests the second group. Two further encoders of clearly different size and lineage
are used alongside the original:

  BAAI/bge-small-zh-v1.5       24M,  512-dim, same family, much smaller
  shibing624/text2vec-base-chinese   different lineage, 768-dim

For each, the human-over-machine ratio of D is computed on all five pairs and compared
against the original. What matters is not that the numbers agree but whether the direction
is stable, and whether the spread stays narrow.
"""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
M = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(M))
import pacsp_core  # noqa: E402

DATA = M / "data"
CACHE = M / "_hf_home"
STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")

MODELS = {
    "bge-large-zh-v1.5": "BAAI/bge-large-zh-v1.5",
    "bge-small-zh-v1.5": "BAAI/bge-small-zh-v1.5",
    "text2vec-base-chinese": "shibing624/text2vec-base-chinese",
}

PAIRS = [("poem", "machine_poem"), ("lyrics", "machine_lyrics"),
         ("techdoc", "machine_techdoc2"),
         ("hc3_human_medicine", "hc3_ai_medicine"),
         ("hc3_human_openqa", "hc3_ai_openqa")]


def sq(A, B):
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def mpd(E):
    n = len(E)
    iu = np.triu_indices(n, k=1)
    return float(np.sqrt(sq(E, E)[iu]).mean())


def embed(texts, model_name):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    E = m.encode(texts, batch_size=8, normalize_embeddings=False)
    return np.asarray(E, dtype=np.float64)


def main():
    os.environ["HF_HOME"] = str(CACHE)
    os.environ["HF_HUB_CACHE"] = str(CACHE / "hub")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    cache = {}
    for h, m in PAIRS:
        for arm in (h, m):
            if arm not in cache and (DATA / arm).is_dir():
                cache[arm] = pacsp_core.load_samples(DATA / arm)[0]

    out = {"models": {}, "pairs": {}}
    for label, repo in MODELS.items():
        print(f"\n{'=' * 74}\n  {label}  ({repo})\n{'=' * 74}")
        try:
            emb = {}
            for arm, texts in cache.items():
                emb[arm] = embed(texts, repo)
            dim = next(iter(emb.values())).shape[1]
            print(f"  dim = {dim}")
        except Exception as e:
            print(f"  cannot load: {type(e).__name__}: {str(e)[:140]}")
            out["models"][label] = {"error": f"{type(e).__name__}: {str(e)[:140]}"}
            continue

        out["models"][label] = {"dim": int(dim), "repo": repo}
        ratios = {}
        print(f"  {'domain':<12} {'human D':>10} {'machine D':>10} {'ratio':>9}")
        for h, m in PAIRS:
            if h not in emb or m not in emb:
                continue
            dh, dm = mpd(emb[h]), mpd(emb[m])
            r = dh / dm
            dom = h.split("_")[-1]
            ratios[dom] = round(r, 4)
            print(f"  {dom:<12} {dh:>10.4f} {dm:>10.4f} {r:>9.4f}")
        vals = list(ratios.values())
        below = sum(1 for v in vals if v < 1)
        out["models"][label].update({
            "ratios": ratios, "below_1": below, "n": len(vals),
            "spread": round(max(vals) / min(vals), 4) if vals else None})
        print(f"  -> below 1: {below}/{len(vals)}   "
              f"spread {max(vals)/min(vals):.2f}x")

    # ------------------------------------------------------------------ compare
    print(f"\n{'=' * 74}\n  comparison\n{'=' * 74}")
    keys = [k for k in MODELS if "error" not in out["models"].get(k, {"e": 1})]
    if keys:
        doms = sorted({d for k in keys for d in out["models"][k].get("ratios", {})})
        print(f"  {'domain':<12} " + "".join(f"{k[:18]:>20}" for k in keys))
        for d in doms:
            row = []
            for k in keys:
                row.append(out["models"][k].get("ratios", {}).get(d))
            print(f"  {d:<12} " + "".join(
                f"{(v if v is not None else float('nan')):>20.4f}" for v in row))
        print(f"  {'below 1':<12} " + "".join(
            f"{str(out['models'][k]['below_1']) + '/' + str(out['models'][k]['n']):>20}"
            for k in keys))
        print(f"  {'spread':<12} " + "".join(
            f"{out['models'][k]['spread']:>20.2f}" for k in keys))

        # direction agreement across models, per domain
        print()
        agree = 0
        for d in doms:
            signs = [out["models"][k]["ratios"].get(d) for k in keys]
            signs = [s for s in signs if s is not None]
            same = all(s < 1 for s in signs) or all(s > 1 for s in signs)
            agree += same
            print(f"  {d:<12} direction consistent across models: {same}")
        print(f"\n  domains with consistent direction: {agree}/{len(doms)}")
        out["verdict"] = {
            "models_compared": keys,
            "domains": doms,
            "domains_direction_consistent": agree,
            "total_domains": len(doms),
            "all_consistent": agree == len(doms),
            "below_1_by_model": {k: out["models"][k]["below_1"] for k in keys},
            "spread_by_model": {k: out["models"][k]["spread"] for k in keys},
        }

    f = M / "records_centroid" / f"n8_model_robustness_{STAMP}.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
