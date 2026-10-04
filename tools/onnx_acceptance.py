"""Acceptance test, restructured: encode each corpus once and cache it.

The first attempt encoded every corpus six times, once per pooling variant, which is what
made it unusably slow; the cost was mine, not the CPU's. This measures one encoder's
output per corpus and stores it, so the comparison is a table lookup afterwards.

Order of operations, cheapest decisive test first:
    1  a single probe corpus decides the pooling, against SentenceTransformer output
    2  the chosen pooling is applied once per corpus, with the embeddings cached
    3  the paper's five ratios are recomputed and compared

The ONNX build installed here has no CUDA provider, so inference is on CPU. For 31 short
Chinese texts that is seconds; the earlier run was slow because of redundant work.
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

M = Path(r"D:\myproject\PACSP-M")
ID = Path(r"D:\myproject\PACSP-ID")
CACHE = M / "_hf_home"
ONNX = M / "onnx" / "bge-large-zh-v1.5.onnx"
EMB_CACHE = M / "onnx" / "embcache"
sys.path.insert(0, str(M))

os.environ["HF_HOME"] = str(CACHE)
os.environ["HF_HUB_CACHE"] = str(CACHE / "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

MODEL_ID = "BAAI/bge-large-zh-v1.5"
PAPER = {
    "poem": ("poem", "machine_poem", 0.8305),
    "lyrics": ("lyrics", "machine_lyrics", 0.8691),
    "techdoc": ("techdoc", "machine_techdoc2", 0.9205),
    "medicine": ("hc3_human_medicine", "hc3_ai_medicine", 0.9147),
    "openqa": ("hc3_human_openqa", "hc3_ai_openqa", 0.9358),
}


def find_corpus(name):
    for base in (ID / "data" / name, M / "data" / name):
        if base.is_dir() and any(base.glob("*.txt")):
            return base
    return None


def load(name):
    import pacsp_core
    p = find_corpus(name)
    return pacsp_core.load_samples(p)[0] if p else []


def sqd(A, B):
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    return np.sqrt(np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0))


def mpd(E):
    n = len(E)
    iu = np.triu_indices(n, k=1)
    return float(sqd(E, E)[iu].mean())


class OnnxEncoder:
    def __init__(self, path, pooling, normalize, threads=None):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), so,
                                         providers=["CPUExecutionProvider"])
        self.names = {i.name for i in self.sess.get_inputs()}
        self.pooling, self.normalize = pooling, normalize

    def encode(self, texts, batch=16):
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok(texts[i:i + batch], padding=True, truncation=True,
                           max_length=512, return_tensors="np")
            feed = {"input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64)}
            if "token_type_ids" in self.names:
                feed["token_type_ids"] = enc.get(
                    "token_type_ids", np.zeros_like(enc["input_ids"])).astype(np.int64)
            h = self.sess.run(None, feed)[0]
            m = enc["attention_mask"]
            if self.pooling == "cls":
                v = h[:, 0, :]
            elif self.pooling == "mean":
                mf = m[..., None].astype(h.dtype)
                v = (h * mf).sum(axis=1) / np.maximum(mf.sum(axis=1), 1e-9)
            elif self.pooling == "last":
                idx = np.maximum(m.sum(axis=1) - 1, 0)
                v = h[np.arange(h.shape[0]), idx]
            else:
                raise ValueError(self.pooling)
            if self.normalize:
                v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
            out.append(v)
        return np.concatenate(out, axis=0)


def main():
    t0 = time.time()
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"  onnx: {ONNX.name} exists={ONNX.exists()}")

    # ---------------------------------------------------------------- 1. pooling
    probe = load("poem")
    print(f"\n=== 1. choosing pooling on one corpus ({len(probe)} texts) ===")
    ref_path = EMB_CACHE / "st_poem.npy"
    if ref_path.exists():
        ref = np.load(ref_path)
        print(f"  reference embeddings loaded from cache: {ref.shape}")
    else:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(MODEL_ID)
        t = time.time()
        ref = np.asarray(st.encode(probe, batch_size=16, normalize_embeddings=False),
                         dtype=np.float64)
        np.save(ref_path, ref)
        del st
        print(f"  reference computed in {time.time()-t:.1f}s: {ref.shape}")

    best, best_score = None, None
    for pooling in ("cls", "mean", "last"):
        for normalize in (False, True):
            enc = OnnxEncoder(ONNX, pooling, normalize)
            t = time.time()
            got = enc.encode(probe)
            dt = time.time() - t
            score = float(np.abs(got - ref).max())
            label = f"{pooling}{'_norm' if normalize else ''}"
            flag = ""
            if best_score is None or score < best_score:
                best, best_score, flag = (pooling, normalize), score, " <=="
            print(f"  {label:<14} maxdiff {score:>10.6f}  ({dt:>5.1f}s){flag}")
            del enc
    print(f"\n  chosen: pooling={best[0]}, normalize={best[1]}, "
          f"maxdiff={best_score:.6f}")

    # ---------------------------------------------------------------- 2. ratios
    print(f"\n=== 2. D ratios through ONNX vs published ===")
    enc = OnnxEncoder(ONNX, best[0], best[1])
    rows = []
    for key, (a, b, want) in PAPER.items():
        embs = {}
        for label, name in (("a", a), ("b", b)):
            f = EMB_CACHE / f"{best[0]}_{'n' if best[1] else 'r'}_{name}.npy"
            if f.exists():
                embs[label] = np.load(f)
            else:
                ts = load(name)
                if not ts:
                    embs[label] = None
                    continue
                t = time.time()
                E = enc.encode(ts)
                np.save(f, E)
                embs[label] = E
                print(f"    encoded {name:<24} {len(ts):>3} texts in {time.time()-t:>5.1f}s")
        if embs.get("a") is None or embs.get("b") is None:
            print(f"  {key:<10} SKIP (corpus missing)")
            continue
        got = mpd(embs["a"]) / mpd(embs["b"])
        err = got - want
        rows.append((key, want, got, err))
        print(f"  {key:<10} onnx {got:.4f}   paper {want:.4f}   "
              f"delta {err:+.4f}   {'OK' if abs(err) <= 0.002 else 'MISMATCH'}")

    print()
    if rows:
        worst = max(abs(r[3]) for r in rows)
        print(f"  worst |delta|: {worst:.4f}   (criterion 0.002)")
        ok = worst <= 0.002
        print(f"  VERDICT: {'ONNX reproduces the paper' if ok else 'DOES NOT MATCH'}")
    else:
        ok = False
        print("  no ratios computed")

    (M / "onnx" / "acceptance.json").write_text(json.dumps({
        "pooling": best[0], "normalize": best[1],
        "embedding_maxdiff_vs_sentencetransformer": best_score,
        "ratios": [{"domain": r[0], "paper": r[1], "onnx": round(r[2], 6),
                    "delta": round(r[3], 6)} for r in rows],
        "verdict_ok": ok,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  total {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
