"""ONNX acceptance test on GPU.

The GPU build needs CUDA 12 and cuDNN 9 DLLs. Neither is installed system-wide, but torch
bundles both, so torch's lib directory is added to the DLL search path before onnxruntime is
imported. That is also what makes this test self-contained: no CUDA toolkit install is
required for the frame to run on the GPU.

Provider availability is reported before anything is timed, because a silent fallback to CPU
is exactly the failure mode that wasted a run earlier.
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
ORTGPU = M / "_ortgpu"
CACHE = M / "_hf_home"
ONNX = M / "onnx" / "bge-large-zh-v1.5.onnx"
EMB = M / "onnx" / "embcache"
sys.path.insert(0, str(M))
sys.path.insert(0, str(ORTGPU))

os.environ["HF_HOME"] = str(CACHE)
os.environ["HF_HUB_CACHE"] = str(CACHE / "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# torch ships cudart64_12, cublas64_12 and cudnn64_9; make them findable.
import importlib.util as _iu
_spec = _iu.find_spec("torch")
if _spec and _spec.origin:
    _torchlib = Path(_spec.origin).parent / "lib"
    if _torchlib.is_dir():
        os.add_dll_directory(str(_torchlib))
        os.environ["PATH"] = str(_torchlib) + os.pathsep + os.environ.get("PATH", "")
        print(f"  DLL path added: {_torchlib}")

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


class Enc:
    def __init__(self, path, pooling, normalize):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        avail = ort.get_available_providers()
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        provs = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in avail]
        self.sess = ort.InferenceSession(str(path), so, providers=provs)
        self.active = self.sess.get_providers()
        self.names = {i.name for i in self.sess.get_inputs()}
        self.pooling, self.normalize = pooling, normalize

    def encode(self, texts, batch=32):
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
    EMB.mkdir(parents=True, exist_ok=True)
    print(f"  onnx model: exists={ONNX.exists()}")

    import onnxruntime as ort
    print(f"  ort version: {ort.__version__}")
    print(f"  available  : {ort.get_available_providers()}")

    probe = load("poem")
    ref_f = EMB / "st_poem.npy"
    if ref_f.exists():
        ref = np.load(ref_f)
        print(f"  reference embeddings from cache: {ref.shape}")
    else:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(MODEL_ID)
        t = time.time()
        ref = np.asarray(st.encode(probe, batch_size=32, normalize_embeddings=False),
                         dtype=np.float64)
        np.save(ref_f, ref)
        print(f"  reference in {time.time()-t:.1f}s")

    print(f"\n=== pooling choice on one corpus ({len(probe)} texts) ===")
    results = {}
    best, best_score = None, None
    for pooling in ("cls", "mean", "last"):
        for normalize in (False, True):
            e = Enc(ONNX, pooling, normalize)
            t = time.time()
            got = e.encode(probe)
            dt = time.time() - t
            score = float(np.abs(got - ref).max())
            label = f"{pooling}{'_norm' if normalize else ''}"
            results[label] = {"maxdiff": score, "seconds": round(dt, 2)}
            flag = ""
            if best_score is None or score < best_score:
                best, best_score, flag = (pooling, normalize), score, " <=="
            print(f"  {label:<12} maxdiff {score:>10.6f}  {dt:>5.2f}s  "
                  f"providers={e.active[:1]}{flag}")
            del e
    print(f"\n  chosen: pooling={best[0]} normalize={best[1]} "
          f"maxdiff={best_score:.6f}")

    print(f"\n=== D ratios: ONNX (GPU) vs published ===")
    e = Enc(ONNX, best[0], best[1])
    tag = f"gpu_{best[0]}{'n' if best[1] else 'r'}"
    rows = []
    for key, (a, b, want) in PAPER.items():
        E = {}
        for lab, name in (("a", a), ("b", b)):
            f = EMB / f"{tag}_{name}.npy"
            if f.exists():
                E[lab] = np.load(f)
            else:
                ts = load(name)
                if not ts:
                    E[lab] = None
                    continue
                t = time.time()
                E[lab] = e.encode(ts)
                np.save(f, E[lab])
                print(f"    {name:<24} {len(ts):>3} texts  {time.time()-t:>5.2f}s")
        if E.get("a") is None or E.get("b") is None:
            print(f"  {key:<10} SKIP")
            continue
        got = mpd(E["a"]) / mpd(E["b"])
        d = got - want
        rows.append((key, want, got, d))
        print(f"  {key:<10} onnx {got:.4f}  paper {want:.4f}  "
              f"delta {d:+.4f}  {'OK' if abs(d) <= 0.002 else 'MISMATCH'}")

    worst = max(abs(r[3]) for r in rows) if rows else None
    ok = worst is not None and worst <= 0.002
    print(f"\n  worst |delta| {worst:.4f}  criterion 0.002")
    print(f"  VERDICT: {'ONNX-GPU reproduces the paper' if ok else 'DOES NOT MATCH'}")
    (M / "onnx" / "acceptance_gpu.json").write_text(json.dumps({
        "provider": Enc.avail if hasattr(Enc, "avail") else "see output",
        "pooling": best[0], "normalize": best[1],
        "maxdiff": best_score, "by_label": results,
        "ratios": [{"domain": r[0], "paper": r[1], "onnx": round(r[2], 6),
                    "delta": round(r[3], 6)} for r in rows],
        "verdict_ok": ok, "seconds": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  total {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
