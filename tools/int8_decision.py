"""Decide int8 on evidence: does it reproduce the paper, and is it actually faster?

Two things were assumed and both need checking.

First, the earlier profile showed the quantised graph is about twenty times slower than fp32
on the GPU, at every sequence length tested, which is what happens when quantised operators
have no CUDA kernel and fall back to CPU. So int8 is a CPU-only option and the timing claim
must be made per backend.

Second, and decisive, the paper's five ratios have not yet been computed on a quantised
graph at all. This does that, on CPU, for each of the three quantisation variants, and
compares against the published values with the same 0.002 criterion used for fp32.

A variant is adopted only if it both reproduces the paper and is smaller. Reported timings
are on the CPU provider, which is the deployment where int8 could matter.
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
sys.path.insert(0, str(M))
sys.path.insert(0, str(M / "_ortgpu"))
sys.path.insert(0, str(M / "_pylibs"))

import importlib.util as iu  # noqa: E402
_s = iu.find_spec("torch")
if _s and _s.origin:
    os.add_dll_directory(str(Path(_s.origin).parent / "lib"))

os.environ["HF_HOME"] = str(M / "_hf_home")
os.environ["HF_HUB_CACHE"] = str(M / "_hf_home" / "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import onnxruntime as ort  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MODEL_ID = "BAAI/bge-large-zh-v1.5"
TOL = 0.002
PAPER = {
    "poem": ("poem", "machine_poem", 0.8305),
    "lyrics": ("lyrics", "machine_lyrics", 0.8691),
    "techdoc": ("techdoc", "machine_techdoc2", 0.9205),
    "medicine": ("hc3_human_medicine", "hc3_ai_medicine", 0.9147),
    "openqa": ("hc3_human_openqa", "hc3_ai_openqa", 0.9358),
}
GRAPHS = [
    ("fp32", "bge-large-zh-v1.5.onnx"),
    ("int8_perchannel", "bge-large-zh-v1.5.int8_perchannel.onnx"),
    ("int8_plain", "bge-large-zh-v1.5.int8_plain.onnx"),
    ("int8_reduced", "bge-large-zh-v1.5.int8_reduced.onnx"),
]


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
    iu_ = np.triu_indices(n, k=1)
    return float(sqd(E, E)[iu_].mean())


class Enc:
    def __init__(self, path, provider):
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), so, providers=[provider])
        self.names = {i.name for i in self.sess.get_inputs()}
        self.provider = self.sess.get_providers()[0]

    def encode(self, texts, batch=32):
        out, self.seconds = [], 0.0
        for i in range(0, len(texts), batch):
            enc = self.tok(texts[i:i + batch], padding=True, truncation=True,
                           max_length=512, return_tensors="np")
            feed = {"input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64)}
            if "token_type_ids" in self.names:
                feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
            t = time.time()
            h = self.sess.run(None, feed)[0]
            self.seconds += time.time() - t
            v = h[:, 0, :]
            v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
            out.append(v)
        return np.concatenate(out, axis=0).astype(np.float64)


def main():
    t0 = time.time()
    provider = "CPUExecutionProvider"
    print(f"  provider {provider}  (int8 has no CUDA kernel, see docstring)")
    print(f"  {'variant':<18} {'size':>9} {'poem':>9} {'lyrics':>9} {'techdoc':>9} "
          f"{'medicine':>9} {'openqa':>9} {'worst':>8} {'sec':>6} {'verdict':>8}")

    ratios_cache = {}
    rows = []
    for label, fname in GRAPHS:
        p = M / "onnx" / fname
        if not p.exists():
            print(f"  {label:<18} missing", flush=True)
            continue
        e = Enc(p, provider)
        got, errs, secs = {}, [], 0.0
        for key, (a, b, want) in PAPER.items():
            ta, tb = load(a), load(b)
            if not ta or not tb:
                continue
            da = mpd(e.encode(ta)); secs += e.seconds
            db = mpd(e.encode(tb)); secs += e.seconds
            r = da / db
            got[key] = round(r, 6)
            errs.append(abs(r - want))
            # print per domain so progress is visible and nothing is lost to an
            # interrupted run: the int8 graphs take minutes on CPU
            print(f"    {label:<18} {key:<10} ratio {r:.6f} "
                  f"paper {want:.4f} delta {r-want:+.4f}  ({secs:.1f}s so far)",
                  flush=True)
        worst = max(errs) if errs else None
        ok = worst is not None and worst <= TOL
        rows.append({"variant": label, "mb": round(p.stat().st_size / 2**20, 1),
                     "ratios": got, "worst": round(worst, 6), "seconds": round(secs, 2),
                     "accepted": ok})
        print(f"  {label:<18} {p.stat().st_size/2**20:>7.1f}MB "
              + " ".join(f"{got[k]:>9.4f}" for k in PAPER)
              + f" {worst:>8.4f} {secs:>6.2f} {'ACCEPT' if ok else 'REJECT':>8}",
              flush=True)
        del e
        # persist after each variant so an interruption keeps what has been decided
        (M / "onnx" / "quantisation.json").write_text(json.dumps({
            "provider_tested": provider, "rows": rows, "tolerance": TOL,
            "accepted": [r["variant"] for r in rows if r["accepted"]],
            "partial": label != GRAPHS[-1][0],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  published        " + " " * 9
          + " ".join(f"{PAPER[k][2]:>9.4f}" for k in PAPER))

    acc = [r for r in rows if r["accepted"]]
    print(f"\n=== decision ===")
    if not acc:
        print("  no variant reproduced the paper; fp32 stays and int8 is not shipped.")
    else:
        small = min(acc, key=lambda r: r["mb"])
        fast = min(acc, key=lambda r: r["seconds"])
        print(f"  accepted      : {[r['variant'] for r in acc]}")
        print(f"  smallest      : {small['variant']} at {small['mb']:.1f} MB")
        print(f"  fastest (CPU) : {fast['variant']} at {fast['seconds']:.2f}s")
        fp32 = next((r for r in rows if r["variant"] == "fp32"), None)
        if fp32:
            print(f"  fp32 for ref  : {fp32['mb']:.1f} MB, {fp32['seconds']:.2f}s")
            print(f"\n  bundle with {small['variant']:<14} "
                  f"{15.1 + small['mb'] + 0.5:.0f} MB (CPU onnxruntime)")
            print(f"  bundle with fp32           {15.1 + fp32['mb'] + 0.5:.0f} MB")
            print(f"  bundle with CUDA runtime   "
                  f"{746.5 + fp32['mb'] + 0.5:.0f} MB (fp32, GPU)")

    (M / "onnx" / "quantisation.json").write_text(json.dumps({
        "provider_tested": provider,
        "note": "int8 operators have no CUDA kernel and fall back to CPU, measured about "
                "20x slower than fp32 on CUDAExecutionProvider",
        "rows": rows, "tolerance": TOL,
        "accepted": [r["variant"] for r in acc],
        "seconds": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written onnx/quantisation.json  ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
