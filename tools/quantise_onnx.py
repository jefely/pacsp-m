"""int8 quantisation of the ONNX graph, accepted only if the paper's ratios survive.

The fp32 graph is 1,238 MB, which is most of the 1,254 MB CPU bundle. Dynamic quantisation
of the MatMul weights typically cuts that to about a quarter.

The risk is that quantisation changes the distances. D is a mean of Euclidean distances
between embeddings, and quantisation perturbs every embedding slightly, so the question is
empirical: do the five published ratios survive? The criterion is the same 0.002 used for the
fp32 acceptance test, so a quantised graph that drifts fails and is not adopted.

Both weight-only and weight-plus-activation quantisation are tried, because they differ in
how much they perturb the output and only measurement can say which is acceptable here.
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
SRC = M / "onnx" / "bge-large-zh-v1.5.onnx"
EMB = M / "onnx" / "embcache"
sys.path.insert(0, str(M))
sys.path.insert(0, str(M / "_ortgpu"))

os.environ["HF_HOME"] = str(CACHE)
os.environ["HF_HUB_CACHE"] = str(CACHE / "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import importlib.util as _iu
_s = _iu.find_spec("torch")
if _s and _s.origin:
    _lib = Path(_s.origin).parent / "lib"
    if _lib.is_dir():
        os.add_dll_directory(str(_lib))
        os.environ["PATH"] = str(_lib) + os.pathsep + os.environ.get("PATH", "")

MODEL_ID = "BAAI/bge-large-zh-v1.5"
PAPER = {
    "poem": ("poem", "machine_poem", 0.8305),
    "lyrics": ("lyrics", "machine_lyrics", 0.8691),
    "techdoc": ("techdoc", "machine_techdoc2", 0.9205),
    "medicine": ("hc3_human_medicine", "hc3_ai_medicine", 0.9147),
    "openqa": ("hc3_human_openqa", "hc3_ai_openqa", 0.9358),
}
TOL = 0.002


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
    def __init__(self, path):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        provs = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(str(path), so, providers=provs)
        self.names = {i.name for i in self.sess.get_inputs()}

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
            v = h[:, 0, :]
            v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
            out.append(v)
        return np.concatenate(out, axis=0).astype(np.float64)


def quantise(src: Path, dst: Path, per_channel: bool, reduce_range: bool) -> bool:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    try:
        quantize_dynamic(
            model_input=str(src), model_output=str(dst),
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
            reduce_range=reduce_range,
            extra_options={"MatMulConstBOnly": True},
        )
        return dst.exists()
    except Exception as e:
        print(f"    quantisation failed: {type(e).__name__}: {str(e)[:140]}")
        return False


def main():
    t0 = time.time()
    if not SRC.exists():
        print(f"  source graph missing: {SRC}")
        return 1
    print(f"  source: {SRC.name}  {SRC.stat().st_size/2**20:.1f} MB")

    variants = [
        ("int8_perchannel", dict(per_channel=True, reduce_range=False)),
        ("int8_plain", dict(per_channel=False, reduce_range=False)),
        ("int8_reduced", dict(per_channel=False, reduce_range=True)),
    ]

    # ratios for the fp32 graph, taken from the accepted run
    acc = json.loads((M / "onnx" / "acceptance_gpu.json").read_text(encoding="utf-8"))
    fp32 = {r["domain"]: r["onnx"] for r in acc["ratios"]}
    print(f"  fp32 ratios (accepted): {fp32}")

    rows = []
    for name, kw in variants:
        dst = M / "onnx" / f"bge-large-zh-v1.5.{name}.onnx"
        print(f"\n=== {name} ===")
        if not dst.exists():
            ok = quantise(SRC, dst, **kw)
            if not ok:
                continue
        print(f"  size {dst.stat().st_size/2**20:.1f} MB "
              f"({dst.stat().st_size/SRC.stat().st_size*100:.1f}% of fp32)")
        e = Enc(dst)
        ratios, errs = {}, []
        for key, (a, b, want) in PAPER.items():
            ta, tb = load(a), load(b)
            if not ta or not tb:
                continue
            got = mpd(e.encode(ta)) / mpd(e.encode(tb))
            ratios[key] = round(got, 6)
            errs.append(abs(got - want))
        worst = max(errs) if errs else None
        ok = worst is not None and worst <= TOL
        print(f"  ratios {ratios}")
        print(f"  worst |delta| vs paper {worst:.4f}  "
              f"{'ACCEPT' if ok else 'REJECT'}")
        rows.append({"variant": name, "size_mb": round(dst.stat().st_size/2**20, 1),
                     "ratios": ratios, "worst_delta": round(worst, 6) if worst else None,
                     "accepted": ok})
        del e

    print("\n=== summary ===")
    print(f"  {'variant':<18} {'size':>9} {'worst |delta|':>14} {'verdict':>9}")
    for r in rows:
        print(f"  {r['variant']:<18} {r['size_mb']:>7.1f} MB "
              f"{r['worst_delta']:>14.4f} "
              f"{'ACCEPT' if r['accepted'] else 'REJECT':>9}")
    good = [r for r in rows if r["accepted"]]
    best = min(good, key=lambda r: r["size_mb"]) if good else None
    if best:
        print(f"\n  smallest accepted: {best['variant']} at {best['size_mb']:.1f} MB "
              f"(fp32 was {SRC.stat().st_size/2**20:.1f} MB)")
        print(f"  bundle: onnxruntime CPU 15.1 + graph {best['size_mb']:.1f} "
              f"+ support 0.5 = {15.1 + best['size_mb'] + 0.5:.1f} MB")
    else:
        print("\n  no quantised variant reproduced the paper; fp32 stays.")
        print("  This is the honest outcome if it happens: the size win is not worth")
        print("  breaking agreement with the published numbers.")

    (M / "onnx" / "quantisation.json").write_text(json.dumps({
        "fp32_ratios": fp32, "variants": rows,
        "accepted_best": best["variant"] if best else None,
        "seconds": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {M / 'onnx' / 'quantisation.json'}  ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
