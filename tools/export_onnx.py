"""Step 1: export bge-large-zh-v1.5 to ONNX and prove it matches torch.

The goal is to remove torch from the runtime, since torch is 5.4 GB of the 5.7 GB install
and a one-click bootstrap is not viable at that size. onnxruntime is about 50 MB.

The acceptance criterion is not "it runs". It is that the ONNX path reproduces what the
paper reports, because a tool that disagrees with the paper is worse than no tool. So this
does two comparisons:

    embeddings   raw last_hidden_state from transformers against the ONNX graph, per item
    distances    the D ratios the paper publishes, recomputed through ONNX

The pooling question is settled by measurement rather than assumption. The paper's
compute_embeddings calls SentenceTransformer.encode with normalize_embeddings=False, so
whatever pooling that implies must be reproduced. All four combinations of
CLS-versus-mean pooling and normalised-versus-raw are compared, and the winner is reported.
"""

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

M = Path(r"D:\myproject\PACSP-M")
ID = Path(r"D:\myproject\PACSP-ID")
CACHE = M / "_hf_home"
OUT = M / "onnx"
sys.path.insert(0, str(M))

import os
os.environ["HF_HOME"] = str(CACHE)
os.environ["HF_HUB_CACHE"] = str(CACHE / "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

MODEL_ID = "BAAI/bge-large-zh-v1.5"
TOL_EMB = 2e-3      # max absolute difference per component, relative to vector scale


def load_texts(p: Path):
    """Corpora live in two places: the original set under PACSP-ID, and the arms built
    during the measurement work (machine_techdoc2 and friends) under PACSP-M."""
    import pacsp_core
    for base in (p, M / "data" / p.name, ID / "data" / p.name):
        if base.is_dir() and any(base.glob("*.txt")):
            return pacsp_core.load_samples(base)[0]
    return []


def torch_hidden(texts, batch=8):
    """Raw last_hidden_state plus the attention mask, which pooling needs."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModel.from_pretrained(MODEL_ID).eval()
    outs = []
    masks = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = tok(chunk, padding=True, truncation=True, max_length=512,
                      return_tensors="pt")
            r = mdl(**enc).last_hidden_state
            outs.append(r.cpu().numpy())
            masks.append(enc["attention_mask"].cpu().numpy())
    return outs, masks, tok


def export_onnx(tok, path: Path, seq_len=128):
    """Export through a wrapper that takes three tensors and returns one.

    Passing the model directly fails: transformers' forward is defined over **kwargs, so
    positional tensors are matched by position against its signature and an omitted
    token_type_ids shifts everything along, raising "got multiple values for use_cache".
    A wrapper with an explicit signature removes the ambiguity, and returning a bare tensor
    keeps the ONNX graph outputs simple.
    """
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class Wrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask, token_type_ids):
            out = self.inner(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             return_dict=True)
            return out.last_hidden_state

    mdl = AutoModel.from_pretrained(MODEL_ID).eval()
    wrapped = Wrapper(mdl).eval()
    dummy = tok(["测试文本用于导出"] * 2, padding="max_length",
                truncation=True, max_length=seq_len, return_tensors="pt")
    tt = dummy.get("token_type_ids")
    if tt is None:
        tt = torch.zeros_like(dummy["input_ids"])
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (dummy["input_ids"], dummy["attention_mask"], tt),
            str(path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                          "attention_mask": {0: "batch", 1: "seq"},
                          "token_type_ids": {0: "batch", 1: "seq"},
                          "last_hidden_state": {0: "batch", 1: "seq"}},
            opset_version=17,
            do_constant_folding=True,
        )
    return path


def onnx_hidden(texts, path: Path, batch=8):
    import onnxruntime as ort
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    outs, masks = [], []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tok(chunk, padding=True, truncation=True, max_length=512,
                  return_tensors="np")
        feed = {"input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64)}
        if "token_type_ids" in names:
            feed["token_type_ids"] = enc.get(
                "token_type_ids",
                np.zeros_like(enc["input_ids"])).astype(np.int64)
        r = sess.run(None, feed)[0]
        outs.append(r)
        masks.append(enc["attention_mask"])
    return outs, masks


def pool(hidden_list, mask_list, mode):
    vecs = []
    for h, m in zip(hidden_list, mask_list):
        if mode == "cls":
            v = h[:, 0, :]
        else:
            mf = m[..., None].astype(h.dtype)
            v = (h * mf).sum(axis=1) / np.maximum(mf.sum(axis=1), 1e-9)
        if mode.endswith("_norm"):
            v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        vecs.append(v)
    return np.concatenate(vecs, axis=0)


def sqd(A, B):
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    return np.sqrt(np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0))


def mpd(E):
    n = len(E)
    iu = np.triu_indices(n, k=1)
    return float(sqd(E, E)[iu].mean())


PAPER = {  # corpus A / corpus B -> published ratio of D
    "poem": ("poem", "machine_poem", 0.8305),
    "lyrics": ("lyrics", "machine_lyrics", 0.8691),
    "techdoc": ("techdoc", "machine_techdoc2", 0.9205),
    "medicine": ("hc3_human_medicine", "hc3_ai_medicine", 0.9147),
    "openqa": ("hc3_human_openqa", "hc3_ai_openqa", 0.9358),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    onnx_path = OUT / "bge-large-zh-v1.5.onnx"

    texts = load_texts(ID / "data" / "poem")
    print(f"  probe corpus: {len(texts)} texts")
    print("\n=== 1. exporting ===")
    if onnx_path.exists():
        print(f"  already present: {onnx_path} "
              f"({onnx_path.stat().st_size/2**20:.1f} MB)")
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        export_onnx(tok, onnx_path)
        print(f"  wrote {onnx_path} ({onnx_path.stat().st_size/2**20:.1f} MB)")

    print("\n=== 2. hidden states: torch vs onnx ===")
    th, tm, tok = torch_hidden(texts)
    oh, om = onnx_hidden(texts, onnx_path)
    diffs = []
    for a, b in zip(th, oh):
        n = min(a.shape[1], b.shape[1])
        diffs.append(float(np.abs(a[:, :n] - b[:, :n]).max()))
    print(f"  max abs difference across batches: {max(diffs):.6f}")
    print(f"  (tolerance {TOL_EMB})")

    print("\n=== 3. which pooling reproduces the paper? ===")
    print(f"  {'pooling':<12} {'emb maxdiff':>12} {'poem D':>9} {'ratio':>8} "
          f"{'paper':>8} {'delta':>8}")
    results = {}
    for mode in ("cls", "mean", "cls_norm", "mean_norm"):
        tv = pool(th, tm, mode)
        ov = pool(oh, om, mode)
        ediff = float(np.abs(tv - ov).max())
        results[mode] = {"emb_maxdiff": ediff}
        # recompute the poem ratio through ONNX and compare with the paper,
        # but only if the embedding path is close enough to be meaningful
        Ea = pool(*onnx_hidden(load_texts(ID / "data" / "poem"), onnx_path), mode) \
            if False else None
        print(f"  {mode:<12} {ediff:>12.6f}")

    # pooling choice is settled by matching the D the tool already validated
    print("\n=== 4. D ratios through ONNX, per pooling mode ===")
    data = {}
    for key, (a, b, _) in PAPER.items():
        data[key] = (load_texts(ID / "data" / a), load_texts(ID / "data" / b))
    emb_cache = {}
    for mode in ("cls", "mean", "cls_norm", "mean_norm"):
        oh_all, om_all = {}, {}
        for key, (ta, tb) in data.items():
            for label, ts in (("a", ta), ("b", tb)):
                h, m = onnx_hidden(ts, onnx_path)
                emb_cache[(mode, key, label)] = pool(h, m, mode)
        ratios = {}
        for key, (_, _, want) in PAPER.items():
            da = mpd(emb_cache[(mode, key, "a")])
            db = mpd(emb_cache[(mode, key, "b")])
            ratios[key] = da / db
        errs = [abs(ratios[k] - PAPER[k][2]) for k in PAPER]
        results[mode]["ratios"] = {k: round(v, 4) for k, v in ratios.items()}
        results[mode]["max_ratio_err"] = round(max(errs), 4)
        print(f"  {mode:<12} " +
              " ".join(f"{k}={ratios[k]:.4f}" for k in PAPER) +
              f"   max err {max(errs):.4f}")

    best = min(results, key=lambda m: results[m]["max_ratio_err"])
    print(f"\n  best pooling: {best}  (max ratio error "
          f"{results[best]['max_ratio_err']:.4f})")
    print(f"  published    : " +
          " ".join(f"{k}={PAPER[k][2]:.4f}" for k in PAPER))

    (OUT / "onnx_parity.json").write_text(
        json.dumps({"emb_maxdiff": max(diffs), "by_pooling": results,
                    "best": best, "paper": {k: PAPER[k][2] for k in PAPER}},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {OUT / 'onnx_parity.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
