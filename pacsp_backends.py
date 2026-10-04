"""Embedding backends for pacsp_tool.

Two backends produce the same vectors by different routes.

    sentence-transformers  the original path; requires torch, about 5.4 GB
    onnx                   onnxruntime plus an exported graph; about 330 MB

The ONNX backend is what makes a small distribution possible. It is not trusted on faith:
tools/onnx_gpu_acceptance.py shows it reproduces the five published D ratios to four
decimals, and the pooling it uses is the one the model's own modules.json specifies, CLS
followed by L2 normalisation.

The CUDA provider is used when it is available. Neither CUDA nor cuDNN is installed
system-wide on the machine this was built on; torch bundles both, so torch's lib directory
is added to the DLL search path before onnxruntime is imported. That makes the GPU path work
without a toolkit install, and the backend reports which provider it actually got rather
than assuming.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

DEFAULT_ONNX = "bge-large-zh-v1.5"


def _add_torch_cuda_dlls() -> str | None:
    """Make torch's bundled CUDA and cuDNN DLLs visible to onnxruntime.

    Only done when torch is present. A failure here is not fatal: the CPU provider still
    works, just more slowly.
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec("torch")
        if not spec or not spec.origin:
            return None
        lib = Path(spec.origin).parent / "lib"
        if not lib.is_dir():
            return None
        os.add_dll_directory(str(lib))
        os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
        return str(lib)
    except Exception:
        return None


class BaseEmbedder:
    name = "base"

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"backend": self.name}


class SentenceTransformerEmbedder(BaseEmbedder):
    name = "sentence-transformers"

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None
        self._cache: dict = {}

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise SystemExit(
                    "sentence-transformers is required for this backend:\n"
                    "  pip install sentence-transformers\n"
                    f"(import failed: {e})")
            self._model = SentenceTransformer(self.model_id)
        return self._model

    def encode(self, texts):
        return np.asarray(self._load().encode(
            texts, batch_size=16, normalize_embeddings=False, show_progress_bar=False),
            dtype=np.float64)


class OnnxEmbedder(BaseEmbedder):
    """CLS pooling plus L2 normalisation, matching the model's modules.json."""

    name = "onnx"

    def __init__(self, onnx_path: Path, tokenizer_id: str, prefer_gpu: bool = True):
        self.onnx_path = Path(onnx_path)
        self.tokenizer_id = tokenizer_id
        if not self.onnx_path.exists():
            raise SystemExit(
                f"ONNX model not found: {self.onnx_path}\n"
                "Export it once with:  python tools/export_onnx.py")
        self.dll_dir = _add_torch_cuda_dlls() if prefer_gpu else None
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tokenizer_id)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        avail = ort.get_available_providers()
        order = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in avail]
        self.sess = ort.InferenceSession(str(self.onnx_path), so, providers=order)
        self.providers = self.sess.get_providers()
        self.input_names = {i.name for i in self.sess.get_inputs()}

    def encode(self, texts, batch: int = 32):
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok(texts[i:i + batch], padding=True, truncation=True,
                           max_length=512, return_tensors="np")
            feed = {"input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64)}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = enc.get(
                    "token_type_ids", np.zeros_like(enc["input_ids"])).astype(np.int64)
            h = self.sess.run(None, feed)[0]
            v = h[:, 0, :]                                  # CLS
            v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
            out.append(v)
        return np.concatenate(out, axis=0).astype(np.float64)

    def describe(self) -> dict:
        return {"backend": self.name, "model": str(self.onnx_path),
                "providers": self.providers, "dll_dir": self.dll_dir}


def make_embedder(backend: str, model_id: str, onnx_path: Path | None = None,
                  onnx_dir: Path | None = None,
                  prefer_gpu: bool = True) -> BaseEmbedder:
    """Choose a backend, falling back only when the preferred one cannot be built."""
    if backend == "onnx":
        p = Path(onnx_path) if onnx_path else (
            (onnx_dir or Path.cwd() / "onnx") / f"{DEFAULT_ONNX}.onnx")
        return OnnxEmbedder(p, model_id, prefer_gpu=prefer_gpu)
    return SentenceTransformerEmbedder(model_id)
