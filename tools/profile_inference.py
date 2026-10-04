"""Profile where the time actually goes, and whether the GPU is being used.

The batch size of 32 over 31 short texts means one batch, so there is little to keep the GPU
busy; the suspicion is that tokenisation and Python overhead dominate. This measures the two
components separately rather than guessing, and reports the provider actually in use.
"""

import os
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

M = Path(r"D:\myproject\PACSP-M")
sys_path = [str(M / "_ortgpu"), str(M / "_pylibs")]
import sys
for p in sys_path:
    sys.path.insert(0, p)

import importlib.util as iu
_s = iu.find_spec("torch")
if _s and _s.origin:
    os.add_dll_directory(str(Path(_s.origin).parent / "lib"))

os.environ["HF_HOME"] = str(M / "_hf_home")
os.environ["HF_HUB_OFFLINE"] = "1"

import onnxruntime as ort  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def main():
    print(f"  ort {ort.__version__}  providers {ort.get_available_providers()}")
    tok = AutoTokenizer.from_pretrained("BAAI/bge-large-zh-v1.5")

    base = "这是一段用于测试编码速度的中文文本，内容需要足够长以便产生有意义的序列长度。"
    for length_mult in (1, 6, 20):
        texts = [base * length_mult] * 31
        for graph in ("bge-large-zh-v1.5.int8_perchannel.onnx",
                      "bge-large-zh-v1.5.onnx"):
            gp = M / "onnx" / graph
            if not gp.exists():
                continue
            so = ort.SessionOptions()
            so.log_severity_level = 3
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(str(gp), so,
                                        providers=["CUDAExecutionProvider",
                                                   "CPUExecutionProvider"])
            names = {i.name for i in sess.get_inputs()}

            t = time.time()
            enc = tok(texts, padding=True, truncation=True, max_length=512,
                      return_tensors="np")
            tok_t = time.time() - t
            feed = {"input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64)}
            if "token_type_ids" in names:
                feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
            sess.run(None, feed)                       # warm up
            ts = []
            for _ in range(3):
                t = time.time()
                sess.run(None, feed)
                ts.append(time.time() - t)
            print(f"  len~{enc['input_ids'].shape[1]:>3}  {graph[:28]:<30} "
                  f"tokenize {tok_t:.4f}s  onnx {min(ts):.4f}s  "
                  f"provider {sess.get_providers()[0]}")
    print("\n  reading: if onnx time is far below tokenize time, the GPU is idle because")
    print("  the model is small relative to per-call overhead, not because it is unused.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
