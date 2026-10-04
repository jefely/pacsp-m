"""Measure the distributable footprint with the ONNX backend, against the torch path.

The reason for the ONNX work was size: torch alone is 5.4 GB, and a one-click bootstrap is
not viable at that. This reports what each path actually needs, itemised, so the comparison
is on measured bytes rather than on estimates.

Excluded from both figures: the tokenizer and config files, which are a few megabytes and
required by either path, and the corpora, which belong to the user.
"""

import importlib.util
import sys
from pathlib import Path

M = Path(r"D:\myproject\PACSP-M")


def dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


# The tokenizer lives inside the same snapshot directory as the model weights. An earlier
# version of this script summed that whole directory under the label "tokenizer + config",
# which double-counted the 1.2 GB of weights and made the ONNX path look 1.2 GB larger than
# it is. Only the small support files are counted now.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.txt",
                   "special_tokens_map.json", "config.json",
                   "config_sentence_transformers.json", "sentence_bert_config.json",
                   "modules.json")


def support_files_size(snapshot: Path) -> tuple[int, int]:
    total = 0
    n = 0
    for name in TOKENIZER_FILES:
        for f in snapshot.rglob(name):
            if f.is_file():
                total += f.stat().st_size
                n += 1
    return total, n


def pkg_dir(name: str) -> Path | None:
    spec = importlib.util.find_spec(name)
    if not spec or not spec.origin:
        return None
    d = Path(spec.origin).parent
    return d.parent if name == "torch" else d


def main():
    print("=== what the ONNX path needs ===")
    onnx_gpu = M / "_ortgpu" / "onnxruntime"
    onnx_cpu = M / "_pylibs" / "onnxruntime"
    onnx_lib = M / "_pylibs" / "onnx"
    graph = M / "onnx" / "bge-large-zh-v1.5.onnx"
    tok = None
    for base in (M / "_hf_home" / "hub",
                 Path(r"D:\myproject\PACSP-ID\_hf_home\hub")):
        d = base / "models--BAAI--bge-large-zh-v1.5"
        for s in d.rglob("tokenizer.json"):
            tok = s.parent
            break
        if tok:
            break

    sup_bytes, sup_n = support_files_size(tok) if tok else (0, 0)
    print(f"  tokenizer and config files: {sup_n} files, {sup_bytes/2**20:.1f} MB")

    items = [
        ("onnxruntime (CPU build)", onnx_cpu),
        ("onnxruntime-gpu (CUDA build)", onnx_gpu),
        ("onnx (exporter, build-time only)", onnx_lib),
        ("ONNX graph, bge-large-zh", graph if graph.is_file() else None),
    ]
    for label, p in items:
        if p is None or not p.exists():
            print(f"  {label:<36} absent")
            continue
        sz = p.stat().st_size if p.is_file() else dir_size(p)
        print(f"  {label:<36} {sz/2**20:>9.1f} MB")

    if graph.exists():
        onnx_cpu_total = dir_size(onnx_cpu) + graph.stat().st_size + sup_bytes
        print(f"\n  CPU-only distribution                       "
              f"{onnx_cpu_total/2**20:>7.1f} MB")
        if onnx_gpu.exists():
            gpu_total = dir_size(onnx_gpu) + graph.stat().st_size + sup_bytes
            print(f"  CUDA build instead of CPU                   "
                  f"{gpu_total/2**20:>7.1f} MB")

    print("\n=== what the torch path needs ===")
    torch_total = 0
    for name in ("torch", "transformers", "sentence_transformers", "numpy",
                 "scipy", "sklearn", "huggingface_hub", "tokenizers", "safetensors"):
        d = pkg_dir(name)
        if d is None:
            print(f"  {name:<36} absent")
            continue
        sz = dir_size(d)
        torch_total += sz
        print(f"  {name:<36} {sz/2**20:>9.1f} MB")
    if tok:
        torch_total += sup_bytes
    print(f"  {'tokenizer + config':<36} {sup_bytes/2**20:>9.1f} MB")
    print(f"  {'TOTAL':<36} {torch_total/2**20:>9.1f} MB")

    print("\n=== comparison ===")
    if graph.exists() and onnx_gpu.exists():
        gpu_total = dir_size(onnx_gpu) + graph.stat().st_size + sup_bytes
        ratio = torch_total / gpu_total if gpu_total else 0
        print(f"  torch path            {torch_total/2**20:>9.1f} MB")
        print(f"  onnx path (GPU)       {gpu_total/2**20:>9.1f} MB")
        print(f"  reduction             {ratio:>9.1f}x")
        cpu_total = dir_size(onnx_cpu) + graph.stat().st_size + sup_bytes
        print(f"  onnx path (CPU)       {cpu_total/2**20:>9.1f} MB")
        print(f"  reduction             {torch_total/cpu_total:>9.1f}x")
        print(f"\n  The remaining bulk is the fp32 graph itself. int8 quantisation "
              f"would cut it")
        print(f"  to roughly a quarter, at the cost of a parity re-check; that is not "
              f"done here.")
    print("\n  note: the onnx exporter itself is build-time only and need not ship.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
