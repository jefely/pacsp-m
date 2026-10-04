"""Check what is available for the ONNX path, without blocking on a download."""

import importlib.util
import sys
from pathlib import Path

print("=== import availability ===")
for m in ("torch", "onnx", "onnxruntime", "transformers", "sentence_transformers",
          "numpy", "optimum", "onnxscript"):
    spec = importlib.util.find_spec(m)
    if not spec:
        print(f"  {m:<22} missing")
        continue
    try:
        mod = __import__(m)
        ver = getattr(mod, "__version__", "?")
        print(f"  {m:<22} {ver}")
    except Exception as e:
        print(f"  {m:<22} found but import failed: {type(e).__name__}: {e}")

print("\n=== torch.onnx export surface ===")
try:
    import torch
    print(f"  torch {torch.__version__}")
    print(f"  has torch.onnx.export        : {hasattr(torch.onnx, 'export')}")
    print(f"  has torch.onnx.dynamo_export : {hasattr(torch.onnx, 'dynamo_export')}")
    print(f"  cuda available               : {torch.cuda.is_available()}")
except Exception as e:
    print(f"  torch unavailable: {e}")

print("\n=== the model we need to export ===")
for base in (Path(r"D:\myproject\PACSP-M\_hf_home\hub"),
             Path(r"D:\myproject\PACSP-ID\_hf_home\hub")):
    if not base.exists():
        continue
    for d in sorted(base.iterdir()):
        if not d.name.startswith("models--BAAI--bge-large-zh"):
            continue
        snaps = list((d / "snapshots").glob("*")) if (d / "snapshots").exists() else []
        print(f"  {d.name}  ({base.parent.parent.name})")
        for s in snaps:
            files = sorted(p.name for p in s.iterdir() if p.is_file())
            print(f"    snapshot {s.name[:12]}: {len(files)} files")
            for f in files:
                sz = (s / f).stat().st_size
                print(f"      {f:<34} {sz/2**20:>8.1f} MB")
        # is there already an onnx in the cache?
        onnx = list(d.rglob("*.onnx"))
        print(f"    existing .onnx files: {len(onnx)}")
        for o in onnx[:5]:
            print(f"      {o}")
