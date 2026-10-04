"""Report which ONNX Runtime providers are actually usable here.

The first acceptance run used CPUExecutionProvider and was too slow to finish, which is a
mistake worth not repeating: the machine has an RTX 3080 Ti. onnxruntime (CPU build) does
not ship the CUDA provider, so this reports what the CPU build offers, what the installed
torch reports about CUDA, and whether onnxruntime-gpu is present.
"""

import importlib.util
import sys

print("=== torch / CUDA ===")
try:
    import torch
    print(f"  torch          {torch.__version__}")
    print(f"  cuda available {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device         {torch.cuda.get_device_name(0)}")
        print(f"  cuda version   {torch.version.cuda}")
        free, total = torch.cuda.mem_get_info()
        print(f"  vram           {free/2**30:.1f} free / {total/2**30:.1f} GB total")
except Exception as e:
    print(f"  torch unavailable: {e}")

print("\n=== onnxruntime ===")
try:
    import onnxruntime as ort
    print(f"  version        {ort.__version__}")
    print(f"  providers      {ort.get_available_providers()}")
    cuda = "CUDAExecutionProvider" in ort.get_available_providers()
    print(f"  CUDA provider  {cuda}")
    if not cuda:
        print("  -> this is the CPU build; onnxruntime-gpu is a separate wheel")
except Exception as e:
    print(f"  onnxruntime unavailable: {e}")

print("\n=== what is installed in _pylibs ===")
from pathlib import Path
tgt = Path(r"D:\myproject\PACSP-M\_pylibs")
if tgt.exists():
    for d in sorted(tgt.glob("*.dist-info")):
        print(f"  {d.name}")
    sz = sum(f.stat().st_size for f in tgt.rglob("*") if f.is_file())
    print(f"  total {sz/2**20:.1f} MB")
