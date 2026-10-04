"""Find a Python installation that can be bundled, and size the staged runtime.

A distributable bundle needs its own interpreter, the packages, and the model. The
interpreter is the piece that has to come from somewhere outside pip, so this reports what is
available and how large each candidate is, rather than assuming the one used for development
is the right one to ship.

It also reports the dependency closure of transformers, since that is what determines the
runtime size and an unstated dependency is a bundle that fails on another machine.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

M = Path(r"D:\myproject\PACSP-M")


def dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main():
    print(f"  current interpreter: {sys.executable}")
    print(f"  version            : {sys.version.split()[0]}")
    base = Path(sys.base_prefix)
    print(f"  base prefix        : {base}")
    print(f"  stdlib size        : {dir_size(base / 'Lib')/2**20:.1f} MB")
    print(f"  include size       : {dir_size(base / 'include')/2**20:.1f} MB")
    print(f"  DLLs size          : {dir_size(base / 'DLLs')/2**20:.1f} MB")

    print("\n=== candidate interpreters on this machine ===")
    cands = []
    for p in (Path(r"C:\Users\Administrator\AppData\Local\Programs\Python"),
              Path(r"C:\Python310"), Path(r"C:\Python311"),
              Path(r"C:\Program Files\Python310")):
        if p.is_dir():
            for exe in p.rglob("python.exe"):
                cands.append(exe)
    exe_here = Path(sys.executable)
    for c in cands:
        is_here = c.resolve() == exe_here.resolve()
        try:
            out = subprocess.run([str(c), "-c", "import sys;print(sys.version.split()[0])"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            out = "?"
        print(f"  {str(c):<62} {out}{'  <- current' if is_here else ''}")
    if not cands:
        print("  none found under the usual locations")

    print("\n=== what transformers actually pulls in ===")
    try:
        from importlib.metadata import requires
        reqs = requires("transformers") or []
        names = []
        for r in reqs:
            n = r.split(";")[0].strip().split()[0].split("[")[0]
            if n and not r.split(";")[1:]:
                names.append(n)
        print(f"  hard requirements ({len(names)}): {', '.join(sorted(set(names)))}")
    except Exception as e:
        print(f"  could not read metadata: {e}")

    print("\n=== installed packages that a bundle would need ===")
    import importlib.util
    total = 0
    for name in ("numpy", "onnxruntime", "transformers", "tokenizers",
                 "huggingface_hub", "safetensors", "regex", "requests", "packaging",
                 "pyyaml", "filelock", "tqdm", "typing_extensions", "certifi",
                 "charset_normalizer", "idna", "urllib3", "fsspec", "yaml"):
        spec = importlib.util.find_spec(name)
        if not spec or not spec.origin:
            print(f"  {name:<24} absent")
            continue
        d = Path(spec.origin).parent
        sz = dir_size(d)
        total += sz
        print(f"  {name:<24} {sz/2**20:>8.1f} MB")
    print(f"  {'TOTAL':<24} {total/2**20:>8.1f} MB")

    print("\n=== the model files a bundle would carry ===")
    sup = 0
    for base_ in (M / "_hf_home" / "hub",):
        d = base_ / "models--BAAI--bge-large-zh-v1.5"
        for f in sorted(d.rglob("*")):
            if f.is_file() and f.suffix in (".json", ".txt") or f.name.endswith(".json"):
                sz = f.stat().st_size
                sup += sz
                if sz > 2000:
                    print(f"  {f.name:<34} {sz/1024:>8.1f} KB")
    print(f"  {'support files total':<34} {sup/2**20:>8.2f} MB")
    graph = M / "onnx" / "bge-large-zh-v1.5.onnx"
    if graph.exists():
        print(f"  {'ONNX graph':<34} {graph.stat().st_size/2**20:>8.1f} MB")

    print("\n=== projected bundle ===")
    print(f"  interpreter + stdlib    ~{dir_size(base/'Lib')/2**20:.0f} MB "
          f"(of which most is unused)")
    print(f"  packages                ~{total/2**20:.0f} MB")
    print(f"  model + graph           "
          f"~{(sup + (graph.stat().st_size if graph.exists() else 0))/2**20:.0f} MB")
    print("\n  note: a bundle needs only the stdlib modules actually imported. "
          "A trimmed")
    print("  stdlib is a separate step and is not attempted here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
