"""Build a self-contained bundle: interpreter, packages, tool, launchers.

Layout:

    dist/pacsp/
      python/            interpreter plus the stdlib modules actually needed
      vendor/            numpy, onnxruntime, transformers and their dependencies
      model/             tokenizer and config; the ONNX graph lands here on first run
      pacsp_*.py         the tool
      tools/             the exporter, so the graph can be built locally instead
      pacsp.bat / pacsp.sh   launchers

The stdlib copy is filtered rather than wholesale. A full Lib directory on this machine is
5.4 GB, most of it test suites, tkinter, idlelib and __pycache__ that nothing imports. The
filter is a list of names to skip, which is a heuristic and is stated as one: the bundle is
verified by running the tool from it, not by reasoning about what was left out.

The ONNX graph is deliberately NOT included. It is 1.2 GB and immutable, and the launcher
fetches it once, so the artifact that gets shared is roughly 55 MB.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

M = Path(r"D:\myproject\PACSP-M")
OUT = M / "dist" / "pacsp"

# The stdlib is copied with a blocklist, not an allowlist.
#
# The first attempt used an allowlist of top-level names and omitted re.py, which broke
# everything downstream of json. That failure is instructive: the stdlib has hundreds of
# top-level entries, hand-enumerating them is unreliable, and the cost of copying a module
# that is not needed is a few tens of kilobytes.
#
# The earlier belief that the stdlib was 5.4 GB was a measurement error: that figure was
# site-packages. The real Lib directory is about 12 MB, of which the entries below are the
# only substantial dead weight.
SKIP_DIRS = {
    "test", "tests", "idlelib", "tkinter", "turtledemo", "lib2to3",
    "__pycache__", "site-packages", "ensurepip", "pydoc_data",
    "config-3.10", "distutils",
}
SKIP_SUFFIX = (".pyc", ".pdb", ".lib", ".exp", ".chm")


def dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def copy_stdlib(src_lib: Path, dst_lib: Path) -> tuple[int, int]:
    """Copy the stdlib, skipping only the entries in SKIP_DIRS.

    A blocklist rather than an allowlist, for the reason given above the constant.
    """
    dst_lib.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    for item in sorted(src_lib.iterdir()):
        if item.name in SKIP_DIRS:
            skipped += 1
            continue
        if item.suffix in SKIP_SUFFIX:
            skipped += 1
            continue
        dest = dst_lib / item.name
        try:
            if item.is_dir():
                shutil.copytree(item, dest,
                                ignore=shutil.ignore_patterns(
                                    "__pycache__", "*.pyc", "*.pdb"),
                                dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
            kept += 1
        except Exception as e:
            print(f"    skip {item.name}: {type(e).__name__}")
            skipped += 1
    return kept, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="build the pacsp bundle")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--skip-pip", action="store_true",
                    help="reuse an existing vendor/ directory")
    args = ap.parse_args(argv)
    out = Path(args.out)

    t0 = time.time()
    print(f"  target: {out}")
    if out.exists():
        print("  removing previous build")
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- interpreter
    print("\n=== interpreter ===")
    base = Path(sys.base_prefix)
    (out / "python").mkdir(parents=True, exist_ok=True)
    for name in ("python.exe", "pythonw.exe", "python3.dll", "python310.dll",
                 "vcruntime140.dll", "vcruntime140_1.dll"):
        src = base / name
        if src.exists():
            shutil.copy2(src, out / "python" / name)
            print(f"  {name}")
    print("  copying stdlib (filtered)")
    kept, skipped = copy_stdlib(base / "Lib", out / "python" / "Lib")
    print(f"  kept {kept} top-level names, skipped {skipped}")
    if (base / "DLLs").is_dir():
        shutil.copytree(base / "DLLs", out / "python" / "DLLs",
                        ignore=shutil.ignore_patterns("*.pdb"),
                        dirs_exist_ok=True)
        print(f"  DLLs/  {dir_size(out/'python'/'DLLs')/2**20:.1f} MB")

    # ---------------------------------------------------------------- packages
    print("\n=== packages ===")
    vendor = out / "vendor"
    vendor.mkdir(exist_ok=True)
    if not args.skip_pip:
        req = ["numpy", "onnxruntime", "transformers", "tokenizers",
               "huggingface_hub", "safetensors", "regex", "packaging",
               "pyyaml", "filelock", "tqdm", "typing_extensions", "requests",
               "certifi", "charset_normalizer", "idna", "urllib3", "fsspec",
               # pacsp_attest needs Ed25519 for L2. Omitting it produced a bundle where
               # --check passed and attest would have failed at the signing step.
               "cryptography", "cffi", "pycparser"]
        cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir",
               "--only-binary=:all:", "--disable-pip-version-check",
               "--no-warn-script-location", "--target", str(vendor)] + req
        print(f"  pip install {len(req)} packages into vendor/")
        r = subprocess.run(cmd, capture_output=True, text=True)
        tail = (r.stdout or "").strip().splitlines()[-3:]
        for line in tail:
            print(f"    {line}")
        if r.returncode != 0:
            print(f"  pip failed ({r.returncode})")
            print((r.stderr or "")[-800:])
            print("\n  The CUDA build of onnxruntime is not required; the CPU one is")
            print("  enough and much smaller.")
            return 1
    print(f"  vendor/ {dir_size(vendor)/2**20:.1f} MB")

    # ---------------------------------------------------------------- tool
    print("\n=== tool files ===")
    for name in ("pacsp_tool.py", "pacsp_core.py", "pacsp_backends.py",
                 "pacsp_serve.py", "pacsp_bootstrap.py", "README.md",
                 "TOOL.md", "PACSP-M-1.2.0.md", "LICENSE"):
        src = M / name
        if src.exists():
            shutil.copy2(src, out / name)
            print(f"  {name}")
    (out / "tools").mkdir(exist_ok=True)
    for name in ("export_onnx.py",):
        src = M / "tools" / name
        if src.exists():
            shutil.copy2(src, out / "tools" / name)
            print(f"  tools/{name}")

    # tokenizer files travel with the bundle; the graph does not
    print("\n=== tokenizer ===")
    tok_dst = out / "model"
    tok_dst.mkdir(exist_ok=True)
    hub = M / "_hf_home" / "hub" / "models--BAAI--bge-large-zh-v1.5"

    def best_match(pattern: str) -> Path | None:
        """The largest matching file, and never an empty one.

        The Hugging Face cache contains zero-byte placeholders under .no_exist/, for files
        the loader probed and did not find. A plain rglob can return one of those, and an
        empty config.json produced a bundle that failed at load time with "not a valid JSON
        file". Largest-wins also prefers the real snapshot over any partial copy.
        """
        cands = [p for p in hub.rglob(pattern)
                 if p.is_file() and p.stat().st_size > 0]
        return max(cands, key=lambda p: p.stat().st_size) if cands else None

    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.txt",
                 "special_tokens_map.json"):
        s = hub / "snapshots"
        found = best_match(name)
        if found:
            shutil.copy2(found, tok_dst / name)
            print(f"  {name:<38} {found.stat().st_size:>8} B")
        else:
            print(f"  {name:<38} not found in cache")

    for name in ("config.json", "config_sentence_transformers.json",
                 "sentence_bert_config.json", "modules.json",
                 "1_Pooling/config.json"):
        found = best_match(name)
        if not found:
            print(f"  {name:<38} not found in cache")
            continue
        dest = tok_dst / Path(name).name
        # 1_Pooling/config.json would otherwise overwrite the model config, so only the
        # top-level config.json is written under that name
        if name == "1_Pooling/config.json":
            dest = tok_dst / "pooling_config.json"
        shutil.copy2(found, dest)
        print(f"  {Path(name).name:<38} {found.stat().st_size:>8} B")

    # reject a bundle whose support files are not parseable, before it is shipped
    import json as _json
    bad = []
    for f in sorted(tok_dst.glob("*.json")):
        try:
            _json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            bad.append(f"{f.name}: {type(e).__name__}")
    if bad:
        print(f"\n  ABORT: the following support files are not valid JSON: {bad}")
        print("  A bundle with an unparseable config fails at load time, so this is")
        print("  treated as a build failure rather than a warning.")
        return 1
    print(f"  all {len(list(tok_dst.glob('*.json')))} JSON support files parse")

    # ---------------------------------------------------------------- launchers
    print("\n=== launchers ===")
    (out / "pacsp.bat").write_text(
        "@echo off\r\n"
        "setlocal\r\n"
        "set HERE=%~dp0\r\n"
        "set PYTHONPATH=%HERE%vendor\r\n"
        "set PYTHONHOME=%HERE%python\r\n"
        "\"%HERE%python\\python.exe\" \"%HERE%pacsp_bootstrap.py\" %*\r\n"
        "if errorlevel 1 pause\r\n", encoding="ascii", newline="")
    (out / "pacsp.sh").write_text(
        "#!/bin/sh\n"
        "HERE=$(cd \"$(dirname \"$0\")\" && pwd)\n"
        "export PYTHONPATH=\"$HERE/vendor\"\n"
        "export PYTHONHOME=\"$HERE/python\"\n"
        "exec \"$HERE/python/bin/python3\" \"$HERE/pacsp_bootstrap.py\" \"$@\"\n",
        encoding="utf-8")
    os_chmod = getattr(Path(out / "pacsp.sh"), "chmod", None)
    if os_chmod:
        try:
            (out / "pacsp.sh").chmod(0o755)
        except Exception:
            pass
    print("  pacsp.bat, pacsp.sh")

    # ---------------------------------------------------------------- report
    print("\n=== bundle ===")
    for part in ("python", "vendor", "model", "tools"):
        p = out / part
        if p.exists():
            print(f"  {part:<10} {dir_size(p)/2**20:>9.1f} MB")
    print(f"  {'TOTAL':<10} {dir_size(out)/2**20:>9.1f} MB   "
          f"(plus ~1.2 GB fetched on first run)")
    print(f"\n  built in {time.time()-t0:.1f}s")
    print(f"  run: {out / 'pacsp.bat'}")
    print("\n  NEXT: the bundle is unverified. Run tools/verify_bundle.py against it;")
    print("  a bundle that has not reproduced the paper is not a deliverable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
