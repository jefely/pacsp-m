"""A bootstrap launcher: check the environment, fetch what is missing, start the interface.

The design decision is that the 1.2 GB model is downloaded on first run rather than shipped.
Sharing a 55 MB tool that fetches its model once is far more practical than sharing 1.3 GB,
and the model is immutable and publicly hosted.

What it does, in order, with a single message per failure rather than a traceback:

    interpreter   verify the running Python is one this was built for
    packages      verify numpy, onnxruntime and transformers are importable
    model         look for the ONNX graph; download it if absent, with progress
    tokenizer     look for the tokenizer files; download them if absent
    serve         start the web interface and open a browser

Each step reports what it found and what it did, so a failure on someone else's machine says
which step failed. The download source is configurable by environment variable, and a local
file path works too, for an offline install.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_STEM = "bge-large-zh-v1.5"
ONNX_PATH = ROOT / "onnx" / f"{MODEL_STEM}.onnx"
TOK_DIR = ROOT / "model"
TOK_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.txt",
             "special_tokens_map.json", "config.json",
             "config_sentence_transformers.json", "sentence_bert_config.json",
             "modules.json")

# Where the pieces come from. Overridable so a deployment can point at a mirror, and a
# filesystem path is accepted as well as a URL.
GRAPH_URL = os.environ.get(
    "PACSP_GRAPH_URL",
    "https://huggingface.co/SergeySavinov/bge-large-zh-v1.5-onnx/resolve/main/"
    "model.onnx")
TOK_URL = os.environ.get(
    "PACSP_TOKENIZER_URL",
    "https://huggingface.co/BAAI/bge-large-zh-v1.5/resolve/main/{name}")

REQUIRED_PACKAGES = (("numpy", "numpy"), ("onnxruntime", "onnxruntime"),
                     ("transformers", "transformers"))


def human(n: int) -> str:
    return f"{n/2**20:.1f} MB" if n >= 2**20 else f"{n/1024:.0f} KB"


def step(n: int, total: int, title: str):
    print(f"\n[{n}/{total}] {title}")


def check_interpreter() -> bool:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 9)
    print(f"  python {v.major}.{v.minor}.{v.micro} at {sys.executable}")
    if not ok:
        print("  FAIL: python 3.9 or newer is required")
    return ok


def check_packages() -> list[str]:
    missing = []
    for import_name, dist_name in REQUIRED_PACKAGES:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "?")
            print(f"  {import_name:<16} {ver}")
        except Exception as e:
            print(f"  {import_name:<16} MISSING ({type(e).__name__})")
            missing.append(dist_name)
    if missing:
        print("\n  install with:")
        print(f"    {Path(sys.executable).name} -m pip install {' '.join(missing)}")
    return missing


def fetch(url: str, dest: Path, label: str) -> bool:
    """Download with a progress line. A local path is copied instead."""
    src = Path(url)
    if src.exists() and src.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print(f"  copied {label} from {src} ({human(dest.stat().st_size)})")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pacsp-bootstrap"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got, t0, last = 0, time.time(), 0.0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                now = time.time()
                if now - last > 1.0:
                    last = now
                    rate = got / max(now - t0, 1e-6) / 2**20
                    if total:
                        pct = 100 * got / total
                        print(f"\r  {label}: {pct:5.1f}%  {human(got)} / "
                              f"{human(total)}  {rate:.1f} MB/s", end="", flush=True)
                    else:
                        print(f"\r  {label}: {human(got)}  {rate:.1f} MB/s",
                              end="", flush=True)
        print()
        tmp.replace(dest)
        print(f"  wrote {dest.name}  {human(dest.stat().st_size)}")
        return True
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        print(f"\n  FAILED: {type(e).__name__}: {str(e)[:160]}")
        if tmp.exists():
            tmp.unlink()
        print("  If this machine has no direct access, either set a mirror:")
        print("    set PACSP_GRAPH_URL=<url or local path>")
        print("  or copy the file in by hand to:")
        print(f"    {ONNX_PATH}")
        return False


def ensure_graph() -> bool:
    if ONNX_PATH.exists():
        mb = ONNX_PATH.stat().st_size / 2**20
        print(f"  present: {ONNX_PATH.name}  {mb:.1f} MB")
        if mb < 100:
            print(f"  WARNING: only {mb:.1f} MB, which is too small for this model.")
            print("  It may be a partial download; delete it and rerun to refetch.")
        return True
    print(f"  not found: {ONNX_PATH}")
    print(f"  downloading about 1.2 GB, once. This is the only large transfer.")
    return fetch(GRAPH_URL, ONNX_PATH, "model graph")


def ensure_tokenizer() -> bool:
    TOK_DIR.mkdir(parents=True, exist_ok=True)
    missing = [n for n in TOK_FILES if not (TOK_DIR / n).exists()]
    have = [n for n in TOK_FILES if (TOK_DIR / n).exists()]
    print(f"  present: {len(have)}/{len(TOK_FILES)} files")
    if not missing:
        return True
    print(f"  fetching: {', '.join(missing)}")
    ok = True
    for name in missing:
        if not fetch(TOK_URL.format(name=name), TOK_DIR / name, name):
            ok = False
    return ok


def serve(port: int, host: str, open_browser: bool) -> int:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "vendor"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", str(ROOT / "model"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    import pacsp_serve
    return pacsp_serve.main(["--port", str(port), "--host", host]
                            + (["--open"] if open_browser else []))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="pacsp",
        description="Bootstrap and run the pacsp measurement tool.")
    ap.add_argument("--port", type=int, default=8731)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-open", action="store_true",
                    help="do not open a browser")
    ap.add_argument("--check", action="store_true",
                    help="verify the environment and exit; fetch nothing")
    ap.add_argument("--fetch-only", action="store_true",
                    help="fetch the model and exit without serving")
    args = ap.parse_args(argv)

    print("=" * 66)
    print("  pacsp — 认知沉积的可测框架 / a measurement tool for text collections")
    print("=" * 66)

    TOTAL = 4
    step(1, TOTAL, "interpreter")
    if not check_interpreter():
        return 2

    step(2, TOTAL, "packages")
    missing = check_packages()
    if missing:
        return 2

    step(3, TOTAL, "model")
    if args.check:
        print(f"  graph    : {'present' if ONNX_PATH.exists() else 'MISSING'}")
        print(f"  tokenizer: {TOK_DIR} "
              f"({'present' if all((TOK_DIR/n).exists() for n in TOK_FILES) else 'incomplete'})")
        print("\n  check only; nothing fetched.")
        return 0

    if not ensure_graph():
        return 3
    if not ensure_tokenizer():
        return 3

    if args.fetch_only:
        print("\n  fetch complete; not serving.")
        return 0

    step(4, TOTAL, "serving")
    return serve(args.port, args.host, not args.no_open)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n  interrupted")
        sys.exit(130)
