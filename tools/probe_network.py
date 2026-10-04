"""Probe PyPI reachability and look for already-downloaded wheels.

A previous pip download produced no output for several minutes and had to be cancelled, so
this establishes whether the network is the problem or pip's resolution is, using short
explicit timeouts. It also checks the pip cache and any existing target directory, because
an offline path only makes sense if the wheels are already on disk.
"""

import json
import socket
import ssl
import sys
import urllib.request
from pathlib import Path

TIMEOUT = 8


def head(url: str) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "pacsp-probe"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return True, f"HTTP {r.status}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:80]}"


def main():
    print("=== DNS ===")
    for host in ("pypi.org", "files.pythonhosted.org", "github.com"):
        try:
            ip = socket.gethostbyname(host)
            print(f"  {host:<26} {ip}")
        except Exception as e:
            print(f"  {host:<26} FAILED {type(e).__name__}")

    print(f"\n=== HTTPS (timeout {TIMEOUT}s) ===")
    for url in ("https://pypi.org/simple/onnxruntime/",
                "https://files.pythonhosted.org/",
                "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/onnxruntime/"):
        ok, msg = head(url)
        print(f"  {'OK  ' if ok else 'FAIL'} {url[:56]:<58} {msg}")

    print("\n=== existing wheels or installs on disk ===")
    cands = [
        Path(r"D:\myproject\PACSP-M\_pylibs"),
        Path(r"D:\myproject\PACSP-ID\_pylibs"),
        Path(r"D:\myproject\PACSP-ID\dist"),
        Path.home() / "AppData" / "Local" / "pip" / "Cache",
        Path(r"C:\Users\Administrator\AppData\Local\pip\Cache"),
    ]
    for c in cands:
        if not c.exists():
            print(f"  {str(c):<58} absent")
            continue
        sz = sum(f.stat().st_size for f in c.rglob("*") if f.is_file())
        whl = list(c.rglob("*.whl"))
        onnx_whl = [w for w in whl if "onnx" in w.name.lower()]
        print(f"  {str(c):<58} {sz/2**20:>8.1f} MB, {len(whl)} wheels, "
              f"{len(onnx_whl)} onnx")

    print("\n=== verdict ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
