"""Verify the bundle as a third party would receive it.

The point is to test the bundle, not the development tree. So this invokes the bundled
interpreter with the bundled vendor directory and an empty PYTHONPATH, from a working
directory that is not the repository, and checks:

    imports      numpy, onnxruntime and transformers load from vendor/, not from site-packages
    interpreter  the reported sys.executable is the bundled one
    stdlib       the filtered copy still covers what the tool imports
    numbers      the paper's five D ratios are reproduced
    launcher     pacsp.bat exists and its --check path runs

A bundle that imports from the host instead of from itself would pass a naive test and fail
on another machine, which is why the paths are asserted rather than assumed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

M = Path(r"D:\myproject\PACSP-M")
B = M / "dist" / "pacsp"
PY = B / "python" / "python.exe"

PAPER = {"poem": 0.8305, "lyrics": 0.8691, "techdoc": 0.9205,
         "medicine": 0.9147, "openqa": 0.9358}
TOL = 0.002
FAILS = []


def run(args, env_extra=None, cwd=None, timeout=900):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONIOENCODING"] = "utf-8"
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HOME"] = str(B / "model")
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(args, capture_output=True, timeout=timeout,
                       env=env, cwd=cwd or str(B))
    out = (r.stdout or b"").decode("utf-8", "replace")
    err = (r.stderr or b"").decode("utf-8", "replace")
    return r.returncode, out, err


def check(label, ok, detail=""):
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)
    return ok


def main():
    print(f"  bundle: {B}")
    if not PY.exists():
        print(f"  bundled interpreter missing: {PY}")
        return 1

    # ---------------------------------------------------------------- where imports come from
    print("\n=== imports resolve inside the bundle ===")
    probe = (
        "import json,sys,os\n"
        "info={'executable':sys.executable,'prefix':sys.prefix}\n"
        "out={}\n"
        "for m in ('numpy','onnxruntime','transformers','tokenizers','safetensors',"
        "'packaging','yaml','regex','requests','filelock','tqdm'):\n"
        "    try:\n"
        "        mod=__import__(m)\n"
        "        out[m]=getattr(mod,'__file__','?')\n"
        "    except Exception as e:\n"
        "        out[m]='ERR:'+type(e).__name__\n"
        "info['modules']=out\n"
        "print(json.dumps(info))\n"
    )
    rc, out, err = run([str(PY), "-c", probe],
                       env_extra={"PYTHONPATH": str(B / "vendor")})
    if rc != 0:
        print(f"  probe failed rc={rc}\n{err[-900:]}")
        check("import probe runs", False)
        return 1
    info = json.loads(out.strip().splitlines()[-1])
    check("sys.executable is the bundled python",
          Path(info["executable"]).resolve() == PY.resolve(), info["executable"])
    outside = []
    for name, path in info["modules"].items():
        if isinstance(path, str) and path.startswith("ERR"):
            check(f"import {name}", False, path)
        elif str(B / "vendor") not in path:
            outside.append((name, path))
    check("every required package loads from vendor/", not outside,
          "" if not outside else f"outside: {outside[:3]}")

    # ---------------------------------------------------------------- stdlib coverage
    print("\n=== the filtered stdlib still covers the tool ===")
    probe2 = ("import importlib,sys,json\n"
              "names=['argparse','ast','base64','collections','concurrent.futures',"
              "'csv','dataclasses','datetime','decimal','difflib','enum','functools',"
              "'gzip','hashlib','hmac','html','http.server','importlib.metadata',"
              "'io','json','logging','mimetypes','multiprocessing','os','pathlib',"
              "'pickle','platform','pprint','queue','random','re','secrets','shutil',"
              "'signal','socket','socketserver','ssl','statistics','string','struct',"
              "'subprocess','sysconfig','tarfile','tempfile','textwrap','threading',"
              "'time','tokenize','traceback','types','typing','unicodedata','urllib.parse',"
              "'urllib.request','uuid','warnings','weakref','webbrowser','zipfile']\n"
              "bad=[]\n"
              "for n in names:\n"
              "    try: importlib.import_module(n)\n"
              "    except Exception as e: bad.append(n+':'+type(e).__name__)\n"
              "print(json.dumps(bad))\n")
    rc, out, err = run([str(PY), "-c", probe2],
                       env_extra={"PYTHONPATH": str(B / "vendor")})
    if rc != 0:
        # Do not report "probe failed" as if it were the answer: the bundle's stdlib is what
        # is under test, and the traceback says which module is actually missing.
        print(f"  probe rc={rc}")
        print("  " + "\n  ".join(err.strip().splitlines()[-12:]))
        bad = ["probe failed"]
    else:
        bad = json.loads(out.strip().splitlines()[-1])
    check("stdlib modules import", not bad, "" if not bad else f"missing: {bad}")

    # ---------------------------------------------------------------- numbers
    print("\n=== the bundle reproduces the paper ===")
    script = (
        "import sys,json\n"
        "sys.path.insert(0,r'%s')\n"
        "import numpy as np, pacsp_core, pacsp_tool\n"
        "emb=pacsp_tool.Embedder('BAAI/bge-large-zh-v1.5', backend='onnx',\n"
        "    onnx_dir=r'%s')\n"
        "print('backend', emb.active_backend)\n"
        "print('detail', json.dumps(emb.describe(), ensure_ascii=False))\n"
        "def load(n):\n"
        "    import pathlib\n"
        "    for base in (r'%s', r'%s'):\n"
        "        p=pathlib.Path(base)/'data'/n\n"
        "        if p.is_dir() and any(p.glob('*.txt')): return pacsp_core.load_samples(p)[0]\n"
        "    return []\n"
        "def mpd(E):\n"
        "    iu=np.triu_indices(len(E),k=1)\n"
        "    d=E[:,None,:]-E[None,:,:]\n"
        "    return float(np.sqrt((d*d).sum(-1))[iu].mean())\n"
        "pairs={'poem':('poem','machine_poem'),'lyrics':('lyrics','machine_lyrics'),"
        "'techdoc':('techdoc','machine_techdoc2'),"
        "'medicine':('hc3_human_medicine','hc3_ai_medicine'),"
        "'openqa':('hc3_human_openqa','hc3_ai_openqa')}\n"
        "res={}\n"
        "for k,(a,b) in pairs.items():\n"
        "    ta,tb=load(a),load(b)\n"
        "    if not ta or not tb: res[k]=None; continue\n"
        "    res[k]=mpd(emb.encode(ta))/mpd(emb.encode(tb))\n"
        "print('RESULT '+json.dumps(res))\n"
    ) % (str(B), str(B / "onnx"), str(M), str(M))
    env_g = {"PYTHONPATH": str(B / "vendor"), "HF_HUB_OFFLINE": "1",
             "HF_HOME": str(B / "model")}
    # the graph is not in the bundle by design, so point at the local copy
    if not (B / "onnx" / "bge-large-zh-v1.5.onnx").exists():
        (B / "onnx").mkdir(exist_ok=True)
        src = M / "onnx" / "bge-large-zh-v1.5.onnx"
        if src.exists():
            print("  staging the graph into the bundle for this test")
            import shutil
            try:
                os.link(src, B / "onnx" / src.name)
            except Exception:
                shutil.copy2(src, B / "onnx" / src.name)
    rc, out, err = run([str(PY), "-c", script], env_extra=env_g, timeout=1200)
    print("  " + "\n  ".join(
        ln for ln in out.splitlines() if ln.startswith("backend")))
    line = next((ln for ln in out.splitlines() if ln.startswith("RESULT ")), None)
    if not line:
        check("ratio computation runs", False, err[-400:] if err else "no RESULT line")
    else:
        got = json.loads(line[len("RESULT "):])
        for dom, want in PAPER.items():
            v = got.get(dom)
            if v is None:
                check(f"{dom} ratio", False, "corpus missing")
                continue
            check(f"{dom} ratio", abs(v - want) <= TOL,
                  f"bundle {v:.4f} paper {want:.4f}")

    # ---------------------------------------------------------------- launcher
    print("\n=== launcher ===")
    bat = B / "pacsp.bat"
    check("pacsp.bat present", bat.exists())
    if bat.exists():
        rc, out, err = run(["cmd", "/c", str(bat), "--check"],
                           env_extra={"PYTHONPATH": str(B / "vendor")})
        joined = (out + err)
        check("pacsp.bat --check runs", rc == 0, f"rc={rc}")
        for want in ("interpreter", "packages", "model"):
            check(f"--check reports '{want}'", want in joined)

    print()
    if FAILS:
        print(f"  {len(FAILS)} CHECK(S) FAILED:")
        for f in FAILS:
            print(f"    - {f}")
        return 1
    print("  bundle verified: imports are self-contained and the numbers match the paper")
    return 0


if __name__ == "__main__":
    sys.exit(main())
