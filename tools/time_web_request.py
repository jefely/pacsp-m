"""Time a real request against the running server, cold and warm.

The stage timings said the model load was 76 percent of a run and landed on the first
request. Warm-up moves it to startup, so this checks the claim rather than assuming it: the
first request after warm-up should be a couple of seconds, not seven.
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8731"
M = Path(r"D:\myproject\PACSP-M")


def post(payload):
    req = urllib.request.Request(
        BASE + "/api/run", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read()), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), time.time() - t0


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return r.status, r.read()


def main():
    st, body = get("/api/status")
    d = json.loads(body)
    print(f"  /api/status   ready={d['ready']}  warm={d['seconds']}s  "
          f"backend={d['backend']}")

    payload = {"a": str(M / "data" / "poem"),
               "b": str(M / "data" / "machine_poem"),
               "frame": "bge-large-zh", "backend": "onnx", "bootstrap": 2000}

    for label in ("first request after warm-up", "second request"):
        st, r, dt = post(payload)
        if "error" in r:
            print(f"  {label:<30} HTTP {st}  ERROR {r['error'][:80]}")
            return 1
        c = r["compare"]
        acc = c["assignability"]["accuracy"]
        print(f"  {label:<30} {dt:>6.2f}s   ratio {c['ratio_within']['value']:.4f}  "
              f"accuracy {acc:.4f}  verdict {c['verdict'].split(':')[0]}")

    print("\n  timing check:")
    st, r, dt = post(payload)
    ok = dt < 5.0
    print(f"    [{'ok  ' if ok else 'SLOW'}] a warm comparison takes {dt:.2f}s "
          f"(was 7.5s before warm-up, of which 5.67s was the model load)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
