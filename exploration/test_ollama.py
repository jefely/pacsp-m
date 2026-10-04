"""Test whether the local Ollama API can actually generate text right now."""

import json
import socket
import urllib.error
import urllib.request

socket.setdefaulttimeout(120)
BASE = "http://127.0.0.1:11434"


def post(path, payload, timeout=120):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def get(path):
    req = urllib.request.Request(BASE + path, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, r.read()


print("=== /api/tags ===")
try:
    code, raw = get("/api/tags")
    d = json.loads(raw)
    print(f"  HTTP {code}")
    for m in d.get("models", []):
        print(f"    {m.get('name'):<22} {m.get('size', 0)/1e9:.2f} GB  "
              f"family={m.get('details', {}).get('family')}")
except Exception as e:
    print(f"  FAIL {type(e).__name__}: {str(e)[:110]}")

print()
print("=== /api/generate (short prompt, qwen2.5:7b) ===")
try:
    code, raw = post("/api/generate", {
        "model": "qwen2.5:7b",
        "prompt": "用一句话说明什么是高考作文。",
        "stream": False,
        "options": {"num_predict": 60, "temperature": 0.7},
    }, timeout=180)
    d = json.loads(raw)
    print(f"  HTTP {code}")
    print(f"  response: {(d.get('response') or '')[:180]!r}")
    print(f"  done_reason: {d.get('done_reason')}")
except urllib.error.HTTPError as e:
    print(f"  HTTP {e.code}: {e.read()[:200]!r}")
except Exception as e:
    print(f"  FAIL {type(e).__name__}: {str(e)[:130]}")
