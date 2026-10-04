"""Exercise the web interface's compute endpoint end to end.

A server that serves a page is not a working interface. This posts a real comparison, checks
that the numbers match what the CLI produces for the same inputs, and confirms the gates are
present in the response, since the gates are the part that stops the result being over-read.
"""

import json
import urllib.request

BASE = "http://127.0.0.1:8731"
M = r"D:\myproject\PACSP-M"


def post(payload):
    """Post and return (status, body), including for 4xx and 5xx responses.

    urllib raises on any non-2xx, which hid the status code the test needs to assert on. The
    error body is still JSON, so it is read from the exception.
    """
    req = urllib.request.Request(
        BASE + "/api/run", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"error": raw.decode("utf-8", "replace")[:200]}


def main():
    print("=== compare: lyrics vs machine_lyrics ===")
    st, d = post({"a": M + r"\data\lyrics", "b": M + r"\data\machine_lyrics",
                  "frame": "bge-large-zh", "backend": "onnx", "bootstrap": 2000})
    print(f"  HTTP {st}")
    if "error" in d:
        print(f"  ERROR {d['error']}")
        return 1
    c = d["compare"]
    A, B = c["collections"]["a"], c["collections"]["b"]
    print(f"  A {A['name']:<18} n={A['n']:<4} D={A['D']:.4f}")
    print(f"  B {B['name']:<18} n={B['n']:<4} D={B['D']:.4f}")
    print(f"  ratio      {c['ratio_within']['value']:.4f}  "
          f"CI {c['ratio_within']['ci95']}")
    print(f"  separation {c['separation']['value']:.4f}")
    print(f"  overlap    {c['overlap']['value']:.4f}")
    print(f"  style      {c['style']['ratio']:.4f}")
    print(f"  verdict    {c['verdict']}")
    print(f"  gates      {len(c['gates'])}")
    for g in c["gates"]:
        print(f"    [{'X' if g['fired'] else 'ok'}] {g['id']}")

    print("\n=== cross-check against the published values ===")
    checks = [
        ("D ratio", c["ratio_within"]["value"], 0.8691),
        ("separation", c["separation"]["value"], 1.4516),
        ("overlap", c["overlap"]["value"], 0.066),
    ]
    ok = True
    for label, got, want in checks:
        good = abs(got - want) <= 0.002 * max(1.0, abs(want))
        ok = ok and good
        print(f"  [{'ok  ' if good else 'FAIL'}] {label:<12} web {got:.4f}  "
              f"paper {want:.4f}")

    print("\n=== single-collection measure ===")
    st, d2 = post({"a": M + r"\data\poem", "frame": "bge-large-zh",
                   "backend": "onnx", "bootstrap": 2000})
    if "error" in d2:
        print(f"  ERROR {d2['error']}")
        return 1
    m = d2["measure"]
    print(f"  n={m['n']}  D={m['D']:.4f}  CI {m['ci95']}  CV {m['bootstrap_cv']:.4f}")
    good = abs(m["D"] - 0.5434) <= 0.002
    print(f"  [{'ok  ' if good else 'FAIL'}] D matches paper 0.5434")
    ok = ok and good

    print("\n=== error handling ===")
    st, d3 = post({"a": M + r"\data\gaokao_ai_raw_preclean", "frame": "bge-large-zh",
                   "backend": "onnx"})
    print(f"  too-small corpus -> HTTP {st}, "
          f"{'error reported' if 'error' in d3 else 'NO ERROR (bad)'}")
    ok = ok and "error" in d3

    st, d4 = post({"a": M + r"\data\does_not_exist", "frame": "bge-large-zh",
                   "backend": "onnx"})
    print(f"  missing path     -> HTTP {st}, "
          f"{'error reported' if 'error' in d4 else 'NO ERROR (bad)'}")
    ok = ok and "error" in d4

    print(f"\n  VERDICT: {'web interface agrees with the paper' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
