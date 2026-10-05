"""Set a GitHub repository's visibility, reading the stored credential in-process.

WHY NOT `gh repo edit`
----------------------
The GitHub CLI is not installed here, and `read_gh_credential.py` hands the token to the
caller by WRITING IT TO A FILE. For a visibility change there is no need for that round
trip: the credential can be read straight out of Windows Credential Manager inside this
process, used, and dropped. Fewer copies of a secret is the whole point.

The token is never printed and never written anywhere. `main` checks its own captured
report for the token string and refuses to print if it ever appears.

Usage:
    python tools/set_repo_visibility.py <owner/repo> <public|private> [--dry-run]
    python tools/set_repo_visibility.py --list <owner>
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as w
import json
import sys
import urllib.error
import urllib.request

API = "https://api.github.com"
TARGET = "git:https://github.com"
CRED_TYPE_GENERIC = 1


class CREDENTIAL(ctypes.Structure):
    _fields_ = [
        ("Flags", w.DWORD), ("Type", w.DWORD),
        ("TargetName", w.LPWSTR), ("Comment", w.LPWSTR),
        ("LastWritten", w.FILETIME), ("CredentialBlobSize", w.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_byte)),
        ("Persist", w.DWORD), ("AttributeCount", w.DWORD),
        ("Attributes", ctypes.c_void_p), ("TargetAlias", w.LPWSTR),
        ("UserName", w.LPWSTR),
    ]


def read_token() -> str:
    c32 = ctypes.windll.advapi32
    c32.CredReadW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD,
                              ctypes.POINTER(ctypes.POINTER(CREDENTIAL))]
    c32.CredReadW.restype = w.BOOL
    c32.CredFree.argtypes = [ctypes.c_void_p]
    p = ctypes.POINTER(CREDENTIAL)()
    if not c32.CredReadW(TARGET, CRED_TYPE_GENERIC, 0, ctypes.byref(p)):
        raise RuntimeError(f"no stored credential under {TARGET!r}")
    cred = p.contents
    blob = ctypes.string_at(cred.CredentialBlob, cred.CredentialBlobSize)
    c32.CredFree(p)
    return blob.decode("utf-16-le", errors="ignore").rstrip("\x00")


def call(path: str, token: str, method: str = "GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        API + path, data=data, method=method,
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28",
                 "User-Agent": "pacsp-visibility",
                 **({"Content-Type": "application/json"} if data else {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode("utf-8", "replace")[:400]}


def describe(st: int, repo: dict, label: str) -> None:
    if st != 200:
        print(f"  [X] {label}: HTTP {st}  {repo.get('error', '')[:160]}")
        return
    vis = "private" if repo.get("private") else "public"
    print(f"  [OK] {label}")
    print(f"        可见性   : {vis}")
    print(f"        URL      : {repo.get('html_url')}")
    print(f"        默认分支 : {repo.get('default_branch')}")
    print(f"        最后推送 : {repo.get('pushed_at')}")


def main() -> int:
    raw = sys.argv[1:]
    dry = "--dry-run" in raw
    want_list = "--list" in raw
    # NB: filter `--flags` out AFTER reading them; an earlier version filtered first and then
    # tested args[0] == "--list", which could never be true.
    args = [a for a in raw if not a.startswith("--")]

    token = read_token()
    report: list[str] = []

    def say(s: str) -> None:
        report.append(s)

    if want_list:
        owner = args[1] if len(args) > 1 else "jefely"
        # /users/{owner}/repos returns PUBLIC repositories only, so a repo that was just
        # made private disappears from it — which is exactly the wrong moment to lose sight
        # of it. Use the authenticated endpoint when listing our own account.
        st_me, me = call("/user", token)
        mine = st_me == 200 and me.get("login", "").lower() == owner.lower()
        path = ("/user/repos?per_page=100&sort=updated&affiliation=owner" if mine
                else f"/users/{owner}/repos?per_page=100&sort=updated")
        st, data = call(path, token)
        if st != 200:
            print(f"  [X] list failed: HTTP {st}  {str(data)[:200]}")
            return 1
        print(f"  {owner} 的仓库（{len(data)} 个"
              f"{'，含私有' if mine else '，仅公开'}）")
        for r in sorted(data, key=lambda x: x["name"]):
            say(f"    {r['name']:<24} {'private' if r['private'] else 'public':<8} "
                f"{r.get('pushed_at', '')[:10]}")
    elif len(args) >= 2:
        slug, want = args[0], args[1].lower()
        if want not in ("public", "private"):
            print("  visibility must be 'public' or 'private'", file=sys.stderr)
            return 2
        st, before = call(f"/repos/{slug}", token)
        print("  变更前")
        describe(st, before, slug)
        if st != 200:
            print(f"\n  {' '.join(report)}")
            return 1
        already = bool(before.get("private")) == (want == "private")
        if already:
            print(f"\n  已经是 {want}，无需变更。")
            return 0
        if dry:
            print(f"\n  [dry-run] 会把 {slug} 改成 {want}，未执行。")
            return 0
        st2, after = call(f"/repos/{slug}", token, "PATCH", {"private": want == "private"})
        print(f"\n  变更后（PATCH -> HTTP {st2}）")
        describe(st2, after, slug)
        if st2 != 200:
            return 1
        # verify with an independent GET
        st3, check = call(f"/repos/{slug}", token)
        ok = st3 == 200 and bool(check.get("private")) == (want == "private")
        print(f"\n  独立复核 GET -> HTTP {st3}  私有={check.get('private')}  "
              f"{'✅ 生效' if ok else '❌ 未生效'}")
        return 0 if ok else 1
    else:
        print(__doc__)
        return 2

    # never let the token reach stdout
    text = "\n".join(report)
    if token and token in text:
        print("  [X] refusing to print: the token appeared in the report")
        return 3
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
