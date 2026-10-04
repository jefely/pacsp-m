"""Validate the stored credential and report what it is allowed to do.

Creating a repository needs a different permission from pushing to one, so the scopes are
checked before anything is created. The token is never printed.
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

TOKEN_FILE = Path(sys.argv[1])
API = "https://api.github.com"


def call(path, token, method="GET", body=None):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        API + path, data=data, method=method,
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28",
                 "User-Agent": "pacsp-m-push",
                 **({"Content-Type": "application/json"} if data else {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            scopes = r.headers.get("x-oauth-scopes", "")
            return r.status, json.loads(r.read() or b"{}"), scopes
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:400]
        return e.code, {"error": body}, e.headers.get("x-oauth-scopes", "")
    except Exception as e:
        return 0, {"error": f"{type(e).__name__}: {e}"}, ""


def main():
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    print(f"  token length: {len(token)}, prefix {token[:4]!r}")

    st, me, scopes = call("/user", token)
    print(f"\n  GET /user -> {st}")
    if st == 200:
        print(f"    login : {me.get('login')}")
        print(f"    name  : {me.get('name')}")
        print(f"    repos : {me.get('public_repos')} public")
    else:
        print(f"    {me}")
    print(f"  x-oauth-scopes: {scopes or '(none reported)'}")

    st2, repo, _ = call("/repos/jefely/pacsp-m", token)
    print(f"\n  GET /repos/jefely/pacsp-m -> {st2}")
    if st2 == 200:
        print(f"    already exists: {repo.get('html_url')}")
        print(f"    default branch: {repo.get('default_branch')}")
        print(f"    size: {repo.get('size')} KB, pushed: {repo.get('pushed_at')}")
    elif st2 == 404:
        print("    does not exist yet")
    else:
        print(f"    {repo}")

    st3, repos, _ = call("/user/repos?per_page=100&sort=updated", token)
    if st3 == 200:
        mine = [r["name"] for r in repos if r["owner"]["login"] == me.get("login")]
        print(f"\n  existing repos ({len(mine)}): {', '.join(sorted(mine))}")

    out = {"login": me.get("login"), "scopes": scopes,
           "pacsp_m_exists": st2 == 200}
    Path(sys.argv[2]).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
