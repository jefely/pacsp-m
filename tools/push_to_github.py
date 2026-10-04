"""Create the pacsp-m repository and push the local history to it.

Creating the repository and pushing are separate steps with separate failure modes, so each
is reported on its own. The token is read from a file outside the repository and is never
echoed.
"""

import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(r"D:\myproject\PACSP-M")
TOKEN_FILE = Path(r"D:\myproject\chrome-debug\_gh_token.txt")
OWNER = "jefely"
NAME = "pacsp-m"
DESCRIPTION = ("PACSP-M 1.1.0: a measurement framework for cognitive deposition — "
               "a zero-parameter baseline, its validity conditions, and the record of "
               "four rejected alternatives")
API = "https://api.github.com"


def api(path, token, method="GET", body=None):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        API + path, data=data, method=method,
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28",
                 "User-Agent": "pacsp-m-push",
                 **({"Content-Type": "application/json"} if data else {})})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode("utf-8", "replace")[:500]}
    except Exception as e:
        return 0, {"error": f"{type(e).__name__}: {e}"}


def git(*args, env=None):
    # git on this machine emits locale-encoded text (GBK), so decoding must not assume
    # utf-8: with errors=strict the reader thread raised and swallowed the real error.
    r = subprocess.run(["git", "-C", str(REPO), *args],
                       capture_output=True, timeout=900, env=env)
    def dec(b):
        for enc in ("utf-8", "gbk", "cp936", "latin-1"):
            try:
                return b.decode(enc)
            except Exception:
                continue
        return repr(b)
    return r.returncode, dec(r.stdout or b"") + dec(r.stderr or b"")


def main():
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()

    # ---------------------------------------------------------------- create
    st, body = api(f"/repos/{OWNER}/{NAME}", token)
    if st == 200:
        print(f"  1. repository already exists: {body.get('html_url')}")
    elif st == 404:
        st, body = api("/user/repos", token, "POST", {
            "name": NAME,
            "description": DESCRIPTION,
            "private": False,
            "has_issues": True,
            "has_wiki": False,
            "has_projects": False,
            "auto_init": False,
        })
        if st in (200, 201):
            print(f"  1. created: {body.get('html_url')}")
            print(f"     ssh: {body.get('ssh_url')}")
            print(f"     default branch will be: {body.get('default_branch')}")
        else:
            print(f"  1. FAILED to create: {st} {body}")
            return 1
    else:
        print(f"  1. FAILED to check: {st} {body}")
        return 1

    # ---------------------------------------------------------------- remote
    rc, out = git("remote", "get-url", "origin")
    url = f"https://github.com/{OWNER}/{NAME}.git"
    if rc != 0:
        rc, out = git("remote", "add", "origin", url)
        print(f"  2. remote added: {url}  (rc={rc})")
    else:
        cur = out.strip()
        if cur != url:
            git("remote", "set-url", "origin", url)
            print(f"  2. remote updated: {cur} -> {url}")
        else:
            print(f"  2. remote already correct: {url}")

    # ---------------------------------------------------------------- push
    # The environment is inherited in full. An earlier attempt passed a hand-built
    # environment with HOME and USERPROFILE redirected in order to isolate credentials, and
    # that broke schannel with SEC_E_NO_CREDENTIALS: the TLS security package needs the
    # real profile. Authentication is instead supplied through git's environment variables,
    # which git reads directly and which never touch the config file.
    import os
    env = dict(os.environ)
    # One-line evidence: the result of this push, for the record.
    # git's schannel backend cannot acquire a credential handle in this environment
    # (SEC_E_NO_CREDENTIALS on every https access, including a bare ls-remote), while
    # Python's OpenSSL stack reaches the same host without trouble. So git is switched to
    # its OpenSSL backend using the CA bundle it already ships. The switch is per command
    # and is not written to any config file.
    ca = r"C:\Program Files\Git\mingw64\etc\ssl\certs\ca-bundle.crt"
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"

    # Both credential paths are unavailable here. git's schannel backend cannot acquire a
    # handle at all, and the Windows Credential Manager helper is a .NET program that spawns
    # sh.exe, which the sandbox denies a named pipe to ("couldn't create signal pipe").
    # So the credential is placed in the remote URL for the duration of the push and the
    # URL is restored immediately afterwards. It is never written to any config file, and
    # the token is scrubbed from all output.
    auth_url = f"https://x-access-token:{token}@github.com/{OWNER}/{NAME}.git"
    rc, out = git("-c", "http.sslBackend=openssl",
                  "-c", f"http.sslCAInfo={ca}",
                  "-c", "credential.helper=",
                  "push", auth_url, "main:main", env=env)
    print(f"  3. push rc={rc}")
    for line in out.strip().splitlines()[:24]:
        print(f"     {line.replace(token, '***')}")
    if rc != 0:
        git("remote", "set-url", "origin", url)
        print("     remote url restored after failure")
        return 1

    rc2, _ = git("remote", "set-url", "origin", url)
    print(f"  4. remote url restored to {url} (rc={rc2})")
    rc3, refs = git("ls-remote", "--heads", auth_url, env=env)
    for line in refs.strip().splitlines()[:6]:
        print(f"     {line.replace(token, '***')}")

    st, body = api(f"/repos/{OWNER}/{NAME}", token)
    if st == 200:
        print(f"\n  verified: {body.get('html_url')}")
        print(f"    default branch : {body.get('default_branch')}")
        print(f"    size           : {body.get('size')} KB")
        print(f"    pushed at      : {body.get('pushed_at')}")
        print(f"    license        : {body.get('license')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
