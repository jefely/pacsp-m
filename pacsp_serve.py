"""pacsp serve — a local web interface over pacsp_tool.

Why a local server rather than tkinter: tkinter is absent from some Python builds, its
rendering differs across platforms, and a table of statistics with gates reads better as
HTML than as widgets. The browser is already installed, the server binds to 127.0.0.1 only,
and nothing is uploaded anywhere.

The interface does three things the command line cannot do as comfortably: browse the
filesystem to pick corpora, show the gates as a colour-coded list next to the numbers, and
let the backend be chosen per run. The measurement itself is the same code the CLI and the
regression tests use, so the numbers are the same numbers.

Security: bound to loopback, so only this machine can reach it. Directory listing is offered
because the point is to pick a corpus, and the tool is meant to be run by the person whose
files are being listed.
"""

from __future__ import annotations

import html
import json
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

M = Path(__file__).resolve().parent
sys.path.insert(0, str(M))

import pacsp_tool as T  # noqa: E402

STATE: dict = {"embedder": None, "lock": threading.Lock(), "busy": False,
               "warm": False, "warm_seconds": None, "warm_error": None}


def warm_up_async(frame: str | None = None, backend: str = "auto"):
    """Load the model and encode one text before the first request arrives.

    Timing a compare showed the model load is 5.67 s of a 7.5 s run, 76 percent, and it lands
    entirely on the first request. Warming it in a background thread at startup moves that
    cost off the user's first click, which is the difference between the interface feeling
    slow and feeling immediate.
    """
    import pacsp_tool as T

    def work():
        t0 = time.time()
        try:
            with STATE["lock"]:
                if STATE["embedder"] is None:
                    key = frame or T.DEFAULT_FRAME
                    repo = T.KNOWN_FRAMES.get(key, (key,))[0]
                    STATE["embedder"] = T.Embedder(
                        repo, backend=backend,
                        onnx_dir=Path(__file__).resolve().parent / "onnx")
                    STATE["embedder"].encode(["预热"])
            STATE["warm"] = True
        except Exception as e:
            STATE["warm_error"] = f"{type(e).__name__}: {e}"
        finally:
            STATE["warm_seconds"] = round(time.time() - t0, 2)
            print(f"  warm-up {STATE['warm_seconds']}s  ready={STATE['warm']}",
                  flush=True)

    threading.Thread(target=work, daemon=True).start()

PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>pacsp — 语义分散度测量</title>
<style>
:root{--bg:#fbfbfd;--fg:#1a1a1c;--mut:#6b6b73;--line:#e3e3e8;--acc:#2b5fd9;
--ok:#1a7f45;--warn:#b4590a;--bad:#c02a2a}
*{box-sizing:border-box}
body{margin:0;font:14px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,
"Microsoft YaHei",sans-serif;background:var(--bg);color:var(--fg)}
header{padding:18px 26px;border-bottom:1px solid var(--line);background:#fff}
h1{margin:0;font-size:17px;font-weight:650}
h1 span{color:var(--mut);font-weight:400;font-size:13px;margin-left:10px}
main{padding:22px 26px;max-width:1080px}
.row{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-end;margin-bottom:16px}
label{display:block;font-size:12px;color:var(--mut);margin-bottom:4px}
input[type=text]{width:340px;padding:8px 10px;border:1px solid var(--line);
border-radius:7px;font:inherit;background:#fff}
input[type=text]:focus{outline:2px solid #cfe0ff;border-color:var(--acc)}
select{padding:8px 10px;border:1px solid var(--line);border-radius:7px;
font:inherit;background:#fff}
button{padding:9px 16px;border:0;border-radius:7px;background:var(--acc);color:#fff;
font:inherit;font-weight:560;cursor:pointer}
button:disabled{background:#b9c4dc;cursor:default}
button.ghost{background:#fff;color:var(--fg);border:1px solid var(--line)}
table{border-collapse:collapse;width:100%;margin:8px 0 20px;font-size:13px}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:left}
th{background:#f4f5f8;font-weight:600}
td.n{text-align:right;font-variant-numeric:tabular-nums}
.gate{display:flex;gap:9px;padding:7px 11px;border-radius:7px;margin:4px 0;
font-size:13px;border:1px solid var(--line);background:#fff}
.gate.f{border-color:#f0c9c9;background:#fff6f6}
.gate.ok{border-color:#c9e6d4;background:#f5fdf8}
.tag{font-weight:650;font-family:ui-monospace,Consolas,monospace;font-size:12px}
.tag.ok{color:var(--ok)} .tag.f{color:var(--bad)}
#verdict{padding:13px 16px;border-radius:8px;font-weight:600;margin:14px 0;
border:1px solid var(--line);background:#fff}
#verdict.sep{border-color:#c9e6d4;background:#f3fdf7}
#verdict.mid{border-color:#f0dcc0;background:#fffaf1}
#verdict.none{border-color:#f0c9c9;background:#fff7f7}
#msg{color:var(--mut);margin-left:12px}
.browse{margin-top:6px;max-height:210px;overflow:auto;border:1px solid var(--line);
border-radius:7px;background:#fff}
.browse div{padding:6px 11px;cursor:pointer;border-bottom:1px solid #f2f2f5;
font-size:13px;display:flex;justify-content:space-between}
.browse div:hover{background:#f6f8ff}
.browse div span{color:var(--mut);font-size:12px}
.hint{color:var(--mut);font-size:12px;margin-top:6px}
code{font-family:ui-monospace,Consolas,monospace;background:#f2f3f6;padding:1px 5px;
border-radius:4px;font-size:12.5px}
.hidden{display:none}
</style></head><body>
<header><h1>pacsp<span>语义分散度测量 · 本地运行 · 不上传任何文件</span></h1></header>
<main>
  <div class="row">
    <div><label>语料 A（目录）</label>
      <input type="text" id="a" placeholder="D:\\corpus\\human">
      <div class="hint"><a href="#" onclick="browse('a');return false">浏览目录</a></div>
    </div>
    <div><label>语料 B（可选，留空则只测 A）</label>
      <input type="text" id="b" placeholder="D:\\corpus\\machine">
      <div class="hint"><a href="#" onclick="browse('b');return false">浏览目录</a></div>
    </div>
  </div>
  <div class="row">
    <div><label>参照系</label>
      <select id="frame">
        <option value="bge-large-zh">bge-large-zh（已校准，1024 维）</option>
        <option value="bge-small-zh">bge-small-zh（已校准，512 维）</option>
      </select></div>
    <div><label>推理后端</label>
      <select id="backend">
        <option value="auto">自动（有 ONNX 图则用 ONNX）</option>
        <option value="onnx">ONNX（体积小）</option>
        <option value="sentence-transformers">sentence-transformers</option>
      </select></div>
    <div><label>自助法次数</label>
      <select id="boot">
        <option value="2000">2000（默认）</option>
        <option value="500">500（快）</option>
        <option value="8000">8000（精）</option>
      </select></div>
    <div><button id="go" onclick="run()">计算</button>
      <button class="ghost" onclick="download()">导出 JSON</button></div>
  </div>
  <div id="msg"></div>
<div id="status" class="hint">正在加载模型…</div>
  <div id="browse" class="browse hidden"></div>
  <div id="out"></div>
</main>
<script>
let RESULT=null, BROWSE_TARGET=null, READY=false;
function poll(){
  fetch('/api/status').then(r=>r.json()).then(d=>{
    READY=!!d.ready;
    const s=document.getElementById('status');
    if(READY){ s.textContent='模型已就绪'+(d.backend?('（'+d.backend+'）'):'')
      +(d.seconds?('  预热 '+d.seconds+'s'):''); s.style.color='#1a7f45';
      document.getElementById('go').disabled=false; }
    else if(d.error){ s.textContent='模型加载失败：'+d.error; s.style.color='#c02a2a';
      document.getElementById('go').disabled=false; READY=true; }
    else { s.textContent='正在加载模型…（首次约 6 秒；模型本身很大，之后各次测量约 2 秒）';
      s.style.color='#b4590a'; document.getElementById('go').disabled=true;
      setTimeout(poll, 700); }
  }).catch(()=>setTimeout(poll,1000));
}
poll();
function q(id){return document.getElementById(id)}
function browse(target){
  BROWSE_TARGET=target;
  const cur=q(target).value||'';
  fetch('/api/list?path='+encodeURIComponent(cur)).then(r=>r.json()).then(d=>{
    const box=q('browse'); box.classList.remove('hidden'); box.innerHTML='';
    if(d.parent){const e=document.createElement('div');
      e.innerHTML='<b>..</b><span>上级</span>'; e.onclick=()=>{q(target).value=d.parent;browse(target)};
      box.appendChild(e);}
    d.dirs.forEach(x=>{const e=document.createElement('div');
      e.innerHTML='<b>'+x.name+'</b><span>'+(x.txt?x.txt+' 个 .txt':'无 .txt')+'</span>';
      e.onclick=()=>{q(target).value=x.path;browse(target)}; box.appendChild(e)});
  });
}
function run(){
  const a=q('a').value.trim(); if(!a){q('msg').textContent='请填写语料 A';return}
  const body={a:a,b:q('b').value.trim(),frame:q('frame').value,
    backend:q('backend').value,bootstrap:+q('boot').value};
  q('go').disabled=true; q('msg').textContent='计算中…（首次运行需加载模型）';
  q('browse').classList.add('hidden');
  fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)}).then(r=>r.json()).then(d=>{
      q('go').disabled=false; q('msg').textContent='';
      if(d.error){q('out').innerHTML='<div id="verdict" class="none">错误：'+
        d.error+'</div>';return}
      RESULT=d; render(d);
    }).catch(e=>{q('go').disabled=false;q('msg').textContent='';q('out').innerHTML=
      '<div id="verdict" class="none">请求失败：'+e+'</div>'});
}
function render(d){
  let h='';
  if(d.compare){
    const c=d.compare, A=c.collections.a, B=c.collections.b;
    h+='<table><tr><th>项</th><th>'+A.name+'</th><th>'+B.name+'</th></tr>';
    h+='<tr><td>篇数</td><td class="n">'+A.n+'</td><td class="n">'+B.n+'</td></tr>';
    h+='<tr><td>D（均值距离）</td><td class="n">'+A.D.toFixed(4)+
      '</td><td class="n">'+B.D.toFixed(4)+'</td></tr>';
    h+='<tr><td>D 的 95% 区间</td><td class="n">['+A.ci95[0].toFixed(4)+', '+
      A.ci95[1].toFixed(4)+']</td><td class="n">['+B.ci95[0].toFixed(4)+', '+
      B.ci95[1].toFixed(4)+']</td></tr></table>';
    const r=c.ratio_within, iv=c.interval;
    h+='<table><tr><th>集合间关系</th><th>值</th><th>95% 区间</th><th>解读</th></tr>';
    h+='<tr><td>D 比值（A ÷ B）</td><td class="n">'+r.value.toFixed(4)+
      '</td><td class="n">['+r.ci95[0].toFixed(4)+', '+r.ci95[1].toFixed(4)+
      ']</td><td>'+(r.ci95[0]>1||r.ci95[1]<1?'区间不含 1':'<b>区间含 1，方向不可主张</b>')+
      '</td></tr>';
    h+='<tr><td>交叉均值</td><td class="n">'+iv.cross_mean.toFixed(4)+
      '</td><td class="n">['+iv.ci95[0].toFixed(4)+', '+iv.ci95[1].toFixed(4)+
      ']</td><td>共 '+iv.n_cross_distances+' 个距离</td></tr>';
    h+='<tr><td>分离度</td><td class="n">'+c.separation.value.toFixed(4)+
      '</td><td></td><td>交叉 ÷ 组内合并</td></tr>';
    h+='<tr><td><b>重叠占比</b></td><td class="n"><b>'+c.overlap.value.toFixed(3)+
      '</b></td><td></td><td>'+(c.overlap.value>0.5?'两组大量交错':'两组较分离')+
      '（半径 '+c.overlap.radius.toFixed(3)+'）</td></tr>';
    if(c.style&&c.style.ratio!==null){h+='<tr><td>词汇风格比</td><td class="n">'+
      c.style.ratio.toFixed(4)+'</td><td></td><td>跨集合 ÷ 集合内，与 D 无关</td></tr>'}
    h+='</table>';
    const cls=c.verdict.startsWith('separable')?'sep':
      (c.verdict.startsWith('group-mean')?'mid':'none');
    h+='<div id="verdict" class="'+cls+'">判定：'+c.verdict+'</div>';
    h+='<h3 style="font-size:14px;margin:18px 0 6px">闸门（每条结论的证据）</h3>';
    c.gates.forEach(g=>{h+='<div class="gate '+(g.fired?'f':'ok')+'">'+
      '<span class="tag '+(g.fired?'f':'ok')+'>'+(g.fired?'✕':'✓')+' '+g.id+
      '</span><span>'+g.detail+'</span></div>'});
    h+='<h3 style="font-size:14px;margin:18px 0 6px">限制（随结果一并报告）</h3>';
    h+='<ul style="color:#444;font-size:12.5px">'+c.limits.map(x=>'<li>'+
      x+'</li>').join('')+'</ul>';
  } else {
    const m=d.measure;
    h+='<table><tr><th>项</th><th>值</th></tr>'+
      '<tr><td>篇数</td><td class="n">'+m.n+'</td></tr>'+
      '<tr><td>D（均值距离）</td><td class="n">'+m.D.toFixed(4)+'</td></tr>'+
      '<tr><td>95% 区间</td><td class="n">['+m.ci95[0].toFixed(4)+', '+
      m.ci95[1].toFixed(4)+']</td></tr>'+
      '<tr><td>自助法 CV</td><td class="n">'+m.bootstrap_cv.toFixed(4)+'</td></tr>'+
      '<tr><td>距离数</td><td class="n">'+m.n_distances+'</td></tr></table>';
    m.gates.forEach(g=>{h+='<div class="gate '+(g.fired?'f':'ok')+'">'+
      '<span class="tag '+(g.fired?'f':'ok')+'>'+(g.fired?'✕':'✓')+' '+g.id+
      '</span><span>'+g.detail+'</span></div>'});
  }
  h+='<div class="hint" style="margin-top:14px">后端 '+(d.backend||'')+
    ' · 参照系 '+d.frame+'</div>';
  q('out').innerHTML=h;
}
function download(){
  if(!RESULT){q('msg').textContent='先计算一次';return}
  const b=new Blob([JSON.stringify(RESULT,null,2)],{type:'application/json'});
  const u=URL.createObjectURL(b), a=document.createElement('a');
  a.href=u; a.download='pacsp-result.json'; a.click(); URL.revokeObjectURL(u);
}
</script></main></body></html>
"""


def frame_repo(key: str) -> str:
    return T.KNOWN_FRAMES.get(key, (key,))[0]


def do_measure(path: str, frame: str, backend: str, boot: int) -> dict:
    repo = frame_repo(frame)
    texts, _ = T.load_texts(path)
    if len(texts) < T.GATE_MIN_N:
        raise ValueError(f"语料 {path} 只有 {len(texts)} 篇，至少需要 "
                         f"{T.GATE_MIN_N} 篇")
    rng = np.random.default_rng(0)
    with STATE["lock"]:
        if STATE["embedder"] is None:
            STATE["embedder"] = T.Embedder(repo, backend=backend,
                                           onnx_dir=(Path(__file__).resolve().parent / "onnx"))
        emb = STATE["embedder"]
        E = emb.encode(texts)
        active = emb.active_backend
    w = T.within_distances(E)
    point, lo, hi, cv = T.bootstrap_mean_ci(w, boot, rng)
    gates = [T.Gate("estimate-imprecise", cv > T.GATE_SD_CEILING,
                    f"自助法 CV {cv:.4f}，阈值 {T.GATE_SD_CEILING}")]
    return {"n": len(texts), "D": round(point, 6),
            "ci95": [round(lo, 6), round(hi, 6)], "bootstrap_cv": round(cv, 6),
            "n_distances": int(len(w)),
            "gates": [{"id": g.id, "fired": g.fired, "detail": g.detail}
                      for g in gates],
            "backend": active}


def do_compare(pa: str, pb: str, frame: str, backend: str, boot: int) -> dict:
    repo = frame_repo(frame)
    ta, _ = T.load_texts(pa)
    tb, _ = T.load_texts(pb)
    for label, t, p in (("A", ta, pa), ("B", tb, pb)):
        if len(t) < T.GATE_MIN_N:
            raise ValueError(f"语料 {label}（{p}）只有 {len(t)} 篇，至少需要 "
                             f"{T.GATE_MIN_N} 篇")
    rng = np.random.default_rng(0)
    with STATE["lock"]:
        if STATE["embedder"] is None:
            STATE["embedder"] = T.Embedder(repo, backend=backend,
                                           onnx_dir=(Path(__file__).resolve().parent / "onnx"))
        emb = STATE["embedder"]
        EA, EB = emb.encode(ta), emb.encode(tb)
        active = emb.active_backend

    wA, wB = T.within_distances(EA), T.within_distances(EB)
    X = T.cross_distances(EA, EB)
    ma = T.bootstrap_mean_ci(wA, boot, rng)
    mb = T.bootstrap_mean_ci(wB, boot, rng)
    ratio_boot = []
    for _ in range(boot):
        ia = rng.integers(0, len(EA), len(EA))
        ib = rng.integers(0, len(EB), len(EB))
        ratio_boot.append(T.within_distances(EA[ia]).mean() /
                          T.within_distances(EB[ib]).mean())
    ratio_boot = np.asarray(ratio_boot)
    ratio = {"value": float(ma[0] / mb[0]),
             "ci95": [float(np.percentile(ratio_boot, 2.5)),
                      float(np.percentile(ratio_boot, 97.5))]}
    cm, cl, ch, ccv = T.bootstrap_mean_ci(X, boot, rng)
    pooled = np.concatenate([wA, wB])
    sep = float(X.mean() / pooled.mean())
    thr = float(np.percentile(pooled, 95))
    overlap = float((X < thr).mean())
    style = T.style_ratio(ta, tb, 4000, rng)

    # gates_for_compare keys the ratio dict as "point"/"ci95". Passing the display shape
    # here ("value") silently produced an effect-unmeasurable gate on every comparison,
    # which an end-to-end test caught. The adapter is explicit now.
    direction_ok = ratio["ci95"][0] > 1 or ratio["ci95"][1] < 1
    assign = T.loo_assignability(EA, EB)
    gates = T.gates_for_compare(
        len(ta), len(tb), {"cv": ccv},
        {"point": ratio["value"], "ci95": list(ratio["ci95"])},
        overlap, direction_ok, style=style, assign=assign)
    verdict = T.verdict_from(gates, sep, overlap)
    return {
        "collections": {
            "a": {"name": Path(pa).name, "n": len(ta), "D": round(ma[0], 6),
                  "ci95": [round(ma[1], 6), round(ma[2], 6)]},
            "b": {"name": Path(pb).name, "n": len(tb), "D": round(mb[0], 6),
                  "ci95": [round(mb[1], 6), round(mb[2], 6)]}},
        "ratio_within": {"value": round(ratio["value"], 6),
                         "ci95": [round(ratio["ci95"][0], 6),
                                  round(ratio["ci95"][1], 6)]},
        "interval": {"cross_mean": round(cm, 6),
                     "ci95": [round(cl, 6), round(ch, 6)],
                     "n_cross_distances": int(len(X))},
        "separation": {"value": round(sep, 6)},
        "overlap": {"value": round(overlap, 6), "radius": round(thr, 6)},
        "assignability": {"accuracy": round(assign["accuracy"], 6),
                           "n": assign["n"],
                           "negative_margins": assign["negative_margins"],
                           "chance_level": 0.5,
                           "method": "leave-one-out nearest centroid"},
        "style": style,
        "gates": [{"id": g.id, "fired": g.fired, "detail": g.detail}
                  for g in gates],
        "verdict": verdict,
        "backend": active,
        "limits": [
            "D 与所有距离都依赖参照系；只能在同一个参照系内比较",
            "重叠半径取合并组内距离的 P95，换 P90/P99 数值会变",
            "交叉距离并不独立，所以区间是近似的",
            "风格比是字符二元组 Jaccard，粗糙且与 D 无关",
        ],
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # the progress line is noise here

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if u.path == "/api/status":
            self._json({"ready": bool(STATE.get("warm")),
                        "seconds": STATE.get("warm_seconds"),
                        "error": STATE.get("warm_error"),
                        "backend": (STATE["embedder"].active_backend
                                    if STATE.get("embedder") else None)})
            return
        if u.path == "/api/list":
            q = urllib.parse.parse_qs(u.query)
            raw = (q.get("path") or [""])[0].strip()
            p = Path(raw) if raw else Path.home()
            if p.is_file():
                p = p.parent
            if not p.exists():
                p = Path.home()
            dirs = []
            try:
                for d in sorted(p.iterdir(), key=lambda x: x.name.lower()):
                    if d.is_dir() and not d.name.startswith("."):
                        try:
                            n = sum(1 for f in d.iterdir()
                                    if f.is_file() and f.suffix.lower() in T.TEXT_SUFFIXES)
                        except Exception:
                            n = 0
                        dirs.append({"name": d.name, "path": str(d), "txt": n})
            except PermissionError:
                pass
            parent = str(p.parent) if p.parent != p else None
            self._json({"path": str(p), "parent": parent, "dirs": dirs[:200]})
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path != "/api/run":
            self._send(404, b"not found", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._json({"error": f"bad request: {e}"}, 400)
            return
        a = (req.get("a") or "").strip()
        b = (req.get("b") or "").strip()
        frame = req.get("frame") or T.DEFAULT_FRAME
        backend = req.get("backend") or "auto"
        boot = int(req.get("bootstrap") or 2000)
        try:
            if b:
                out = {"compare": do_compare(a, b, frame, backend, boot)}
            else:
                out = {"measure": do_measure(a, frame, backend, boot)}
            out["frame"] = frame
            self._json(out)
        except (ValueError, FileNotFoundError) as e:
            # bad input, not a server fault: report 400 so the caller can tell the
            # difference between "your corpus is too small" and "the server broke"
            self._json({"error": str(e)}, 400)
        except Exception as e:
            traceback.print_exc()
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="pacsp serve",
                                description="local web interface for pacsp")
    ap.add_argument("--port", type=int, default=8731)
    ap.add_argument("--host", default="127.0.0.1",
                    help="loopback by default; do not expose this")
    ap.add_argument("--open", action="store_true", help="open a browser")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "onnx", "sentence-transformers"],
                    help="embedding backend (default auto)")
    ap.add_argument("--no-warm", action="store_true",
                    help="skip loading the model until the first request")
    args = ap.parse_args(argv)

    url = f"http://{args.host}:{args.port}/"
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"  pacsp serving at {url}")
    print("  loading the model in the background; the first measurement waits for it")
    if not args.no_warm:
        warm_up_async(backend=args.backend)
    print(f"  bound to {args.host}: only this machine can reach it.")
    print(f"  frames: {', '.join(T.KNOWN_FRAMES)}")
    print("  Ctrl-C to stop.")
    if args.open:
        import webbrowser
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
