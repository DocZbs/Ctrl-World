#!/usr/bin/env python3
import argparse
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


INDEX_HTML = """<!doctype html>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Media Gallery</title>
<style>
  :root{--bg:#0b0d10;--panel:#121620;--muted:#9aa4b2;--text:#e7edf6;--bd:rgba(255,255,255,.12);--ac:#5bbcff;--cols:6}
  *{box-sizing:border-box} body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;background:var(--bg);color:var(--text)}
  header{position:sticky;top:0;z-index:9;padding:10px 12px;border-bottom:1px solid var(--bd);background:rgba(11,13,16,.9);backdrop-filter:blur(8px);display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  .brand{font-weight:700;margin-right:6px}
  .input{height:32px;border-radius:10px;border:1px solid var(--bd);background:rgba(18,22,32,.75);color:var(--text);padding:0 10px;min-width:240px}
  .num{min-width:72px;width:72px}
  .btn{height:32px;border-radius:10px;border:1px solid var(--bd);background:rgba(18,22,32,.9);color:var(--text);padding:0 10px;cursor:pointer}
  .btn[disabled]{opacity:.5;cursor:not-allowed}
  .ctl{display:inline-flex;align-items:center;gap:6px;padding:4px 8px;border:1px solid var(--bd);border-radius:12px;background:rgba(18,22,32,.35);font-size:12px;color:var(--muted)}
  main{padding:14px;max-width:1800px;margin:0 auto}
  .row{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}
  .crumbs{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
  .crumb{padding:6px 10px;border:1px solid var(--bd);border-radius:999px;background:rgba(18,22,32,.65);cursor:pointer}
  .grid{margin-top:12px;display:grid;gap:10px;grid-template-columns:repeat(var(--cols),minmax(0,1fr))}
  .card{border:1px solid var(--bd);border-radius:14px;overflow:hidden;background:var(--panel);cursor:pointer}
  .thumb{aspect-ratio:16/9;background:#000;position:relative}
  .thumb video,.thumb img{width:100%;height:100%;object-fit:cover;display:block}
  .tag{position:absolute;left:10px;top:10px;font-size:11px;padding:3px 8px;border-radius:999px;border:1px solid rgba(255,255,255,.18);background:rgba(0,0,0,.35)}
  .body{padding:8px 10px}
  .name{font-size:13px;font-weight:650;word-break:break-all}
  .sub{margin-top:4px;font-size:12px;color:var(--muted);display:flex;gap:8px;flex-wrap:wrap}
  a{color:var(--ac);text-decoration:none} a:hover{text-decoration:underline}
  dialog{border:none;border-radius:16px;padding:0;width:min(1100px,calc(100vw - 24px));background:#0f1320;color:var(--text)}
  dialog::backdrop{background:rgba(0,0,0,.65)}
  .dlgTop{display:flex;justify-content:space-between;gap:10px;align-items:center;padding:10px 12px;border-bottom:1px solid rgba(255,255,255,.12);flex-wrap:wrap}
  .dlgBody{padding:12px}
  .dlgBody video,.dlgBody img{width:100%;height:auto;border-radius:12px;border:1px solid rgba(255,255,255,.10);background:#000}
</style>
<header>
  <span class="brand">Media Gallery</span>
  <input id="path" class="input" placeholder="/mnt/... or /nvme-fast/..."/>
  <button id="go" class="btn">Go</button>
  <input id="filter" class="input" placeholder="filter"/>
  <span class="ctl">Cols <input id="cols" class="input num" type="number" min="1" max="20"/></span>
  <span class="ctl">Rows <input id="rows" class="input num" type="number" min="1" max="60"/></span>
  <span class="ctl"><label><input id="autoplay" type="checkbox"/> Autoplay</label></span>
  <span class="ctl"><label><input id="controls" type="checkbox"/> Controls</label></span>
  <button id="prev" class="btn">Prev</button>
  <span id="page" class="ctl">page</span>
  <button id="next" class="btn">Next</button>
</header>
<main>
  <div class="row">
    <div id="crumbs" class="crumbs"></div>
    <div class="ctl"><span id="count"></span><span id="root"></span></div>
  </div>
  <div id="grid" class="grid"></div>
</main>
<dialog id="dlg">
  <div class="dlgTop">
    <div id="dlgTitle" style="font-weight:700;word-break:break-all"></div>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <a id="open" class="btn" target="_blank" rel="noreferrer">Open</a>
      <button id="copy" class="btn">Copy URL</button>
      <button id="close" class="btn">Close</button>
    </div>
  </div>
  <div id="dlgBody" class="dlgBody"></div>
</dialog>
<script>
const $=s=>document.querySelector(s);
const LS={c:"mg_cols",r:"mg_rows",a:"mg_autoplay",k:"mg_controls"};
const S={root:"",path:"/",items:[],filter:"",cols:6,rows:3,page:0,autoplay:false,controls:false};
const fmtB=n=>{if(!isFinite(n))return"";const u=["B","KB","MB","GB","TB"];let i=0,x=n;for(;x>=1024&&i<u.length-1;i++)x/=1024;return x.toFixed(i?1:0)+" "+u[i]};
const fmtT=t=>new Date(t*1000).toLocaleString();
const urlPath=()=>new URL(location.href).searchParams.get("path")||"/";
const setUrl=p=>{const u=new URL(location.href);u.searchParams.set("path",p);history.pushState({},'',u)};
async function apiList(p){const u=new URL("/__api__/list",location.origin);u.searchParams.set("path",(p||"/").trim());const j=await (await fetch(u)).json();if(!j.ok)throw Error(j.error||"api");return j;}
function crumbs(p){
  const el=$("#crumbs");el.innerHTML="";
  const parts=p.replace(/^\\/+/, "").split("/").filter(Boolean);
  const cs=[{n:"root",p:"/"}];let acc="";
  for(const part of parts){acc+="/"+part;cs.push({n:part,p:acc});}
  cs.forEach((c,i)=>{const b=document.createElement("span");b.className="crumb";b.textContent=c.n;b.onclick=()=>nav(c.p);el.appendChild(b);if(i<cs.length-1){const s=document.createElement("span");s.style.color="var(--muted)";s.textContent="›";el.appendChild(s);}});
}
function lazy(grid){
  const io=new IntersectionObserver(es=>es.forEach(e=>{if(!e.isIntersecting)return;const el=e.target,src=el.dataset.src;if(src){el.src=src;delete el.dataset.src;}if(el.tagName==="VIDEO"&&S.autoplay)el.play().catch(()=>{});io.unobserve(el);}),{rootMargin:"250px"});
  grid.querySelectorAll("video[data-src],img[data-src]").forEach(el=>io.observe(el));
}
function openDlg(it){
  $("#dlgTitle").textContent=it.path;$("#open").href=it.url;$("#dlgBody").innerHTML="";
  const abs=new URL(it.url,location.origin).toString();
  $("#copy").onclick=()=>navigator.clipboard.writeText(abs).then(()=>$("#copy").textContent="Copied!",()=>$("#copy").textContent="Copy failed").finally(()=>setTimeout(()=>$("#copy").textContent="Copy URL",900));
  if(it.kind==="video"){const v=document.createElement("video");v.src=it.url;v.controls=true;v.playsInline=true;v.preload="metadata";$("#dlgBody").appendChild(v);}
  else if(it.kind==="image"){const img=document.createElement("img");img.src=it.url;$("#dlgBody").appendChild(img);}
  else {const p=document.createElement("p");p.innerHTML='File: <a href="'+it.url+'" target="_blank" rel="noreferrer">'+it.url+"</a>";$("#dlgBody").appendChild(p);}
  $("#dlg").showModal();
}
function card(it){
  const d=document.createElement("div");d.className="card";
  const t=document.createElement("div");t.className="thumb";
  const tag=document.createElement("div");tag.className="tag";tag.textContent=it.is_dir?"DIR":it.kind.toUpperCase();t.appendChild(tag);
  if(it.is_dir){t.style.display="grid";t.style.placeItems="center";t.style.color="var(--muted)";t.style.fontSize="22px";t.textContent="DIR";}
  else if(it.kind==="video"){const v=document.createElement("video");v.muted=true;v.loop=true;v.playsInline=true;v.controls=!!S.controls;v.preload=S.autoplay?"metadata":"none";v.dataset.src=it.url;if(!S.autoplay){v.onmouseenter=()=>v.play().catch(()=>{});v.onmouseleave=()=>{v.pause();v.currentTime=0;};}t.appendChild(v);}
  else if(it.kind==="image"){const img=document.createElement("img");img.loading="lazy";img.dataset.src=it.url;t.appendChild(img);}
  else {t.style.display="grid";t.style.placeItems="center";t.style.color="var(--muted)";t.textContent="FILE";}
  const b=document.createElement("div");b.className="body";
  b.innerHTML='<div class="name">'+it.name+'</div><div class="sub"><span>'+fmtB(it.size)+'</span><span>'+fmtT(it.mtime)+'</span></div>';
  d.appendChild(t);d.appendChild(b);
  d.onclick=()=>it.is_dir?nav(it.path):openDlg(it);
  return d;
}
function render(){
  $("#path").value=S.path;$("#filter").value=S.filter;$("#cols").value=S.cols;$("#rows").value=S.rows;
  $("#autoplay").checked=S.autoplay;$("#controls").checked=S.controls;
  document.documentElement.style.setProperty("--cols", String(S.cols));
  crumbs(S.path);
  const f=S.items.filter(it=>!S.filter||it.name.toLowerCase().includes(S.filter.toLowerCase()));
  const pageSize=Math.max(1,S.cols*S.rows), pages=Math.max(1,Math.ceil(f.length/pageSize));
  S.page=Math.min(S.page,pages-1);
  $("#count").textContent=f.length+" items";$("#root").textContent=S.root?(" · root: "+S.root):"";
  $("#page").textContent="page "+(S.page+1)+"/"+pages;
  $("#prev").disabled=S.page<=0;$("#next").disabled=S.page>=pages-1;
  const g=$("#grid");g.innerHTML="";f.slice(S.page*pageSize,S.page*pageSize+pageSize).forEach(it=>g.appendChild(card(it)));
  lazy(g);
}
async function load(p){
  const j=await apiList(p);
  S.root=j.root;S.path=j.path||p;S.items=j.items||[];S.page=0;render();
}
async function nav(p){
  const clean=(p||"/").replace(/\\/+$/,"")||"/";setUrl(clean);await load(clean);
}
function init(){
  const sc=+localStorage.getItem(LS.c), sr=+localStorage.getItem(LS.r);
  if(sc>=1&&sc<=20)S.cols=sc;if(sr>=1&&sr<=60)S.rows=sr;
  S.autoplay=localStorage.getItem(LS.a)==="1";S.controls=localStorage.getItem(LS.k)==="1";
  $("#go").onclick=()=>nav($("#path").value||"/");
  $("#path").onkeydown=e=>{ if(e.key==="Enter"){ e.preventDefault(); nav($("#path").value||"/"); } };
  $("#filter").oninput=e=>{S.filter=e.target.value||"";S.page=0;render();};
  $("#cols").onchange=e=>{S.cols=Math.max(1,Math.min(20,+e.target.value||6));localStorage.setItem(LS.c,S.cols);S.page=0;render();};
  $("#rows").onchange=e=>{S.rows=Math.max(1,Math.min(60,+e.target.value||3));localStorage.setItem(LS.r,S.rows);S.page=0;render();};
  $("#autoplay").onchange=e=>{S.autoplay=!!e.target.checked;localStorage.setItem(LS.a,S.autoplay?"1":"0");render();};
  $("#controls").onchange=e=>{S.controls=!!e.target.checked;localStorage.setItem(LS.k,S.controls?"1":"0");render();};
  $("#prev").onclick=()=>{S.page=Math.max(0,S.page-1);render();};
  $("#next").onclick=()=>{S.page+=1;render();};
  $("#close").onclick=()=>$("#dlg").close();
  window.onpopstate=()=>load(urlPath());
  load(urlPath());
}
init();
</script>
"""

GALLERY_PATHS = {"/", "/__gallery__", "/__gallery__/"}
API_LIST_PATH = "/__api__/list"

VIDEO_EXTS = {".mp4", ".webm", ".mov"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def _norm_query_path(root: Path, path_str: str) -> str:
    p = (path_str or "/").strip()
    root_str = str(root)
    if p == root_str:
        return "/"
    if root_str != "/" and p.startswith(root_str + os.sep):
        p = p[len(root_str) :]
    return p if p.startswith("/") else "/" + p


def _safe_dir(root: Path, req_path: str) -> Path:
    req_path = _norm_query_path(root, req_path)
    # Use absolute paths (not real paths) so that browsing through symlinked
    # mount points under root (e.g. /mnt/nvme-fast -> /nvme-fast) works.
    root_abs = root.absolute()
    candidate = (root_abs / req_path.lstrip("/")).absolute()
    root_str = str(root_abs)
    cand_str = str(candidate)
    if candidate == root_abs or os.path.commonpath([root_str, cand_str]) == root_str:
        return candidate
    raise ValueError("path escapes root")


def _list_dir(root: Path, req_path: str):
    abs_dir = _safe_dir(root, req_path)
    if not abs_dir.exists() or not abs_dir.is_dir():
        raise ValueError("not a directory")
    rel = _norm_query_path(root, req_path).strip("/")
    out = []
    with os.scandir(abs_dir) as it:
        for e in it:
            if e.name.startswith("."):
                continue
            try:
                st = e.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            is_dir = e.is_dir(follow_symlinks=False)
            ext = Path(e.name).suffix.lower()
            kind = (
                "dir"
                if is_dir
                else ("video" if ext in VIDEO_EXTS else ("image" if ext in IMAGE_EXTS else "file"))
            )
            rel_path = "/".join([p for p in [rel, e.name] if p])
            out.append({"name": e.name, "path": "/" + rel_path, "is_dir": is_dir, "size": int(st.st_size), "mtime": float(st.st_mtime), "kind": kind, "url": "/" + rel_path})
    out.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
    return out


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, root=None, **kwargs):
        self._root = Path(root).resolve() if root else None
        self._root_url = str(self._root) if self._root else ""
        super().__init__(*args, directory=str(directory) if directory else None, **kwargs)

    def _send_bytes(self, code: int, content_type: str, body: bytes | None, *, content_length: int | None = None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        if content_length is None:
            content_length = 0 if body is None else len(body)
        self.send_header("Content-Length", str(content_length))
        self.end_headers()
        if body is not None:
            self.wfile.write(body)

    def _send_index(self, with_body: bool):
        b = INDEX_HTML.encode("utf-8")
        self._send_bytes(200, "text/html; charset=utf-8", b if with_body else None, content_length=len(b))

    def _send_json(self, code: int, obj: dict, with_body: bool):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send_bytes(code, "application/json; charset=utf-8", b if with_body else None, content_length=len(b))

    def list_directory(self, path):  # noqa: N802
        self.send_error(403, "Directory listing disabled; use /__gallery__/")
        return None

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in GALLERY_PATHS:
            self._send_index(with_body=True)
            return

        if parsed.path == API_LIST_PATH:
            qs = parse_qs(parsed.query)
            req = (qs.get("path") or [""])[0]
            try:
                items = _list_dir(self._root, req)  # type: ignore[arg-type]
                resp = {
                    "ok": True,
                    "root": str(self._root),
                    "path": _norm_query_path(self._root, req),  # type: ignore[arg-type]
                    "items": items,
                }
                self._send_json(200, resp, with_body=True)
            except Exception as e:
                self._send_json(400, {"ok": False, "error": str(e), "path": req}, with_body=True)
            return

        # Allow URLs like /mnt/... when root is /mnt
        if self._root_url and self._root_url != "/" and parsed.path.startswith(self._root_url + "/"):
            self.path = parsed.path[len(self._root_url) :] + (("?" + parsed.query) if parsed.query else "")
        super().do_GET()

    def do_HEAD(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in GALLERY_PATHS:
            self._send_index(with_body=False)
            return

        if parsed.path == API_LIST_PATH:
            qs = parse_qs(parsed.query)
            req = (qs.get("path") or [""])[0]
            try:
                items = _list_dir(self._root, req)  # type: ignore[arg-type]
                resp = {
                    "ok": True,
                    "root": str(self._root),
                    "path": _norm_query_path(self._root, req),  # type: ignore[arg-type]
                    "items": items,
                }
                self._send_json(200, resp, with_body=False)
            except Exception as e:
                self._send_json(400, {"ok": False, "error": str(e), "path": req}, with_body=False)
            return

        # Let SimpleHTTPRequestHandler handle HEAD (incl. range-less media headers)
        super().do_HEAD()

    def log_message(self, fmt, *args):  # noqa: N802
        print(f"[{self.log_date_time_string()}] {fmt % args}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt")
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"--root is not a directory: {root}")

    def handler(*a, **kw):
        return Handler(*a, directory=str(root), root=str(root), **kw)

    httpd = ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"Serving root: {root}")
    print(f"Gallery:      http://{args.bind}:{args.port}/__gallery__/")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
