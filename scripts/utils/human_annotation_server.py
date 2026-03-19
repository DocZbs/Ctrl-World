#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


APP_PATHS = {"/", "/__annotate__", "/__annotate__/"}
API_ITEMS_PATH = "/__api__/items"
API_SAVE_PATH = "/__api__/save"
API_HEALTH_PATH = "/__api__/health"

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
ALLOWED_LABELS = {"success", "failure", "uncertain", "skip"}
ALLOWED_FAILURE_REASONS = {
    "world_model_failure",
    "vla_failure",
    "task_scene_mismatch",
}
FAILURE_REASON_KEYS = [
    "world_model_failure",
    "vla_failure",
    "task_scene_mismatch",
]
TRAJ_RE = re.compile(r"traj_(\d+)_")
TASK_RE = re.compile(r"task_(\d+)")


INDEX_HTML = """<!doctype html>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Human Annotation</title>
<style>
  :root{--bg:#0b1020;--panel:#121a2e;--panel2:#1a243d;--bd:rgba(255,255,255,.12);--text:#ebf1ff;--muted:#9ca8c4;--ok:#34d399;--bad:#f87171;--warn:#fbbf24;--skip:#93c5fd;--primary:#60a5fa}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial}
  header{position:sticky;top:0;z-index:5;background:rgba(11,16,32,.92);backdrop-filter:blur(10px);border-bottom:1px solid var(--bd);padding:10px 14px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
  .title{font-size:18px;font-weight:700;margin-right:8px}
  .chip{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--bd);border-radius:999px;padding:5px 10px;color:var(--muted);font-size:12px;background:rgba(255,255,255,.03)}
  .input,.btn,textarea{background:var(--panel2);color:var(--text);border:1px solid var(--bd);border-radius:10px}
  .input{height:34px;padding:0 10px;min-width:220px}
  .btn{height:34px;padding:0 12px;cursor:pointer}
  .btn:disabled{opacity:.55;cursor:not-allowed}
  .wrap{display:grid;grid-template-columns:420px 1fr;gap:12px;max-width:1800px;margin:0 auto;padding:12px}
  .card{border:1px solid var(--bd);border-radius:14px;background:var(--panel);overflow:hidden}
  #items{height:calc(100vh - 98px);overflow:auto}
  .row{padding:9px 11px;border-bottom:1px solid rgba(255,255,255,.06);cursor:pointer;display:flex;gap:8px;align-items:flex-start}
  .row:hover{background:rgba(255,255,255,.03)}
  .row.active{background:rgba(96,165,250,.17)}
  .badge{font-size:11px;line-height:1;border-radius:999px;padding:4px 7px;border:1px solid var(--bd);white-space:nowrap}
  .b-ok{color:var(--ok);border-color:rgba(52,211,153,.35)}
  .b-bad{color:var(--bad);border-color:rgba(248,113,113,.35)}
  .b-uncertain{color:var(--warn);border-color:rgba(251,191,36,.35)}
  .b-skip{color:var(--skip);border-color:rgba(147,197,253,.35)}
  .b-empty{color:var(--muted)}
  .meta{display:flex;flex-direction:column;gap:3px;min-width:0}
  .name{font-size:13px;font-weight:650;word-break:break-all}
  .sub{font-size:12px;color:var(--muted);word-break:break-all}
  .viewer{padding:12px;display:flex;flex-direction:column;gap:10px}
  video{width:100%;max-height:62vh;border:1px solid var(--bd);border-radius:12px;background:#000}
  .line{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
  .inst{padding:10px 12px;border:1px solid var(--bd);border-radius:10px;background:rgba(255,255,255,.03);line-height:1.5}
  .labels{display:flex;gap:8px;flex-wrap:wrap}
  .lab{padding:8px 12px;border-radius:10px;border:1px solid var(--bd);background:var(--panel2);cursor:pointer}
  .lab.active{outline:2px solid var(--primary)}
  .lab.success{color:var(--ok)}
  .lab.failure{color:var(--bad)}
  .lab.uncertain{color:var(--warn)}
  .lab.skip{color:var(--skip)}
  .failure-reasons{display:flex;gap:8px;flex-wrap:wrap}
  .reason-btn{padding:7px 10px;border-radius:10px;border:1px solid var(--bd);background:var(--panel2);color:var(--text);cursor:pointer;font-size:13px}
  .reason-btn.active{outline:2px solid var(--primary);color:#dbeafe}
  textarea{width:100%;min-height:94px;padding:9px 10px;resize:vertical}
  .hint{font-size:12px;color:var(--muted)}
  a{color:#93c5fd;text-decoration:none}
  a:hover{text-decoration:underline}
  @media (max-width:1080px){.wrap{grid-template-columns:1fr}.card#items{height:320px}}
</style>

<header>
  <span class="title">人工标注平台</span>
  <span id="summary" class="chip">加载中...</span>
  <span id="outputPath" class="chip"></span>
  <label class="chip">标注人 <input id="annotator" class="input" style="min-width:140px;height:28px" placeholder="可选"/></label>
  <label class="chip"><input id="autoNext" type="checkbox" checked/> 保存后自动下一条</label>
  <span id="saveState" class="chip">未保存</span>
</header>

<main class="wrap">
  <section class="card" id="items"></section>

  <section class="card viewer">
    <video id="video" controls preload="metadata"></video>

    <div class="line">
      <button id="prev" class="btn">上一条 ←</button>
      <button id="next" class="btn">下一条 →</button>
      <button id="nextUnlabeled" class="btn">下一条未标注 U</button>
      <span id="idx" class="chip"></span>
    </div>

    <div class="line">
      <span id="traj" class="chip"></span>
      <span id="rollout" class="chip"></span>
      <a id="openVideo" class="chip" target="_blank" rel="noreferrer">打开视频</a>
      <a id="openInfo" class="chip" target="_blank" rel="noreferrer">打开JSON</a>
    </div>

    <div id="instruction" class="inst"></div>

    <div class="labels">
      <button class="lab success" data-label="success">成功 (1)</button>
      <button class="lab failure" data-label="failure">失败 (0)</button>
      <button class="lab uncertain" data-label="uncertain">不确定 (2)</button>
      <button class="lab skip" data-label="skip">跳过 (3)</button>
      <button class="lab" data-label="">清空标签 (C)</button>
    </div>

    <div id="failureReasonWrap" style="display:none">
      <div class="line"><span class="chip">失败原因（必选）</span></div>
      <div class="failure-reasons">
        <button class="reason-btn" data-reason="world_model_failure">1 世界模型失败</button>
        <button class="reason-btn" data-reason="vla_failure">2 VLA失败</button>
        <button class="reason-btn" data-reason="task_scene_mismatch">3 生成任务和场景不能匹配</button>
      </div>
    </div>

    <textarea id="comment" placeholder="备注（可选，输入后自动保存）"></textarea>
    <div class="hint">快捷键：1 成功，0 失败；失败后按 1-3 选失败原因并完成标注；2 不确定，3 跳过，C 清空，←/→ 切换，U 跳到下一条未标注。</div>
  </section>
</main>

<script>
const S = {
  items: [],
  labels: {},
  idx: 0,
  dirty: false,
  saving: false,
  pendingTimer: null,
  saveSeq: 0,
  lastRenderedUid: "",
};

const $ = (sel) => document.querySelector(sel);

const statusText = (label) => {
  if (label === "success") return ["成功", "b-ok"];
  if (label === "failure") return ["失败", "b-bad"];
  if (label === "uncertain") return ["不确定", "b-uncertain"];
  if (label === "skip") return ["跳过", "b-skip"];
  return ["未标注", "b-empty"];
};

const esc = (s) => String(s ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#39;");

const currentItem = () => S.items[S.idx] || null;
const getLabelEntry = (uid) => S.labels[uid] || {};
const getActiveLabel = () => (document.querySelector(".lab.active")?.dataset.label || "");
const getActiveFailureReason = () => (document.querySelector(".reason-btn.active")?.dataset.reason || "");

function setActiveFailureReason(reason) {
  document.querySelectorAll(".reason-btn").forEach((btn) => {
    const v = btn.dataset.reason || "";
    btn.classList.toggle("active", v === (reason || ""));
  });
}

function clickFailureReasonByIndex(key) {
  const num = Number(key);
  if (!Number.isInteger(num) || num < 1) {
    return false;
  }
  const buttons = Array.from(document.querySelectorAll(".reason-btn"));
  const btn = buttons[num - 1];
  if (!btn) {
    return false;
  }
  btn.click();
  return true;
}

function syncFailureReasonVisibility(label) {
  const wrap = $("#failureReasonWrap");
  if (!wrap) return;
  wrap.style.display = label === "failure" ? "block" : "none";
}

function computeSummary() {
  const total = S.items.length;
  const counts = {success:0, failure:0, uncertain:0, skip:0};
  let labeled = 0;
  for (const item of S.items) {
    const label = (S.labels[item.uid] || {}).label;
    if (counts[label] !== undefined) {
      counts[label] += 1;
      labeled += 1;
    }
  }
  return {total, labeled, remaining: Math.max(total - labeled, 0), counts};
}

function setSummary() {
  const sm = computeSummary();
  $("#summary").textContent = `已标注 ${sm.labeled}/${sm.total} · 剩余 ${sm.remaining} · ✅${sm.counts.success} ❌${sm.counts.failure} 🤔${sm.counts.uncertain} ⏭${sm.counts.skip}`;
}

function setSaveState(text, isError = false) {
  const node = $("#saveState");
  node.textContent = text;
  node.style.color = isError ? "#fca5a5" : "";
}

function setActiveLabel(label) {
  document.querySelectorAll(".lab").forEach((btn) => {
    const v = btn.dataset.label || "";
    btn.classList.toggle("active", v === (label || ""));
  });
  syncFailureReasonVisibility(label || "");
}

function renderList() {
  const box = $("#items");
  box.innerHTML = "";
  S.items.forEach((item, index) => {
    const entry = getLabelEntry(item.uid);
    const [txt, cls] = statusText(entry.label || "");
    const displayId = item.display_id || `traj_${String(item.traj_id).padStart(4, "0")}`;
    const div = document.createElement("div");
    div.className = "row" + (index === S.idx ? " active" : "");
    div.innerHTML = `
      <span class="badge ${cls}">${txt}</span>
      <div class="meta">
        <div class="name">${index + 1}. ${esc(displayId)}</div>
        <div class="sub">${esc(item.rollout_rel)}</div>
      </div>
    `;
    div.onclick = () => gotoIndex(index);
    box.appendChild(div);
  });
}

function renderCurrent() {
  const item = currentItem();
  if (!item) {
    $("#instruction").textContent = "没有可标注的数据。";
    $("#video").removeAttribute("src");
    $("#idx").textContent = "0 / 0";
    renderList();
    return;
  }

  const entry = getLabelEntry(item.uid);
  const displayId = item.display_id || `traj_${String(item.traj_id).padStart(4, "0")}`;
  $("#idx").textContent = `${S.idx + 1} / ${S.items.length}`;
  $("#traj").textContent = displayId;
  $("#rollout").textContent = item.rollout_rel;
  $("#instruction").textContent = item.instruction || "(无 instruction 字段)";
  $("#openVideo").href = item.video_url;
  $("#openInfo").href = item.info_url;
  if (S.lastRenderedUid !== item.uid) {
    const v = $("#video");
    v.src = item.video_url;
    v.load();
    S.lastRenderedUid = item.uid;
  }
  setActiveFailureReason(entry.failure_reason || "");
  $("#comment").value = entry.comment || "";
  setActiveLabel(entry.label || "");
  $("#prev").disabled = S.idx <= 0;
  $("#next").disabled = S.idx >= S.items.length - 1;
  renderList();
  setSummary();
}

function findNextUnlabeled(startExclusive) {
  for (let i = startExclusive + 1; i < S.items.length; i += 1) {
    const uid = S.items[i].uid;
    if (!getLabelEntry(uid).label) {
      return i;
    }
  }
  return -1;
}

async function fetchJson(url, options = {}) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.ok === false) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data;
}

async function saveCurrent({navigateNext = false} = {}) {
  const item = currentItem();
  if (!item) return false;
  if (S.saving) return false;

  const selected = document.querySelector(".lab.active");
  const label = selected ? (selected.dataset.label || null) : null;
  const failureReason = getActiveFailureReason();
  const comment = $("#comment").value.trim();
  const annotator = $("#annotator").value.trim();

  if (label === "failure" && !failureReason) {
    setSaveState("请选择失败原因", true);
    return false;
  }

  S.saving = true;
  const seq = ++S.saveSeq;
  setSaveState("保存中...");

  try {
    const data = await fetchJson("/__api__/save", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        uid: item.uid,
        label,
        failure_reason: label === "failure" ? failureReason : "",
        comment,
        annotator,
      }),
    });

    if (seq !== S.saveSeq) {
      return false;
    }

    if (data.saved) {
      S.labels[item.uid] = data.saved;
    } else {
      delete S.labels[item.uid];
    }
    S.dirty = false;
    setSaveState(`已保存 ${new Date().toLocaleTimeString()}`);
    setSummary();
    renderList();

    if (navigateNext && $("#autoNext").checked) {
      const nextIdx = Math.min(S.items.length - 1, S.idx + 1);
      if (nextIdx !== S.idx) {
        S.idx = nextIdx;
        renderCurrent();
      }
    }
    return true;
  } catch (err) {
    setSaveState(`保存失败: ${err.message || err}`, true);
    return false;
  } finally {
    S.saving = false;
  }
}

function scheduleAutoSave() {
  S.dirty = true;
  if (S.pendingTimer) {
    clearTimeout(S.pendingTimer);
  }
  S.pendingTimer = setTimeout(() => {
    S.pendingTimer = null;
    saveCurrent();
  }, 550);
}

async function gotoIndex(index) {
  if (index < 0 || index >= S.items.length || index === S.idx) return;
  if (S.pendingTimer) {
    clearTimeout(S.pendingTimer);
    S.pendingTimer = null;
  }
  if (S.dirty) {
    await saveCurrent();
  }
  S.idx = index;
  renderCurrent();
}

async function gotoNextUnlabeled() {
  const nextIdx = findNextUnlabeled(S.idx);
  if (nextIdx === -1) {
    setSaveState("后面没有未标注项");
    return;
  }
  await gotoIndex(nextIdx);
}

async function loadData() {
  const data = await fetchJson("/__api__/items");
  S.items = data.items || [];
  S.labels = data.labels || {};
  $("#outputPath").textContent = `输出: ${data.output_path || ""}`;

  let firstUnlabeled = S.items.findIndex((item) => !getLabelEntry(item.uid).label);
  if (firstUnlabeled < 0) firstUnlabeled = 0;
  S.idx = firstUnlabeled;
  renderCurrent();
}

function bindEvents() {
  $("#prev").onclick = () => gotoIndex(S.idx - 1);
  $("#next").onclick = () => gotoIndex(S.idx + 1);
  $("#nextUnlabeled").onclick = () => gotoNextUnlabeled();

  document.querySelectorAll(".lab").forEach((btn) => {
    btn.onclick = () => {
      const nextLabel = btn.dataset.label || "";
      setActiveLabel(nextLabel);
      if (nextLabel !== "failure") {
        setActiveFailureReason("");
      }
      S.dirty = true;
      if (nextLabel === "failure" && !getActiveFailureReason()) {
        setSaveState("请选择失败原因", true);
        return;
      }
      saveCurrent({navigateNext: nextLabel !== ""});
    };
  });

  document.querySelectorAll(".reason-btn").forEach((btn) => {
    btn.onclick = () => {
      setActiveFailureReason(btn.dataset.reason || "");
      if (getActiveLabel() !== "failure") {
        return;
      }
      S.dirty = true;
      saveCurrent({navigateNext: true});
    };
  });

  $("#comment").addEventListener("input", () => scheduleAutoSave());
  $("#annotator").addEventListener("change", () => scheduleAutoSave());

  document.addEventListener("keydown", (e) => {
    if ((e.target && ["INPUT", "TEXTAREA"].includes(e.target.tagName)) && e.key !== "Escape") {
      return;
    }
    if (e.key === "ArrowLeft") { e.preventDefault(); gotoIndex(S.idx - 1); return; }
    if (e.key === "ArrowRight") { e.preventDefault(); gotoIndex(S.idx + 1); return; }
    if (e.key === "u" || e.key === "U") { e.preventDefault(); gotoNextUnlabeled(); return; }

    if (getActiveLabel() === "failure" && /^[1-3]$/.test(e.key)) {
      e.preventDefault();
      clickFailureReasonByIndex(e.key);
      return;
    }

    if (e.key === "1") { e.preventDefault(); document.querySelector('.lab[data-label="success"]').click(); return; }
    if (e.key === "0") { e.preventDefault(); document.querySelector('.lab[data-label="failure"]').click(); return; }
    if (e.key === "2") { e.preventDefault(); document.querySelector('.lab[data-label="uncertain"]').click(); return; }
    if (e.key === "3") { e.preventDefault(); document.querySelector('.lab[data-label="skip"]').click(); return; }
    if (e.key === "c" || e.key === "C") { e.preventDefault(); document.querySelector('.lab[data-label=""]').click(); }
  });

  window.addEventListener("beforeunload", (e) => {
    if (S.dirty || S.saving) {
      e.preventDefault();
      e.returnValue = "";
    }
  });
}

async function main() {
  bindEvents();
  try {
    await loadData();
    setSaveState("就绪");
  } catch (err) {
    setSaveState(`加载失败: ${err.message || err}`, true);
    $("#instruction").textContent = String(err.message || err);
  }
}

main();
</script>
"""


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _extract_traj_id(name: str) -> int | None:
    m = TRAJ_RE.search(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _extract_task_id(name: str) -> int | None:
    m = TASK_RE.search(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _read_instruction(info_path: Path) -> str:
    try:
        data = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    text = data.get("instructions", "")
    return str(text).strip()


def _read_episode_meta(episode_path: Path) -> tuple[str, int | None, str]:
    try:
        data = json.loads(episode_path.read_text(encoding="utf-8"))
    except Exception:
        return "", None, ""

    instruction = ""
    for key in ["task_instruction", "instructions", "instruction", "text", "prompt", "goal"]:
        value = data.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            instruction = text
            break

    task_id = None
    raw_task_id = data.get("task_id")
    if raw_task_id is not None:
        try:
            task_id = int(raw_task_id)
        except (TypeError, ValueError):
            task_id = None

    video_path = str(data.get("video_path") or "").strip()
    return instruction, task_id, video_path


def _collect_rollout_dirs(root: Path, rollout_dir_name: str) -> list[Path]:
    candidates = []
    if (root / "info").is_dir() and (root / "video").is_dir():
        candidates.append(root)
    for p in root.rglob(rollout_dir_name):
        if p.is_dir() and (p / "info").is_dir() and (p / "video").is_dir():
            candidates.append(p)

    uniq: dict[str, Path] = {}
    for p in candidates:
        uniq[str(p.resolve())] = p
    return sorted(uniq.values(), key=lambda x: str(x.relative_to(root)) if x != root else "")


def _collect_items(root: Path, rollout_dir_name: str, limit: int | None = None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    rollout_dirs = _collect_rollout_dirs(root, rollout_dir_name)
    for rollout_dir in rollout_dirs:
        info_dir = rollout_dir / "info"
        video_dir = rollout_dir / "video"
        info_map: dict[int, Path] = {}
        for p in sorted(info_dir.glob("*.json")):
            traj_id = _extract_traj_id(p.name)
            if traj_id is None or traj_id in info_map:
                continue
            info_map[traj_id] = p

        video_map: dict[int, Path] = {}
        for p in sorted(video_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
                continue
            traj_id = _extract_traj_id(p.name)
            if traj_id is None or traj_id in video_map:
                continue
            video_map[traj_id] = p

        common_ids = sorted(set(info_map) & set(video_map))
        for traj_id in common_ids:
            info_path = info_map[traj_id]
            video_path = video_map[traj_id]
            rollout_rel = rollout_dir.relative_to(root).as_posix() if rollout_dir != root else "."
            uid = f"{rollout_rel}::traj_{traj_id:04d}"
            items.append(
                {
                    "uid": uid,
                    "display_id": f"traj_{traj_id:04d}",
                    "traj_id": traj_id,
                    "rollout_rel": rollout_rel,
                    "instruction": _read_instruction(info_path),
                    "video_rel": video_path.relative_to(root).as_posix(),
                    "video_url": "/" + video_path.relative_to(root).as_posix(),
                    "info_rel": info_path.relative_to(root).as_posix(),
                    "info_url": "/" + info_path.relative_to(root).as_posix(),
                    "video_name": video_path.name,
                    "info_name": info_path.name,
                }
            )

    if not items:
        episodes_dir = root / "episodes"
        videos_dir = root / "videos"
        if episodes_dir.is_dir() and videos_dir.is_dir():
            video_map: dict[str, Path] = {}
            for p in sorted(videos_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                    video_map[p.stem] = p

            for episode_path in sorted(episodes_dir.glob("*.json")):
                episode_stem = episode_path.stem
                instruction, task_id_from_json, video_path_hint = _read_episode_meta(episode_path)

                video_path = video_map.get(episode_stem)
                if video_path is None and video_path_hint:
                    candidate = Path(video_path_hint)
                    if not candidate.is_absolute():
                        candidate = (root / candidate).resolve()
                    if candidate.exists() and candidate.is_file():
                        video_path = candidate
                if video_path is None:
                    continue

                task_id = task_id_from_json
                if task_id is None:
                    task_id = _extract_task_id(episode_stem)
                if task_id is None:
                    continue

                uid = f"episodes_videos::{episode_stem}"
                items.append(
                    {
                        "uid": uid,
                        "display_id": episode_stem,
                        "traj_id": int(task_id),
                        "rollout_rel": ".",
                        "instruction": instruction,
                        "video_rel": video_path.relative_to(root).as_posix(),
                        "video_url": "/" + video_path.relative_to(root).as_posix(),
                        "info_rel": episode_path.relative_to(root).as_posix(),
                        "info_url": "/" + episode_path.relative_to(root).as_posix(),
                        "video_name": video_path.name,
                        "info_name": episode_path.name,
                    }
                )

    items.sort(key=lambda x: (x["rollout_rel"], int(x["traj_id"]), str(x.get("display_id") or "")))
    if limit is not None:
        return items[: max(limit, 0)]
    return items


def _atomic_dump_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _build_label_stats(labels: dict[str, dict[str, Any]], total_items: int | None = None) -> dict[str, Any]:
    counts = {"success": 0, "failure": 0, "uncertain": 0, "skip": 0}
    failure_reason_counts = {key: 0 for key in FAILURE_REASON_KEYS}
    for value in labels.values():
        if not isinstance(value, dict):
            continue
        label = str(value.get("label") or "")
        if label in counts:
            counts[label] += 1
        if label == "failure":
            failure_reason = str(value.get("failure_reason") or "")
            if failure_reason in failure_reason_counts:
                failure_reason_counts[failure_reason] += 1

    labeled = int(sum(counts.values()))
    binary_labeled = int(counts["success"] + counts["failure"])
    sr = (counts["success"] / binary_labeled) if binary_labeled > 0 else None
    failure_total = int(counts["failure"])
    failure_reason_distribution = {
        key: (failure_reason_counts[key] / failure_total) if failure_total > 0 else 0.0
        for key in FAILURE_REASON_KEYS
    }

    stats: dict[str, Any] = {
        "counts": counts,
        "labeled": labeled,
        "binary_labeled": binary_labeled,
        "sr": sr,
        "failure_reason_counts": failure_reason_counts,
        "failure_reason_distribution": failure_reason_distribution,
    }
    if total_items is not None:
        stats["total"] = int(total_items)
        stats["remaining"] = max(int(total_items) - labeled, 0)
    return stats


class AnnotationStore:
    def __init__(self, output_path: Path, dataset_root: Path, total_items: int | None = None):
        self.path = output_path.resolve()
        self.dataset_root = str(dataset_root.resolve())
        self.total_items = int(total_items) if total_items is not None else None
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {
            "version": 1,
            "dataset_root": self.dataset_root,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "labels": {},
            "sr": None,
            "summary": {},
        }
        self._load()
        self._refresh_summary_locked()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            obj = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        labels = obj.get("labels", {})
        if not isinstance(labels, dict):
            labels = {}
        clean_labels: dict[str, dict[str, Any]] = {}
        for key, value in labels.items():
            if not isinstance(value, dict):
                continue
            clean_labels[str(key)] = value
        self._data["version"] = int(obj.get("version", 1))
        self._data["dataset_root"] = str(obj.get("dataset_root", self.dataset_root))
        self._data["created_at"] = str(obj.get("created_at", _now_iso()))
        self._data["updated_at"] = str(obj.get("updated_at", _now_iso()))
        self._data["labels"] = clean_labels
        if self.total_items is None:
            raw_total = obj.get("total")
            if isinstance(raw_total, int):
                self.total_items = int(raw_total)

    def _refresh_summary_locked(self) -> None:
        labels = self._data.get("labels", {})
        if not isinstance(labels, dict):
            labels = {}
        summary = _build_label_stats(labels, total_items=self.total_items)
        self._data["summary"] = summary
        self._data["sr"] = summary.get("sr")
        if self.total_items is not None:
            self._data["total"] = int(self.total_items)

    def sync_summary(self) -> None:
        with self._lock:
            self._refresh_summary_locked()
            _atomic_dump_json(self.path, self._data)

    def get_labels_snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            labels = self._data.get("labels", {})
            return json.loads(json.dumps(labels, ensure_ascii=False))

    def upsert(
        self,
        uid: str,
        label: str | None,
        failure_reason: str,
        comment: str,
        annotator: str,
    ) -> dict[str, Any] | None:
        now = _now_iso()
        with self._lock:
            labels: dict[str, dict[str, Any]] = self._data.setdefault("labels", {})
            if not label and not failure_reason and not comment and not annotator:
                labels.pop(uid, None)
                self._data["updated_at"] = now
                self._refresh_summary_locked()
                _atomic_dump_json(self.path, self._data)
                return None

            item = labels.get(uid, {})
            item["uid"] = uid
            item["label"] = label
            item["failure_reason"] = failure_reason
            item["comment"] = comment
            item["annotator"] = annotator
            item["updated_at"] = now
            labels[uid] = item
            self._data["updated_at"] = now
            self._refresh_summary_locked()
            _atomic_dump_json(self.path, self._data)
            return json.loads(json.dumps(item, ensure_ascii=False))


def _build_summary(items: list[dict[str, Any]], labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    uid_set = {str(item["uid"]) for item in items}
    scoped_labels = {
        uid: value
        for uid, value in labels.items()
        if isinstance(value, dict) and uid in uid_set
    }
    return _build_label_stats(scoped_labels, total_items=len(items))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, state: dict[str, Any], **kwargs):
        self._state = state
        super().__init__(*args, directory=str(state["root"]), **kwargs)

    def _send_bytes(
        self,
        code: int,
        content_type: str,
        body: bytes | None,
        content_length: int | None = None,
    ) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        if content_length is None:
            content_length = 0 if body is None else len(body)
        self.send_header("Content-Length", str(content_length))
        self.end_headers()
        if body is not None:
            self.wfile.write(body)

    def _send_json(self, code: int, obj: dict[str, Any], with_body: bool = True) -> None:
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send_bytes(
            code,
            "application/json; charset=utf-8",
            payload if with_body else None,
            content_length=len(payload),
        )

    def _send_index(self, with_body: bool = True) -> None:
        payload = INDEX_HTML.encode("utf-8")
        self._send_bytes(
            200,
            "text/html; charset=utf-8",
            payload if with_body else None,
            content_length=len(payload),
        )

    def _read_json_body(self) -> dict[str, Any]:
        raw_len = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_len)
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length <= 0:
            return {}
        if length > 1_500_000:
            raise ValueError("request body too large")
        payload = self.rfile.read(length)
        try:
            data = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            raise ValueError("invalid JSON body") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _items_payload(self) -> dict[str, Any]:
        store: AnnotationStore = self._state["store"]
        labels = store.get_labels_snapshot()
        items = self._state["items"]
        summary = _build_summary(items, labels)
        return {
            "ok": True,
            "dataset_root": str(self._state["root"]),
            "output_path": str(self._state["output_path"]),
            "items": items,
            "labels": labels,
            "summary": summary,
        }

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in APP_PATHS:
            self._send_index(with_body=True)
            return
        if parsed.path == API_ITEMS_PATH:
            self._send_json(200, self._items_payload(), with_body=True)
            return
        if parsed.path == API_HEALTH_PATH:
            self._send_json(200, {"ok": True, "time": _now_iso()}, with_body=True)
            return
        super().do_GET()

    def do_HEAD(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in APP_PATHS:
            self._send_index(with_body=False)
            return
        if parsed.path in {API_ITEMS_PATH, API_HEALTH_PATH}:
            if parsed.path == API_ITEMS_PATH:
                self._send_json(200, self._items_payload(), with_body=False)
            else:
                self._send_json(200, {"ok": True, "time": _now_iso()}, with_body=False)
            return
        super().do_HEAD()

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != API_SAVE_PATH:
            self._send_json(404, {"ok": False, "error": "not found"})
            return

        try:
            body = self._read_json_body()
            uid = str(body.get("uid", "")).strip()
            if not uid:
                raise ValueError("uid is required")
            if uid not in self._state["uid_set"]:
                raise ValueError(f"unknown uid: {uid}")

            raw_label = body.get("label")
            label = str(raw_label).strip().lower() if raw_label is not None else ""
            if not label:
                label_value: str | None = None
            else:
                if label not in ALLOWED_LABELS:
                    allowed = ", ".join(sorted(ALLOWED_LABELS))
                    raise ValueError(f"label must be one of: {allowed}")
                label_value = label

            comment = str(body.get("comment", "")).strip()
            annotator = str(body.get("annotator", "")).strip()
            raw_failure_reason = body.get("failure_reason")
            failure_reason = str(raw_failure_reason).strip().lower() if raw_failure_reason is not None else ""

            if label_value == "failure":
                if not failure_reason:
                    raise ValueError("failure_reason is required when label=failure")
                if failure_reason not in ALLOWED_FAILURE_REASONS:
                    allowed = ", ".join(sorted(ALLOWED_FAILURE_REASONS))
                    raise ValueError(f"failure_reason must be one of: {allowed}")
            else:
                failure_reason = ""

            store: AnnotationStore = self._state["store"]
            saved = store.upsert(
                uid=uid,
                label=label_value,
                failure_reason=failure_reason,
                comment=comment,
                annotator=annotator,
            )
            labels = store.get_labels_snapshot()
            summary = _build_summary(self._state["items"], labels)
            self._send_json(
                200,
                {
                    "ok": True,
                    "saved": saved,
                    "summary": summary,
                },
            )
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})

    def log_message(self, fmt, *args):  # noqa: N802
        print(f"[{self.log_date_time_string()}] {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local web app for human video annotation.")
    parser.add_argument(
        "--root",
        required=True,
        help="Dataset root to scan (supports nested Rollouts_interact_gr00t/info+video).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Default: <root>/human_annotations.json",
    )
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument(
        "--rollout-dir-name",
        default="Rollouts_interact_gr00t",
        help="Name of rollout folder to search recursively.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"--root is not a directory: {root}")

    items = _collect_items(root=root, rollout_dir_name=args.rollout_dir_name, limit=args.limit)
    if not items:
        raise SystemExit(
            f"No matched info/video pairs found in: {root}\n"
            f"Expected folders like <rollout>/info/*.json and <rollout>/video/*.mp4"
        )

    output_path = Path(args.output).expanduser().resolve() if args.output else (root / "human_annotations.json")
    store = AnnotationStore(output_path=output_path, dataset_root=root, total_items=len(items))
    store.sync_summary()

    state: dict[str, Any] = {
        "root": root,
        "items": items,
        "uid_set": {item["uid"] for item in items},
        "output_path": output_path,
        "store": store,
    }

    def handler(*a, **kw):
        return Handler(*a, state=state, **kw)

    httpd = ThreadingHTTPServer((args.bind, args.port), handler)
    summary = _build_summary(items, store.get_labels_snapshot())
    print(f"Serving dataset: {root}")
    print(f"Matched items:   {len(items)}")
    print(f"Already labeled: {summary['labeled']}")
    print(f"Output file:     {output_path}")
    print(f"Open in browser: http://{args.bind}:{args.port}/__annotate__/")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
