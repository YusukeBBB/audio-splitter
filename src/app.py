"""
音声分割ツール Web UI
FastAPIサーバー + HTML UI（波形表示・再生・クロップ・削除対応）
"""

import json
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from splitter import load_audio, split_and_save

app = FastAPI()

WORK_DIR = Path(tempfile.gettempdir()) / "audio-splitter"
WORK_DIR.mkdir(exist_ok=True)


def compute_waveform_peaks(audio: np.ndarray, num_bars: int = 200) -> list[float]:
    """波形表示用にダウンサンプルしたピーク値を返す。"""
    if len(audio) == 0:
        return [0.0] * num_bars
    chunk_size = max(1, len(audio) // num_bars)
    peaks = []
    for i in range(num_bars):
        start = i * chunk_size
        end = min(start + chunk_size, len(audio))
        if start >= len(audio):
            peaks.append(0.0)
        else:
            peaks.append(float(np.max(np.abs(audio[start:end]))))
    max_peak = max(peaks) if max(peaks) > 0 else 1.0
    return [p / max_peak for p in peaks]


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/api/split")
async def split_audio(file: UploadFile):
    session_id = uuid.uuid4().hex[:8]
    session_dir = WORK_DIR / session_id
    input_dir = session_dir / "input"
    output_dir = session_dir / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    input_path = input_dir / file.filename
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    segments = split_and_save(
        input_path=str(input_path),
        output_dir=str(output_dir),
    )

    # 各トラックの波形データを生成
    tracks = []
    for seg in segments:
        fname = f"{input_path.stem}_track{seg.index + 1:02d}.wav"
        wav_path = output_dir / fname
        audio, sr = sf.read(str(wav_path), dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        peaks = compute_waveform_peaks(audio)
        tracks.append({
            "index": seg.index + 1,
            "filename": fname,
            "start": seg.start_sec,
            "end": seg.end_sec,
            "duration": seg.duration_sec,
            "waveform": peaks,
        })

    return {"session_id": session_id, "tracks": tracks}


@app.get("/api/audio/{session_id}/{filename}")
async def serve_audio(session_id: str, filename: str):
    """個別トラックの音声を返す。"""
    file_path = WORK_DIR / session_id / "output" / filename
    if not file_path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(file_path, media_type="audio/wav")


@app.post("/api/crop/{session_id}/{filename}")
async def crop_track(session_id: str, filename: str, crop_start: float = 0, crop_end: float = -1):
    """トラックをクロップして上書き保存。波形を返す。"""
    file_path = WORK_DIR / session_id / "output" / filename
    if not file_path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)

    audio, sr = sf.read(str(file_path), dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    start_sample = int(crop_start * sr)
    end_sample = int(crop_end * sr) if crop_end >= 0 else len(audio)
    start_sample = max(0, min(start_sample, len(audio)))
    end_sample = max(start_sample, min(end_sample, len(audio)))

    cropped = audio[start_sample:end_sample]
    sf.write(str(file_path), cropped, sr)

    peaks = compute_waveform_peaks(cropped)
    duration = len(cropped) / sr
    return {"duration": duration, "waveform": peaks}


@app.delete("/api/track/{session_id}/{filename}")
async def delete_track(session_id: str, filename: str):
    """トラックを削除。"""
    file_path = WORK_DIR / session_id / "output" / filename
    if file_path.exists():
        os.unlink(file_path)
    return {"ok": True}


@app.post("/api/download-zip/{session_id}")
async def download_zip(session_id: str, request: Request):
    """残っているトラックをZIPにまとめてダウンロード。namesでリネーム。"""
    body = await request.json()
    names = body.get("names", {})
    output_dir = WORK_DIR / session_id / "output"
    zip_path = WORK_DIR / session_id / "tracks.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for wav_file in sorted(output_dir.glob("*.wav")):
            arc_name = names.get(wav_file.name)
            if arc_name:
                # 拡張子を保持
                if not arc_name.endswith(".wav"):
                    arc_name += ".wav"
            else:
                arc_name = wav_file.name
            zf.write(wav_file, arc_name)

    return FileResponse(zip_path, filename="tracks.zip", media_type="application/zip")


HTML_PAGE = """\
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Audio Splitter</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0f0f0f;
    color: #e0e0e0;
    min-height: 100vh;
    display: flex;
    justify-content: center;
    padding: 40px 20px;
  }
  .container { max-width: 800px; width: 100%; }
  h1 { font-size: 1.5rem; margin-bottom: 24px; color: #fff; }

  /* ドロップゾーン */
  .dropzone {
    border: 2px dashed #444; border-radius: 12px;
    padding: 48px 24px; text-align: center; cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
  }
  .dropzone:hover, .dropzone.dragover {
    border-color: #6c8cff; background: rgba(108, 140, 255, 0.05);
  }
  .dropzone p { color: #888; margin-bottom: 12px; }
  .dropzone .browse { color: #6c8cff; text-decoration: underline; cursor: pointer; }
  .dropzone .filename { margin-top: 12px; color: #aaa; font-size: 0.9rem; }

  /* プログレス */
  .progress { margin-top: 24px; display: none; }
  .progress .bar-bg { background: #222; border-radius: 8px; overflow: hidden; height: 6px; }
  .progress .bar { height: 100%; background: #6c8cff; width: 0%; transition: width 0.3s; }
  .progress .status { margin-top: 8px; font-size: 0.85rem; color: #888; }

  /* 結果エリア */
  .result { margin-top: 24px; display: none; }
  .result h2 { font-size: 1.1rem; color: #fff; margin-bottom: 12px; }

  /* トラックカード */
  .track-card {
    background: #1a1a1a; border-radius: 10px; padding: 16px;
    margin-bottom: 12px; position: relative;
  }
  .track-card.deleted { display: none; }
  .track-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 10px;
  }
  .track-name {
    background: transparent; border: 1px solid #333; border-radius: 4px;
    color: #6c8cff; font-weight: 600; font-size: 0.95rem;
    padding: 2px 6px; outline: none; max-width: 50%;
    font-family: inherit;
  }
  .track-name:hover { border-color: #444; }
  .track-name:focus { border-color: #6c8cff; background: #222; }
  .track-header .info { color: #888; font-size: 0.85rem; }
  .track-actions { display: flex; gap: 8px; }
  .track-actions button {
    background: #2a2a2a; border: 1px solid #444; color: #ccc;
    border-radius: 6px; padding: 4px 10px; font-size: 0.8rem;
    cursor: pointer; transition: background 0.15s;
  }
  .track-actions button:hover { background: #3a3a3a; }
  .track-actions button.danger:hover { background: #5a2020; border-color: #a44; color: #f88; }

  /* 波形 */
  .waveform-container {
    position: relative; height: 64px; margin-bottom: 8px;
    cursor: pointer; user-select: none;
  }
  .waveform-container canvas { width: 100%; height: 100%; display: block; border-radius: 6px; }
  /* クロップハンドル */
  .crop-overlay {
    position: absolute; top: 0; height: 100%; display: none;
  }
  .crop-overlay.active { display: block; }
  .crop-shade {
    position: absolute; top: 0; height: 100%; background: rgba(0,0,0,0.6);
  }
  .crop-handle {
    position: absolute; top: 0; width: 4px; height: 100%;
    background: #ff6c6c; cursor: ew-resize; z-index: 2;
  }
  .crop-handle::after {
    content: ''; position: absolute; top: 50%; transform: translateY(-50%);
    width: 12px; height: 24px; background: #ff6c6c; border-radius: 4px;
    left: -4px;
  }

  /* 再生バー */
  .playback-bar {
    position: absolute; top: 0; width: 2px; height: 100%;
    background: #fff; pointer-events: none; z-index: 1; display: none;
  }

  /* ボタン */
  .btn {
    display: inline-block; padding: 10px 24px;
    border: none; border-radius: 8px; font-size: 0.95rem;
    cursor: pointer; text-decoration: none; transition: background 0.2s;
  }
  .btn-primary { background: #6c8cff; color: #fff; }
  .btn-primary:hover { background: #5a7ae6; }
  .btn-primary:disabled { background: #444; cursor: not-allowed; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid #fff4; border-top-color: #fff; border-radius: 50%; animation: spin 0.6s linear infinite; vertical-align: middle; margin-right: 6px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .btn-secondary { background: #2a2a2a; color: #ccc; border: 1px solid #444; }
  .btn-secondary:hover { background: #3a3a3a; }
  .bottom-actions { margin-top: 20px; display: flex; gap: 12px; flex-wrap: wrap; }

  /* クロップ情報 */
  .crop-info {
    font-size: 0.8rem; color: #aaa; margin-top: 4px; display: none;
  }
  .crop-info.active { display: flex; gap: 12px; align-items: center; }
  .crop-info button {
    background: #6c8cff; border: none; color: #fff;
    border-radius: 4px; padding: 3px 10px; font-size: 0.78rem;
    cursor: pointer;
  }
  .crop-info button.cancel { background: #444; }
</style>
</head>
<body>
<div class="container">
  <h1>Audio Splitter</h1>

  <div class="dropzone" id="dropzone">
    <p>音声ファイルをドラッグ&ドロップ</p>
    <span class="browse">またはファイルを選択</span>
    <input type="file" id="fileInput" accept=".wav,.m4a,.mp3,.flac,.ogg,.aac" hidden>
    <div class="filename" id="filename"></div>
  </div>

  <div class="progress" id="progress">
    <div class="bar-bg"><div class="bar" id="bar"></div></div>
    <div class="status" id="status">アップロード中...</div>
  </div>

  <div class="result" id="result">
    <h2>分割結果</h2>
    <div id="trackList"></div>
    <div class="bottom-actions">
      <button class="btn btn-primary" id="downloadBtn" onclick="downloadZip()">ZIPダウンロード</button>
    </div>
  </div>
</div>

<script>
// --- State ---
let sessionId = null;
let tracks = []; // {index, filename, duration, waveform, audio, cropStart, cropEnd, deleted}

// --- DOM ---
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const filenameEl = document.getElementById('filename');
const progressEl = document.getElementById('progress');
const bar = document.getElementById('bar');
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');
const trackListEl = document.getElementById('trackList');
const downloadBtn = document.getElementById('downloadBtn');

// --- Upload ---
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', e => {
  e.preventDefault(); dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
dropzone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

function fmt(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return m + ':' + String(s).padStart(2, '0');
}

function handleFile(file) {
  filenameEl.textContent = file.name;
  resultEl.style.display = 'none';
  progressEl.style.display = 'block';
  bar.style.width = '0%';
  statusEl.textContent = 'アップロード中...';

  const fd = new FormData();
  fd.append('file', file);
  const xhr = new XMLHttpRequest();

  xhr.upload.addEventListener('progress', e => {
    if (e.lengthComputable) bar.style.width = (e.loaded / e.total * 50) + '%';
  });
  xhr.upload.addEventListener('load', () => {
    bar.style.width = '60%';
    statusEl.textContent = '分割処理中...';
    const iv = setInterval(() => {
      const w = parseFloat(bar.style.width);
      if (w < 95) bar.style.width = (w + 1) + '%'; else clearInterval(iv);
    }, 500);
  });
  xhr.addEventListener('load', () => {
    if (xhr.status === 200) {
      bar.style.width = '100%';
      statusEl.textContent = '完了!';
      showResult(JSON.parse(xhr.responseText));
    } else { statusEl.textContent = 'エラーが発生しました'; }
  });
  xhr.addEventListener('error', () => { statusEl.textContent = 'エラーが発生しました'; });
  xhr.open('POST', '/api/split');
  xhr.send(fd);
}

// --- Result ---
function showResult(data) {
  sessionId = data.session_id;
  tracks = data.tracks.map(t => ({
    ...t, audio: null, playing: false, cropMode: false,
    cropStart: 0, cropEnd: t.duration, deleted: false, ver: 1,
  }));
  renderTracks();
  resultEl.style.display = 'block';
  // DOM配置後に波形を描画
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      for (const t of tracks) { if (!t.deleted) drawWaveform(t); }
    });
  });
}

function renderTracks() {
  trackListEl.innerHTML = '';
  for (const t of tracks) {
    if (t.deleted) continue;
    const card = document.createElement('div');
    card.className = 'track-card';
    card.id = 'card-' + t.index;
    card.innerHTML = `
      <div class="track-header">
        <input class="track-name" id="name-${t.index}" value="Track ${t.index}" spellcheck="false">
        <span class="info" id="info-${t.index}">${fmt(t.duration)} (${fmt(t.start)} - ${fmt(t.end)})</span>
      </div>
      <div class="waveform-container" id="wc-${t.index}">
        <canvas id="cv-${t.index}"></canvas>
        <div class="playback-bar" id="pb-${t.index}"></div>
        <div class="crop-overlay" id="co-${t.index}">
          <div class="crop-shade" id="shade-l-${t.index}" style="left:0;"></div>
          <div class="crop-handle" id="handle-l-${t.index}"></div>
          <div class="crop-handle" id="handle-r-${t.index}"></div>
          <div class="crop-shade" id="shade-r-${t.index}"></div>
        </div>
      </div>
      <div class="crop-info" id="ci-${t.index}">
        <span id="crop-range-${t.index}"></span>
        <button onclick="applyCrop(${t.index})">適用</button>
        <button class="cancel" onclick="cancelCrop(${t.index})">キャンセル</button>
      </div>
      <div class="track-actions">
        <button id="playbtn-${t.index}" onclick="togglePlay(${t.index})">&#9654; 再生</button>
        <button onclick="startCrop(${t.index})">&#9986; クロップ</button>
        <button class="danger" onclick="deleteTrack(${t.index})">&#10005; 削除</button>
      </div>
    `;
    trackListEl.appendChild(card);
  }
}

// --- Waveform ---
function drawWaveform(t) {
  const cv = document.getElementById('cv-' + t.index);
  const rect = cv.parentElement.getBoundingClientRect();
  cv.width = rect.width * devicePixelRatio;
  cv.height = rect.height * devicePixelRatio;
  const ctx = cv.getContext('2d');
  ctx.scale(devicePixelRatio, devicePixelRatio);
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);

  const bars = t.waveform;
  const barW = w / bars.length;
  const half = h / 2;

  for (let i = 0; i < bars.length; i++) {
    const val = bars[i] * half * 0.9;
    ctx.fillStyle = '#6c8cff';
    ctx.fillRect(i * barW, half - val, Math.max(barW - 1, 1), val * 2);
  }
}

function audioUrl(t) {
  return '/api/audio/' + sessionId + '/' + t.filename + '?v=' + t.ver;
}

// --- Waveform click to seek ---
document.addEventListener('click', e => {
  const wc = e.target.closest('.waveform-container');
  if (!wc) return;
  const idx = parseInt(wc.id.replace('wc-', ''));
  const t = tracks.find(x => x.index === idx);
  if (!t || t.cropMode) return;

  const rect = wc.getBoundingClientRect();
  const pct = (e.clientX - rect.left) / rect.width;
  const seekTime = pct * t.duration;

  // 既に再生中ならシーク
  if (t.playing && t.audio) {
    t.audio.currentTime = seekTime;
    return;
  }

  // 再生していなければその位置から再生開始
  stopAll();
  const audio = new Audio(audioUrl(t));
  t.audio = audio;
  t.playing = true;
  markPlaying(idx);

  const pbBar = document.getElementById('pb-' + idx);
  pbBar.style.display = 'block';
  pbBar.style.left = (pct * wc.clientWidth) + 'px';

  audio.addEventListener('loadedmetadata', () => { audio.currentTime = seekTime; });
  audio.addEventListener('timeupdate', () => {
    pbBar.style.left = (audio.currentTime / audio.duration * wc.clientWidth) + 'px';
  });
  audio.addEventListener('ended', () => {
    t.playing = false; t.audio = null; pbBar.style.display = 'none';
  });
  audio.play();
});

function stopAll() {
  for (const ot of tracks) {
    if (ot.playing && ot.audio) { ot.audio.pause(); ot.audio = null; ot.playing = false; }
    const pb = document.getElementById('pb-' + ot.index);
    if (pb) pb.style.display = 'none';
    const btn = document.getElementById('playbtn-' + ot.index);
    if (btn) btn.innerHTML = '&#9654; 再生';
  }
}

function markPlaying(idx) {
  const btn = document.getElementById('playbtn-' + idx);
  if (btn) btn.innerHTML = '&#9632; 停止';
}

// --- Playback ---
function togglePlay(idx) {
  const t = tracks.find(x => x.index === idx);
  if (!t) return;

  if (t.playing) {
    stopAll();
    return;
  }

  stopAll();

  const audio = new Audio(audioUrl(t));
  t.audio = audio;
  t.playing = true;
  markPlaying(idx);

  const pbBar = document.getElementById('pb-' + idx);
  const wc = document.getElementById('wc-' + idx);
  pbBar.style.display = 'block';

  audio.addEventListener('timeupdate', () => {
    const pct = audio.currentTime / audio.duration;
    pbBar.style.left = (pct * wc.clientWidth) + 'px';
  });
  audio.addEventListener('ended', () => {
    t.playing = false; t.audio = null;
    pbBar.style.display = 'none';
  });
  audio.play();
}

// --- Crop ---
function startCrop(idx) {
  const t = tracks.find(x => x.index === idx);
  if (!t) return;
  t.cropMode = true;
  t.cropStart = 0;
  t.cropEnd = t.duration;

  const overlay = document.getElementById('co-' + idx);
  overlay.classList.add('active');
  const info = document.getElementById('ci-' + idx);
  info.classList.add('active');

  const wc = document.getElementById('wc-' + idx);
  const wcW = wc.clientWidth;

  const handleL = document.getElementById('handle-l-' + idx);
  const handleR = document.getElementById('handle-r-' + idx);
  const shadeL = document.getElementById('shade-l-' + idx);
  const shadeR = document.getElementById('shade-r-' + idx);

  handleL.style.left = '0px';
  handleR.style.left = (wcW - 4) + 'px';
  shadeL.style.width = '0px';
  shadeR.style.left = wcW + 'px';
  shadeR.style.width = '0px';

  updateCropInfo(idx);
  setupCropDrag(idx);
}

function setupCropDrag(idx) {
  const t = tracks.find(x => x.index === idx);
  const wc = document.getElementById('wc-' + idx);
  const handleL = document.getElementById('handle-l-' + idx);
  const handleR = document.getElementById('handle-r-' + idx);

  function makeDrag(handle, side) {
    handle.onmousedown = (e) => {
      e.preventDefault();
      const wcRect = wc.getBoundingClientRect();
      const wcW = wcRect.width;

      function onMove(e2) {
        let x = e2.clientX - wcRect.left;
        x = Math.max(0, Math.min(x, wcW));
        const pct = x / wcW;

        if (side === 'left') {
          const rightPct = t.cropEnd / t.duration;
          if (pct < rightPct) {
            t.cropStart = pct * t.duration;
            handle.style.left = x + 'px';
            document.getElementById('shade-l-' + idx).style.width = x + 'px';
          }
        } else {
          const leftPct = t.cropStart / t.duration;
          if (pct > leftPct) {
            t.cropEnd = pct * t.duration;
            handle.style.left = x + 'px';
            const shadeR = document.getElementById('shade-r-' + idx);
            shadeR.style.left = (x + 4) + 'px';
            shadeR.style.width = (wcW - x - 4) + 'px';
          }
        }
        updateCropInfo(idx);
      }
      function onUp() {
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
      }
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    };
  }
  makeDrag(handleL, 'left');
  makeDrag(handleR, 'right');
}

function updateCropInfo(idx) {
  const t = tracks.find(x => x.index === idx);
  const el = document.getElementById('crop-range-' + idx);
  el.textContent = fmt(t.cropStart) + ' - ' + fmt(t.cropEnd) +
    ' (' + fmt(t.cropEnd - t.cropStart) + ')';
}

async function applyCrop(idx) {
  const t = tracks.find(x => x.index === idx);
  if (!t) return;

  const url = '/api/crop/' + sessionId + '/' + t.filename +
    '?crop_start=' + t.cropStart + '&crop_end=' + t.cropEnd;
  const res = await fetch(url, { method: 'POST' });
  const data = await res.json();

  t.duration = data.duration;
  t.waveform = data.waveform;
  t.ver++;
  t.cropMode = false;
  t.cropStart = 0;
  t.cropEnd = t.duration;

  document.getElementById('co-' + idx).classList.remove('active');
  document.getElementById('ci-' + idx).classList.remove('active');
  document.getElementById('info-' + idx).textContent = fmt(t.duration);

  drawWaveform(t);
}

function cancelCrop(idx) {
  const t = tracks.find(x => x.index === idx);
  if (t) { t.cropMode = false; t.cropStart = 0; t.cropEnd = t.duration; }
  document.getElementById('co-' + idx).classList.remove('active');
  document.getElementById('ci-' + idx).classList.remove('active');
}

// --- Delete ---
async function deleteTrack(idx) {
  const t = tracks.find(x => x.index === idx);
  if (!t) return;
  if (t.audio) { t.audio.pause(); t.audio = null; }
  t.deleted = true;
  document.getElementById('card-' + idx).classList.add('deleted');
  await fetch('/api/track/' + sessionId + '/' + t.filename, { method: 'DELETE' });
}

// --- Download ZIP with names ---
async function downloadZip() {
  const btn = document.getElementById('downloadBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>準備中...';

  const names = {};
  for (const t of tracks) {
    if (t.deleted) continue;
    const nameInput = document.getElementById('name-' + t.index);
    if (nameInput && nameInput.value.trim()) {
      names[t.filename] = nameInput.value.trim();
    }
  }
  try {
    const res = await fetch('/api/download-zip/' + sessionId, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ names }),
    });
    const blob = await res.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'tracks.zip';
    a.click();
    URL.revokeObjectURL(a.href);
  } finally {
    btn.disabled = false;
    btn.textContent = 'ZIPダウンロード';
  }
}
</script>
</body>
</html>
"""
