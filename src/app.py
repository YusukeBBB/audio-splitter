"""
音声分割ツール Web UI
FastAPIサーバー + HTML UI（波形表示・再生・クロップ・削除対応）
"""

import json
import os
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.responses import RedirectResponse, StreamingResponse

from splitter import load_audio, split_and_save

# .env ファイルから環境変数を読み込み
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

app = FastAPI()

WORK_DIR = Path(tempfile.gettempdir()) / "audio-splitter"
WORK_DIR.mkdir(exist_ok=True)

# Google OAuth config
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
]

# In-memory token storage: {session_id: {"access_token": ..., ...}}
oauth_tokens: dict[str, dict] = {}


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


@app.post("/api/split-track/{session_id}/{filename}")
async def split_track_endpoint(session_id: str, filename: str, split_at: float = 0):
    """トラックを指定位置で2つに分割する。"""
    file_path = WORK_DIR / session_id / "output" / filename
    if not file_path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)

    audio, sr = sf.read(str(file_path), dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    split_sample = int(split_at * sr)
    split_sample = max(1, min(split_sample, len(audio) - 1))

    part1 = audio[:split_sample]
    part2 = audio[split_sample:]

    # 新ファイル名を生成（衝突回避）
    stem = file_path.stem
    suffix = "b"
    new_filename = f"{stem}{suffix}.wav"
    new_path = file_path.parent / new_filename
    while new_path.exists():
        suffix += "b"
        new_filename = f"{stem}{suffix}.wav"
        new_path = file_path.parent / new_filename

    sf.write(str(file_path), part1, sr)
    sf.write(str(new_path), part2, sr)

    return {
        "track1": {
            "filename": filename,
            "duration": len(part1) / sr,
            "waveform": compute_waveform_peaks(part1),
        },
        "track2": {
            "filename": new_filename,
            "duration": len(part2) / sr,
            "waveform": compute_waveform_peaks(part2),
        },
    }


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


# ============================================================
# Google Drive 連携
# ============================================================

def _get_redirect_uri() -> str:
    base = os.environ.get("BASE_URL", "http://localhost:8000")
    return base.rstrip("/") + "/auth/callback"


@app.get("/api/google-config")
async def google_config():
    """Google OAuth/Picker用の設定を返す。"""
    return {"client_id": GOOGLE_CLIENT_ID, "api_key": GOOGLE_API_KEY}


@app.get("/auth/google")
async def auth_google(session_id: str):
    """OAuth認証を開始。ポップアップウィンドウで開かれる想定。"""
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": _get_redirect_uri(),
        "response_type": "code",
        "scope": " ".join(GOOGLE_SCOPES),
        "access_type": "offline",
        "state": session_id,
        "prompt": "consent",
    }
    auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
    return RedirectResponse(auth_url)


@app.get("/auth/callback")
async def auth_callback(code: str, state: str):
    """OAuthコールバック。トークンを保存してポップアップを閉じる。"""
    from google_auth_oauthlib.flow import Flow

    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/v2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        },
        scopes=GOOGLE_SCOPES,
        redirect_uri=_get_redirect_uri(),
    )
    flow.fetch_token(code=code)
    creds = flow.credentials

    oauth_tokens[state] = {
        "access_token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
    }

    return HTMLResponse(
        "<html><body><script>"
        "window.opener.postMessage({type:'google-auth-success'},'*');"
        "window.close();"
        "</script><p>認証完了。このウィンドウは自動的に閉じます。</p></body></html>"
    )


@app.get("/auth/token/{session_id}")
async def get_token(session_id: str):
    """フロントエンドがPicker用のアクセストークンを取得する。"""
    token_data = oauth_tokens.get(session_id)
    if not token_data:
        return JSONResponse({"authenticated": False}, status_code=401)
    return {"authenticated": True, "access_token": token_data["access_token"]}


@app.post("/api/upload-drive/{session_id}")
async def upload_to_drive(session_id: str, request: Request):
    """WAV→MP3変換してGoogle Driveにアップロード。SSEでプログレスを返す。"""
    body = await request.json()
    parent_folder_id = body["parent_folder_id"]
    subfolder_name = body["subfolder_name"]
    names = body.get("names", {})

    token_data = oauth_tokens.get(session_id)
    if not token_data:
        return JSONResponse({"error": "not authenticated"}, status_code=401)

    output_dir = WORK_DIR / session_id / "output"
    wav_files = sorted(output_dir.glob("*.wav"))

    if not wav_files:
        return JSONResponse({"error": "no tracks found"}, status_code=404)

    async def event_stream():
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        try:
            creds = Credentials(
                token=token_data["access_token"],
                refresh_token=token_data.get("refresh_token"),
                token_uri=token_data["token_uri"],
                client_id=token_data["client_id"],
                client_secret=token_data["client_secret"],
            )
            service = build("drive", "v3", credentials=creds)

            # サブフォルダを作成
            folder_meta = {
                "name": subfolder_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent_folder_id],
            }
            folder = service.files().create(body=folder_meta, fields="id").execute()
            subfolder_id = folder["id"]

            total = len(wav_files)
            yield f"data: {json.dumps({'type': 'start', 'total': total})}\n\n"

            for i, wav_path in enumerate(wav_files):
                custom_name = names.get(wav_path.name, wav_path.stem)
                if custom_name.endswith(".wav"):
                    custom_name = custom_name[:-4]
                mp3_name = custom_name + ".mp3"

                # WAV → MP3 変換
                mp3_path = wav_path.parent / (wav_path.stem + ".mp3")
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", str(wav_path),
                        "-c:a", "libmp3lame", "-b:a", "192k",
                        str(mp3_path),
                    ],
                    capture_output=True,
                    check=True,
                )

                # Google Drive にアップロード
                file_meta = {"name": mp3_name, "parents": [subfolder_id]}
                media = MediaFileUpload(str(mp3_path), mimetype="audio/mpeg", resumable=True)
                service.files().create(body=file_meta, media_body=media, fields="id").execute()

                mp3_path.unlink(missing_ok=True)

                yield f"data: {json.dumps({'type': 'progress', 'current': i + 1, 'total': total, 'name': mp3_name})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


HTML_PAGE = """\
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Audio Splitter</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap" rel="stylesheet">
<style>
  /* MD3 Design Tokens (Light Theme - Blue) */
  :root {
    --md-sys-color-primary: #1565C0;
    --md-sys-color-on-primary: #FFFFFF;
    --md-sys-color-primary-container: #D4E3FF;
    --md-sys-color-on-primary-container: #001C3A;
    --md-sys-color-secondary: #555F71;
    --md-sys-color-on-secondary: #FFFFFF;
    --md-sys-color-secondary-container: #D9E3F8;
    --md-sys-color-on-secondary-container: #121C2B;
    --md-sys-color-tertiary: #6E5676;
    --md-sys-color-on-tertiary: #FFFFFF;
    --md-sys-color-tertiary-container: #F8D8FF;
    --md-sys-color-on-tertiary-container: #271430;
    --md-sys-color-error: #BA1A1A;
    --md-sys-color-on-error: #FFFFFF;
    --md-sys-color-error-container: #FFDAD6;
    --md-sys-color-on-error-container: #410002;
    --md-sys-color-surface: #FAFCFF;
    --md-sys-color-on-surface: #1A1C1E;
    --md-sys-color-on-surface-variant: #43474E;
    --md-sys-color-surface-container-lowest: #FFFFFF;
    --md-sys-color-surface-container-low: #F4F6F9;
    --md-sys-color-surface-container: #EEF0F4;
    --md-sys-color-surface-container-high: #E8EAEE;
    --md-sys-color-surface-container-highest: #E2E4E8;
    --md-sys-color-outline: #73777F;
    --md-sys-color-outline-variant: #C3C6CF;
    --md-sys-color-inverse-surface: #2F3033;
    --md-sys-color-inverse-on-surface: #F1F0F4;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: "Roboto", "Noto Sans JP", sans-serif;
    background: var(--md-sys-color-surface);
    color: var(--md-sys-color-on-surface);
    min-height: 100vh;
    display: flex;
    justify-content: center;
    padding: 40px 20px;
  }
  .container { max-width: 800px; width: 100%; }
  h1 {
    font-size: 24px; line-height: 32px; font-weight: 400;
    margin-bottom: 24px; color: var(--md-sys-color-on-surface);
  }

  /* ドロップゾーン */
  .dropzone {
    border: 2px dashed var(--md-sys-color-outline);
    border-radius: 16px; padding: 48px 24px; text-align: center;
    cursor: pointer;
    background: var(--md-sys-color-surface-container-low);
    transition: border-color 0.2s cubic-bezier(0.2,0,0,1), background 0.2s cubic-bezier(0.2,0,0,1);
  }
  .dropzone:hover, .dropzone.dragover {
    border-color: var(--md-sys-color-primary);
    background: var(--md-sys-color-surface-container);
  }
  .dropzone p {
    color: var(--md-sys-color-on-surface-variant);
    font-size: 14px; line-height: 20px; margin-bottom: 12px;
  }
  .dropzone .browse {
    color: var(--md-sys-color-primary);
    text-decoration: underline; cursor: pointer;
    font-weight: 500; font-size: 14px;
  }
  .dropzone .filename {
    margin-top: 12px; color: var(--md-sys-color-on-surface-variant);
    font-size: 12px; line-height: 16px;
  }

  /* プログレス */
  .progress { margin-top: 24px; display: none; }
  .progress .bar-bg {
    background: var(--md-sys-color-surface-container-highest);
    border-radius: 9999px; overflow: hidden; height: 4px;
  }
  .progress .bar {
    height: 100%; background: var(--md-sys-color-primary);
    width: 0%; transition: width 0.3s cubic-bezier(0.2,0,0,1);
    border-radius: 9999px;
  }
  .progress .status {
    margin-top: 8px; font-size: 12px; line-height: 16px;
    color: var(--md-sys-color-on-surface-variant);
  }

  /* 結果エリア */
  .result { margin-top: 24px; display: none; }
  .result h2 {
    font-size: 22px; line-height: 28px; font-weight: 400;
    color: var(--md-sys-color-on-surface); margin-bottom: 16px;
  }

  /* トラックカード (Elevated Card) */
  .track-card {
    background: var(--md-sys-color-surface-container);
    border-radius: 12px; padding: 16px;
    margin-bottom: 12px; position: relative;
    box-shadow: 0 1px 2px rgba(0,0,0,.3), 0 1px 3px 1px rgba(0,0,0,.15);
    transition: box-shadow 0.2s cubic-bezier(0.2,0,0,1);
  }
  .track-card:hover {
    box-shadow: 0 1px 2px rgba(0,0,0,.3), 0 2px 6px 2px rgba(0,0,0,.15);
  }
  .track-card.deleted { display: none; }
  .track-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 10px;
  }
  .track-name {
    background: transparent;
    border: 1px solid var(--md-sys-color-outline-variant);
    border-radius: 4px;
    color: var(--md-sys-color-primary);
    font-weight: 500; font-size: 16px; line-height: 24px;
    padding: 2px 8px; outline: none; max-width: 50%;
    font-family: inherit;
    transition: border-color 0.2s, background 0.2s;
  }
  .track-name:hover { border-color: var(--md-sys-color-outline); }
  .track-name:focus {
    border-color: var(--md-sys-color-primary);
    border-width: 2px; padding: 1px 7px;
    background: var(--md-sys-color-surface-container-high);
  }
  .track-header .info {
    color: var(--md-sys-color-on-surface-variant);
    font-size: 12px; line-height: 16px;
  }

  /* トラックアクションボタン (Tonal / Outlined style) */
  .track-actions { display: flex; gap: 8px; }
  .track-actions button {
    background: var(--md-sys-color-surface-container-high);
    border: 1px solid var(--md-sys-color-outline-variant);
    color: var(--md-sys-color-on-surface);
    border-radius: 9999px; padding: 6px 16px;
    font-size: 12px; line-height: 16px; font-weight: 500;
    font-family: inherit;
    cursor: pointer;
    transition: background 0.2s cubic-bezier(0.2,0,0,1);
    position: relative;
  }
  .track-actions button:hover {
    background: var(--md-sys-color-surface-container-highest);
  }
  .track-actions button:focus-visible {
    outline: 2px solid var(--md-sys-color-primary);
    outline-offset: 2px;
  }
  .track-actions button.danger {
    color: var(--md-sys-color-error);
    border-color: var(--md-sys-color-error-container);
  }
  .track-actions button.danger:hover {
    background: var(--md-sys-color-error-container);
    color: var(--md-sys-color-on-error-container);
  }

  /* 波形 */
  .waveform-container {
    position: relative; height: 64px; margin-bottom: 8px;
    cursor: pointer; user-select: none;
    border-radius: 8px;
    background: var(--md-sys-color-surface-container-lowest);
  }
  .waveform-container canvas {
    width: 100%; height: 100%; display: block;
    border-radius: 8px;
  }
  /* クロップハンドル */
  .crop-overlay {
    position: absolute; top: 0; height: 100%; display: none;
  }
  .crop-overlay.active { display: block; }
  .crop-shade {
    position: absolute; top: 0; height: 100%;
    background: rgba(255,251,254,0.65);
  }
  .crop-handle {
    position: absolute; top: 0; width: 4px; height: 100%;
    background: var(--md-sys-color-tertiary); cursor: ew-resize; z-index: 2;
  }
  .crop-handle::after {
    content: ''; position: absolute; top: 50%; transform: translateY(-50%);
    width: 12px; height: 24px;
    background: var(--md-sys-color-tertiary);
    border-radius: 9999px; left: -4px;
  }

  /* 再生バー */
  .playback-bar {
    position: absolute; top: 0; width: 2px; height: 100%;
    background: var(--md-sys-color-on-surface);
    pointer-events: none; z-index: 1; display: none;
  }

  /* Filled Button (Primary) */
  .btn {
    display: inline-flex; align-items: center; justify-content: center;
    height: 40px; padding: 0 24px; gap: 8px;
    border: none; border-radius: 9999px;
    font-family: inherit; font-size: 14px; font-weight: 500;
    line-height: 20px; letter-spacing: 0.1px;
    cursor: pointer; text-decoration: none;
    transition: box-shadow 0.2s cubic-bezier(0.2,0,0,1), background 0.2s;
    position: relative; overflow: hidden;
  }
  .btn::before {
    content: ""; position: absolute; inset: 0;
    border-radius: inherit; opacity: 0; transition: opacity 0.2s;
  }
  .btn:hover::before { opacity: 0.08; }
  .btn:active::before { opacity: 0.12; }
  .btn:focus-visible {
    outline: 2px solid var(--md-sys-color-primary);
    outline-offset: 2px;
  }
  .btn-primary {
    background: var(--md-sys-color-primary);
    color: var(--md-sys-color-on-primary);
  }
  .btn-primary::before { background: var(--md-sys-color-on-primary); }
  .btn-primary:hover {
    box-shadow: 0 1px 2px rgba(0,0,0,.3), 0 1px 3px 1px rgba(0,0,0,.15);
  }
  .btn-primary:disabled {
    background: rgba(28,27,31,0.12);
    color: rgba(28,27,31,0.38);
    box-shadow: none; cursor: not-allowed;
  }
  .btn-primary:disabled::before { display: none; }
  .spinner {
    display: inline-block; width: 14px; height: 14px;
    border: 2px solid rgba(255,255,255,0.4);
    border-top-color: var(--md-sys-color-on-primary);
    border-radius: 50%; animation: spin 0.6s linear infinite;
    vertical-align: middle; margin-right: 6px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .btn-secondary {
    background: var(--md-sys-color-surface-container-high);
    color: var(--md-sys-color-on-surface);
    border: 1px solid var(--md-sys-color-outline);
  }
  .btn-secondary::before { background: var(--md-sys-color-on-surface); }
  .btn-secondary:hover {
    background: var(--md-sys-color-surface-container-highest);
  }
  .bottom-actions { margin-top: 20px; display: flex; gap: 12px; flex-wrap: wrap; }

  /* クロップ・カット情報 */
  .crop-info, .cut-info {
    font-size: 12px; line-height: 16px;
    color: var(--md-sys-color-on-surface-variant);
    margin-top: 4px; display: none;
  }
  .crop-info.active, .cut-info.active { display: flex; gap: 12px; align-items: center; }
  .crop-info button, .cut-info button {
    background: var(--md-sys-color-primary);
    color: var(--md-sys-color-on-primary);
    border: none; border-radius: 9999px;
    padding: 4px 12px; font-size: 12px; font-weight: 500;
    cursor: pointer; font-family: inherit;
    transition: box-shadow 0.2s;
  }
  .crop-info button:hover, .cut-info button:hover {
    box-shadow: 0 1px 2px rgba(0,0,0,.3), 0 1px 3px 1px rgba(0,0,0,.15);
  }
  .crop-info button.cancel, .cut-info button.cancel {
    background: var(--md-sys-color-surface-container-highest);
    color: var(--md-sys-color-on-surface);
  }

  /* カットライン */
  .cut-line {
    position: absolute; top: 0; width: 3px; height: 100%;
    background: var(--md-sys-color-tertiary);
    cursor: ew-resize; z-index: 3; display: none;
  }

  /* Google Drive Modal */
  .modal-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,0.4);
    display: none; justify-content: center; align-items: center;
    z-index: 100;
  }
  .modal-overlay.active { display: flex; }
  .modal {
    background: var(--md-sys-color-surface-container-low);
    border-radius: 28px; padding: 24px; max-width: 560px; width: 90%;
    max-height: 80vh; overflow-y: auto;
    box-shadow: 0 8px 12px 6px rgba(0,0,0,.15), 0 4px 4px rgba(0,0,0,.3);
  }
  .modal h3 {
    font-size: 24px; line-height: 32px; font-weight: 400;
    margin-bottom: 16px; color: var(--md-sys-color-on-surface);
  }
  .modal-track-list { margin-bottom: 16px; }
  .modal-track-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 0; border-bottom: 1px solid var(--md-sys-color-outline-variant);
    font-size: 14px; color: var(--md-sys-color-on-surface);
  }
  .modal-track-item .track-dur {
    color: var(--md-sys-color-on-surface-variant); font-size: 12px;
  }
  .modal-input-group { margin-bottom: 16px; }
  .modal-input-group label {
    display: block; font-size: 12px; line-height: 16px;
    color: var(--md-sys-color-on-surface-variant); margin-bottom: 4px;
  }
  .modal-input-group input {
    width: 100%; height: 40px; padding: 0 12px;
    border: 1px solid var(--md-sys-color-outline);
    border-radius: 4px; font-size: 14px; font-family: inherit;
    color: var(--md-sys-color-on-surface);
    background: var(--md-sys-color-surface-container-lowest);
  }
  .modal-input-group input:focus {
    outline: none; border-color: var(--md-sys-color-primary); border-width: 2px;
  }
  .modal-actions { display: flex; gap: 12px; justify-content: flex-end; }
  .upload-progress { margin: 16px 0; }
  .upload-progress .bar-bg {
    background: var(--md-sys-color-surface-container-highest);
    border-radius: 9999px; overflow: hidden; height: 4px;
  }
  .upload-progress .bar {
    height: 100%; background: var(--md-sys-color-primary);
    width: 0%; transition: width 0.3s; border-radius: 9999px;
  }
  .upload-progress .upload-status {
    margin-top: 8px; font-size: 14px; line-height: 20px;
    color: var(--md-sys-color-on-surface-variant);
  }
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
      <button class="btn btn-secondary" id="driveBtn" onclick="openDriveModal()">Google Driveにアップロード</button>
    </div>
  </div>
</div>

<!-- Google Drive Upload Modal -->
<div class="modal-overlay" id="driveModal">
  <div class="modal">
    <h3>Google Driveにアップロード</h3>
    <div class="modal-track-list" id="modalTrackList"></div>
    <div class="modal-input-group">
      <label for="folderName">フォルダ名</label>
      <input type="text" id="folderName">
    </div>
    <div class="upload-progress" id="uploadProgress" style="display:none;">
      <div class="bar-bg"><div class="bar" id="uploadBar"></div></div>
      <div class="upload-status" id="uploadStatus"></div>
    </div>
    <div class="modal-actions" id="modalActions">
      <button class="btn btn-secondary" onclick="closeDriveModal()">キャンセル</button>
      <button class="btn btn-primary" id="proceedBtn" onclick="proceedDrive()">このまま進む</button>
    </div>
  </div>
</div>

<script>
// --- State ---
let sessionId = null;
let tracks = [];
let nextIndex = 0;

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
    ...t, name: 'Track ' + t.index, audio: null, playing: false,
    cropMode: false, cropStart: 0, cropEnd: t.duration,
    deleted: false, ver: 1, cutMode: false, cutAt: 0,
  }));
  nextIndex = Math.max(...tracks.map(t => t.index)) + 1;
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
        <input class="track-name" id="name-${t.index}" value="${t.name}" spellcheck="false">
        <span class="info" id="info-${t.index}">${fmt(t.duration)} (${fmt(t.start)} - ${fmt(t.end)})</span>
      </div>
      <div class="waveform-container" id="wc-${t.index}">
        <canvas id="cv-${t.index}"></canvas>
        <div class="playback-bar" id="pb-${t.index}"></div>
        <div class="cut-line" id="cl-${t.index}"></div>
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
      <div class="cut-info" id="cuti-${t.index}">
        <span id="cut-pos-${t.index}"></span>
        <button onclick="applyCut(${t.index})">適用</button>
        <button class="cancel" onclick="cancelCut(${t.index})">キャンセル</button>
      </div>
      <div class="track-actions">
        <button id="playbtn-${t.index}" onclick="togglePlay(${t.index})">&#9654; 再生</button>
        <button onclick="startCrop(${t.index})">&#9986; クロップ</button>
        <button onclick="startCut(${t.index})">カット</button>
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
    ctx.fillStyle = '#1565C0';
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
  if (!t) return;

  // カットモード: クリックでカット位置を移動
  if (t.cutMode) {
    if (e.target.classList.contains('cut-line')) return;
    const r = wc.getBoundingClientRect();
    const pct = (e.clientX - r.left) / r.width;
    t.cutAt = pct * t.duration;
    document.getElementById('cl-' + idx).style.left = (pct * r.width) + 'px';
    updateCutInfo(idx);
    return;
  }

  if (t.cropMode) return;

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

// --- Cut ---
function saveNames() {
  for (const t of tracks) {
    if (t.deleted) continue;
    const inp = document.getElementById('name-' + t.index);
    if (inp) t.name = inp.value;
  }
}

function startCut(idx) {
  const t = tracks.find(x => x.index === idx);
  if (!t) return;
  t.cutMode = true;
  t.cutAt = t.duration / 2;

  const cl = document.getElementById('cl-' + idx);
  const wc = document.getElementById('wc-' + idx);
  cl.style.display = 'block';
  cl.style.left = (wc.clientWidth / 2) + 'px';

  document.getElementById('cuti-' + idx).classList.add('active');
  updateCutInfo(idx);
  setupCutDrag(idx);
}

function setupCutDrag(idx) {
  const t = tracks.find(x => x.index === idx);
  const wc = document.getElementById('wc-' + idx);
  const cl = document.getElementById('cl-' + idx);

  cl.onmousedown = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const wcRect = wc.getBoundingClientRect();
    const wcW = wcRect.width;

    function onMove(e2) {
      let x = e2.clientX - wcRect.left;
      x = Math.max(0, Math.min(x, wcW));
      t.cutAt = (x / wcW) * t.duration;
      cl.style.left = x + 'px';
      updateCutInfo(idx);
    }
    function onUp() {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    }
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  };
}

function updateCutInfo(idx) {
  const t = tracks.find(x => x.index === idx);
  const el = document.getElementById('cut-pos-' + idx);
  el.textContent = 'カット位置: ' + fmt(t.cutAt);
}

async function applyCut(idx) {
  const t = tracks.find(x => x.index === idx);
  if (!t) return;
  saveNames();

  const url = '/api/split-track/' + sessionId + '/' + t.filename +
    '?split_at=' + t.cutAt;
  const res = await fetch(url, { method: 'POST' });
  const data = await res.json();

  // 元のトラックを更新
  const origEnd = t.end;
  t.duration = data.track1.duration;
  t.waveform = data.track1.waveform;
  t.end = t.start + data.track1.duration;
  t.ver++;
  t.cutMode = false;
  t.cropStart = 0;
  t.cropEnd = t.duration;

  // 新しいトラックを作成
  const newTrack = {
    index: nextIndex++,
    filename: data.track2.filename,
    start: t.end,
    end: origEnd,
    duration: data.track2.duration,
    waveform: data.track2.waveform,
    name: t.name + 'b',
    audio: null, playing: false,
    cropMode: false, cropStart: 0, cropEnd: data.track2.duration,
    deleted: false, ver: 1, cutMode: false, cutAt: 0,
  };

  // 元のトラックの直後に挿入
  const pos = tracks.indexOf(t);
  tracks.splice(pos + 1, 0, newTrack);

  renderTracks();
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      for (const tr of tracks) { if (!tr.deleted) drawWaveform(tr); }
    });
  });
}

function cancelCut(idx) {
  const t = tracks.find(x => x.index === idx);
  if (t) { t.cutMode = false; t.cutAt = 0; }
  document.getElementById('cl-' + idx).style.display = 'none';
  document.getElementById('cuti-' + idx).classList.remove('active');
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

// ============================================================
// Google Drive アップロード
// ============================================================

let googleAccessToken = null;
let googleConfig = null;

function openDriveModal() {
  saveNames();
  const list = document.getElementById('modalTrackList');
  list.innerHTML = '';
  for (const t of tracks) {
    if (t.deleted) continue;
    const nameInput = document.getElementById('name-' + t.index);
    const name = nameInput ? nameInput.value : t.name;
    const item = document.createElement('div');
    item.className = 'modal-track-item';
    item.innerHTML = '<span>' + name + '</span><span class="track-dur">' + fmt(t.duration) + '</span>';
    list.appendChild(item);
  }
  const today = new Date();
  const yyyy = today.getFullYear();
  const mm = String(today.getMonth() + 1).padStart(2, '0');
  const dd = String(today.getDate()).padStart(2, '0');
  document.getElementById('folderName').value = yyyy + '-' + mm + '-' + dd;

  document.getElementById('uploadProgress').style.display = 'none';
  document.getElementById('uploadBar').style.width = '0%';
  document.getElementById('uploadStatus').textContent = '';
  document.getElementById('modalActions').innerHTML =
    '<button class="btn btn-secondary" onclick="closeDriveModal()">キャンセル</button>' +
    '<button class="btn btn-primary" id="proceedBtn" onclick="proceedDrive()">このまま進む</button>';
  document.getElementById('driveModal').classList.add('active');
}

function closeDriveModal() {
  document.getElementById('driveModal').classList.remove('active');
}

document.getElementById('driveModal').addEventListener('click', function(e) {
  if (e.target === this) closeDriveModal();
});

async function ensureGoogleAuth() {
  if (!googleConfig) {
    const res = await fetch('/api/google-config');
    googleConfig = await res.json();
  }
  const tokenRes = await fetch('/auth/token/' + sessionId);
  if (tokenRes.ok) {
    const data = await tokenRes.json();
    if (data.authenticated) {
      googleAccessToken = data.access_token;
      return true;
    }
  }
  return new Promise((resolve) => {
    const popup = window.open(
      '/auth/google?session_id=' + sessionId,
      'google-auth',
      'width=500,height=600,menubar=no,toolbar=no'
    );
    if (!popup) {
      alert('ポップアップがブロックされました。ポップアップを許可してください。');
      resolve(false);
      return;
    }
    function onMessage(event) {
      if (event.data && event.data.type === 'google-auth-success') {
        window.removeEventListener('message', onMessage);
        fetch('/auth/token/' + sessionId)
          .then(r => r.json())
          .then(data => { googleAccessToken = data.access_token; resolve(true); })
          .catch(() => resolve(false));
      }
    }
    window.addEventListener('message', onMessage);
    const pollTimer = setInterval(() => {
      if (popup.closed) {
        clearInterval(pollTimer);
        window.removeEventListener('message', onMessage);
        if (!googleAccessToken) resolve(false);
      }
    }, 500);
  });
}

let pickerApiLoaded = false;

function loadPickerApi() {
  return new Promise((resolve) => {
    if (pickerApiLoaded) { resolve(); return; }
    const script = document.createElement('script');
    script.src = 'https://apis.google.com/js/api.js';
    script.onload = () => {
      gapi.load('picker', () => { pickerApiLoaded = true; resolve(); });
    };
    document.head.appendChild(script);
  });
}

function openPicker() {
  return new Promise((resolve) => {
    const view = new google.picker.DocsView(google.picker.ViewId.FOLDERS)
      .setSelectFolderEnabled(true)
      .setMimeTypes('application/vnd.google-apps.folder');
    const picker = new google.picker.PickerBuilder()
      .setDeveloperKey(googleConfig.api_key)
      .setOAuthToken(googleAccessToken)
      .addView(view)
      .setTitle('アップロード先のフォルダを選択')
      .setCallback((data) => {
        if (data.action === google.picker.Action.PICKED) {
          resolve(data.docs[0]);
        } else if (data.action === google.picker.Action.CANCEL) {
          resolve(null);
        }
      })
      .build();
    picker.setVisible(true);
  });
}

async function proceedDrive() {
  const btn = document.getElementById('proceedBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>認証中...';

  const authed = await ensureGoogleAuth();
  if (!authed) {
    btn.disabled = false;
    btn.textContent = 'このまま進む';
    return;
  }

  btn.innerHTML = '<span class="spinner"></span>フォルダ選択中...';
  await loadPickerApi();
  const folder = await openPicker();
  if (!folder) {
    btn.disabled = false;
    btn.textContent = 'このまま進む';
    return;
  }

  const subfolderName = document.getElementById('folderName').value.trim() || 'tracks';
  const names = {};
  for (const t of tracks) {
    if (t.deleted) continue;
    const nameInput = document.getElementById('name-' + t.index);
    if (nameInput && nameInput.value.trim()) {
      names[t.filename] = nameInput.value.trim();
    }
  }

  document.getElementById('modalActions').style.display = 'none';
  const progressEl = document.getElementById('uploadProgress');
  progressEl.style.display = 'block';
  const uploadBar = document.getElementById('uploadBar');
  const uploadStatus = document.getElementById('uploadStatus');
  uploadStatus.textContent = 'アップロード準備中...';

  const response = await fetch('/api/upload-drive/' + sessionId, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      parent_folder_id: folder.id,
      subfolder_name: subfolderName,
      names: names,
    }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\\n');
    buffer = lines.pop();
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        if (data.type === 'progress') {
          const pct = (data.current / data.total * 100);
          uploadBar.style.width = pct + '%';
          uploadStatus.textContent = data.current + '/' + data.total + '曲アップロード中... (' + data.name + ')';
        } else if (data.type === 'done') {
          uploadBar.style.width = '100%';
          uploadStatus.textContent = 'アップロード完了!';
          document.getElementById('modalActions').innerHTML =
            '<button class="btn btn-primary" onclick="closeDriveModal()">閉じる</button>';
          document.getElementById('modalActions').style.display = 'flex';
        } else if (data.type === 'error') {
          uploadStatus.textContent = 'エラー: ' + data.message;
          document.getElementById('modalActions').innerHTML =
            '<button class="btn btn-secondary" onclick="closeDriveModal()">閉じる</button>';
          document.getElementById('modalActions').style.display = 'flex';
        }
      }
    }
  }
}
</script>
</body>
</html>
"""
