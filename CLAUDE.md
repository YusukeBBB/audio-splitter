# Audio Splitter

スタジオ録音の音源(wav, m4a等)を曲ごとに自動分割するWebツール。

## 技術スタック
- バックエンド: Python FastAPI + numpy + soundfile + ffmpeg
- フロントエンド: HTML/JS (app.py内に埋め込み)

## 起動方法
```bash
cd src
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8000
```
http://localhost:8000 でアクセス

## 主な機能
- 音声ファイルのアップロード（D&D対応）
- RMSエネルギー + スペクトル帯域幅による自動曲間検出
- 各トラックの波形表示・再生・シーク
- クロップ（トリミング）・削除
- トラック名の編集
- ZIPダウンロード（編集した名前が反映される）
