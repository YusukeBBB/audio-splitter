"""
音声分割プロトタイプ
スタジオ録音の音源ファイルを曲ごとに自動分割する。

判定ロジック:
- RMSエネルギー（音量）とスペクトル帯域幅（周波数の広がり）を組み合わせて
  「バンド演奏中」か「曲間（MC/静寂）」かを判定する。
- バンド演奏 = 音量が大きく、低音〜高音まで幅広い周波数が鳴っている
- MC/静寂   = 音量が小さい、または声だけで周波数の広がりが狭い
"""

import argparse
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(file_path: str) -> tuple[np.ndarray, int]:
    """
    音声ファイルを読み込む。wav以外はffmpegでwavに変換してから読む。
    モノラルのfloat32配列とサンプリングレートを返す。
    """
    path = Path(file_path)
    if path.suffix.lower() in ('.wav', '.flac'):
        audio, sr = sf.read(str(path), dtype='float32')
    else:
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.close()
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-i', str(path), '-ac', '1', '-ar', '44100', '-f', 'wav', tmp.name],
                capture_output=True, check=True,
            )
            audio, sr = sf.read(tmp.name, dtype='float32')
        finally:
            os.unlink(tmp.name)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio, sr


def compute_rms(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """フレームごとのRMSエネルギーを計算する。"""
    n_frames = 1 + (len(audio) - frame_length) // hop_length
    rms = np.empty(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + frame_length]
        rms[i] = np.sqrt(np.mean(frame ** 2))
    return rms


def compute_spectral_bandwidth(audio: np.ndarray, sr: int, frame_length: int, hop_length: int) -> np.ndarray:
    """
    フレームごとのスペクトル帯域幅を計算する。
    バンド演奏中は広い（ドラム、ベース、ギター、ボーカルが全帯域で鳴る）。
    MC（トーク）や静寂は狭い（声の周波数帯のみ or 無音）。
    """
    n_frames = 1 + (len(audio) - frame_length) // hop_length
    freqs = np.fft.rfftfreq(frame_length, d=1.0 / sr)
    bandwidth = np.empty(n_frames)

    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + frame_length]
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(frame_length)))
        power = spectrum ** 2
        total_power = power.sum()

        if total_power < 1e-20:
            bandwidth[i] = 0.0
        else:
            centroid = np.sum(freqs * power) / total_power
            bandwidth[i] = np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power)

    return bandwidth


def amplitude_to_db(amplitude: np.ndarray) -> np.ndarray:
    """振幅をdBに変換する（最大値基準）。"""
    ref = np.max(amplitude)
    if ref == 0:
        return np.full_like(amplitude, -80.0)
    return 20 * np.log10(np.maximum(amplitude / ref, 1e-10))


def smooth(data: np.ndarray, window_size: int) -> np.ndarray:
    """移動平均でスムージングする。"""
    if window_size <= 1:
        return data
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='same')


@dataclass
class Segment:
    """分割された1曲分のセグメント"""
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float


def detect_splits(
    audio: np.ndarray,
    sr: int,
    min_silence_duration: float = 5.0,
    min_song_duration: float = 30.0,
    frame_length: int = 4096,
    hop_length: int = 2048,
) -> list[Segment]:
    """
    音声データから曲間を検出し、セグメントのリストを返す。
    RMSエネルギーとスペクトル帯域幅の両方を使って
    「バンド演奏が止んだ区間」を検出する。
    """
    frame_to_sec = hop_length / sr
    # スムージング窓（約1秒分のフレーム数）
    smooth_frames = max(1, int(1.0 / frame_to_sec))

    # --- RMSエネルギー ---
    rms = compute_rms(audio, frame_length, hop_length)
    rms_db = amplitude_to_db(rms)
    rms_db_smooth = smooth(rms_db, smooth_frames)

    # --- スペクトル帯域幅 ---
    bw = compute_spectral_bandwidth(audio, sr, frame_length, hop_length)
    bw_smooth = smooth(bw, smooth_frames)

    # --- 閾値の決定 ---
    # エネルギー: 上位25%のフレーム（演奏中）の平均から一定量下を閾値にする
    rms_sorted = np.sort(rms_db_smooth)
    loud_mean = np.mean(rms_sorted[int(len(rms_sorted) * 0.75):])
    energy_thresh = loud_mean - 15  # 演奏中の平均より15dB低い

    # 帯域幅: 上位25%の平均の40%以下は「狭い」= 演奏してない
    bw_sorted = np.sort(bw_smooth)
    bw_loud_mean = np.mean(bw_sorted[int(len(bw_sorted) * 0.75):])
    bw_thresh = bw_loud_mean * 0.4

    print(f"  エネルギー閾値: {energy_thresh:.1f} dB (演奏中平均: {loud_mean:.1f} dB)")
    print(f"  帯域幅閾値: {bw_thresh:.0f} Hz (演奏中平均: {bw_loud_mean:.0f} Hz)")

    # --- 「演奏していない」フレームを検出 ---
    # エネルギーが低い OR 帯域幅が狭い → 演奏が止んでいる
    is_not_playing = (rms_db_smooth < energy_thresh) | (bw_smooth < bw_thresh)

    # --- 連続する非演奏区間を検出 ---
    quiet_regions = []
    in_quiet = False
    quiet_start = 0

    for i, q in enumerate(is_not_playing):
        if q and not in_quiet:
            quiet_start = i
            in_quiet = True
        elif not q and in_quiet:
            duration = (i - quiet_start) * frame_to_sec
            if duration >= min_silence_duration:
                quiet_regions.append((quiet_start, i))
            in_quiet = False

    if in_quiet:
        quiet_end = len(is_not_playing)
        duration = (quiet_end - quiet_start) * frame_to_sec
        if duration >= min_silence_duration:
            quiet_regions.append((quiet_start, quiet_end))

    print(f"  検出された曲間候補: {len(quiet_regions)} 箇所")
    for idx, (s, e) in enumerate(quiet_regions):
        t_start = s * frame_to_sec
        t_end = e * frame_to_sec
        print(f"    曲間{idx + 1}: {t_start:.1f}s - {t_end:.1f}s ({t_end - t_start:.1f}s間)")

    # --- スプリットポイント ---
    split_samples = [(((s + e) // 2) * hop_length) for s, e in quiet_regions]

    # --- セグメント生成 ---
    total_samples = len(audio)
    boundaries = [0] + split_samples + [total_samples]
    segments = []

    for i in range(len(boundaries) - 1):
        start_sample = boundaries[i]
        end_sample = boundaries[i + 1]
        start_sec = start_sample / sr
        end_sec = end_sample / sr
        segments.append(Segment(
            index=i,
            start_sec=start_sec,
            end_sec=end_sec,
            duration_sec=end_sec - start_sec,
        ))

    # --- 短すぎるセグメントを結合 ---
    merged = []
    for seg in segments:
        if merged and seg.duration_sec < min_song_duration:
            prev = merged[-1]
            merged[-1] = Segment(
                index=prev.index,
                start_sec=prev.start_sec,
                end_sec=seg.end_sec,
                duration_sec=seg.end_sec - prev.start_sec,
            )
        else:
            merged.append(seg)

    for i, seg in enumerate(merged):
        seg.index = i

    return merged


def split_and_save(
    input_path: str,
    output_dir: str,
    min_silence_duration: float = 5.0,
    min_song_duration: float = 30.0,
) -> list[Segment]:
    """
    音声ファイルを曲ごとに分割して保存する。
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"読み込み中: {input_path}")
    audio, sr = load_audio(str(input_path))
    total_duration = len(audio) / sr
    print(f"  長さ: {total_duration / 60:.1f} 分 ({total_duration:.0f} 秒)")
    print(f"  サンプリングレート: {sr} Hz")

    print("曲間を検出中...")
    segments = detect_splits(
        audio, sr,
        min_silence_duration=min_silence_duration,
        min_song_duration=min_song_duration,
    )

    print(f"\n分割結果: {len(segments)} 曲")

    stem = input_path.stem

    for seg in segments:
        start_sample = int(seg.start_sec * sr)
        end_sample = int(seg.end_sec * sr)
        segment_audio = audio[start_sample:end_sample]

        filename = f"{stem}_track{seg.index + 1:02d}.wav"
        output_path = output_dir / filename
        sf.write(str(output_path), segment_audio, sr)

        minutes = int(seg.duration_sec // 60)
        seconds = int(seg.duration_sec % 60)
        print(f"  Track {seg.index + 1:2d}: {minutes}:{seconds:02d} "
              f"({seg.start_sec:.1f}s - {seg.end_sec:.1f}s) -> {filename}")

    print(f"\n出力先: {output_dir}")
    return segments


def main():
    parser = argparse.ArgumentParser(
        description="スタジオ録音の音源(wav, m4a等)を曲ごとに自動分割する"
    )
    parser.add_argument("input", help="入力音声ファイルのパス (wav, m4a等)")
    parser.add_argument(
        "--output-dir", "-o",
        default="./output",
        help="出力ディレクトリ (デフォルト: ./output)",
    )
    parser.add_argument(
        "--min-silence-duration",
        type=float,
        default=5.0,
        help="曲間と判定する最小秒数 (デフォルト: 5.0)",
    )
    parser.add_argument(
        "--min-song-duration",
        type=float,
        default=30.0,
        help="最小曲長(秒)。これより短いセグメントは前の曲と結合 (デフォルト: 30.0)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"エラー: ファイルが見つかりません: {args.input}")
        return

    split_and_save(
        input_path=args.input,
        output_dir=args.output_dir,
        min_silence_duration=args.min_silence_duration,
        min_song_duration=args.min_song_duration,
    )


if __name__ == "__main__":
    main()
