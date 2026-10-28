"""Optional local AI backend for NEON DJ PRO LAB.
Runs CPU-heavy source separation / transcription only when dependencies are installed.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
OUTPUT_ROOT = ROOT / "output"
JOBS_ROOT = OUTPUT_ROOT / "jobs"
MAX_BYTES = int(os.getenv("NEON_MAX_UPLOAD_MB", "200")) * 1024 * 1024
MAX_ANALYZE_SECONDS = int(os.getenv("NEON_MAX_ANALYZE_SECONDS", "600"))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
JOBS_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="NEON DJ PRO LAB API", version="0.2.0")
_GENRE_PIPELINE = None
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("NEON_ALLOWED_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000,null").split(",") if origin.strip()],
    allow_credentials=False, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_ROOT)), name="outputs")


def installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    features = {
        "librosa": installed("librosa"),
        "demucs": installed("demucs"),
        "whisper": installed("faster_whisper"),
        "genre_model": installed("transformers"),
        "torch": installed("torch"),
        "soundfile": installed("soundfile"),
    }
    return {
        "ok": True,
        "app": "NEON DJ PRO LAB",
        "version": app.version,
        "features": features,
        "deep_analysis": features["librosa"],
        "source_separation": features["demucs"] and features["torch"],
        "transcription": features["whisper"],
        "note": "Availability means the Python package is installed; model downloads and compute capacity are checked at runtime.",
    }


def _key_estimate(chroma: Any) -> dict[str, Any]:
    import numpy as np
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    names = ["C", "C♯/D♭", "D", "D♯/E♭", "E", "F", "F♯/G♭", "G", "G♯/A♭", "A", "A♯/B♭", "B"]
    profile = np.asarray(chroma).mean(axis=1).astype(float)
    profile -= profile.mean()
    norm = np.linalg.norm(profile) or 1.0
    best = (-2.0, "C", "major")
    for shift in range(12):
        for mode, template in (("major", major), ("minor", minor)):
            corr = float(np.dot(profile, np.roll(template, shift)) / (norm * (np.linalg.norm(template) or 1)))
            if corr > best[0]:
                best = (corr, names[shift], mode)
    confidence = max(0.0, min(1.0, (best[0] + 0.15) / 0.9))
    return {"key": best[1], "mode": best[2], "confidence": round(confidence, 3)}


def _genre_guess(bpm: float, energy: float, low_ratio: float, centroid: float, onset_density: float) -> dict[str, Any]:
    scores = {"House": 0.0, "Techno": 0.0, "EDM": 0.0, "Hip-Hop / Trap": 0.0,
              "Phonk": 0.0, "Drum & Bass": 0.0, "Pop": 0.0, "Ambient / Downtempo": 0.0}
    if 118 <= bpm <= 134: scores["House"] += .45; scores["EDM"] += .25
    if 125 <= bpm <= 150: scores["Techno"] += .43; scores["EDM"] += .2
    if 160 <= bpm <= 185: scores["Drum & Bass"] += .7
    if 60 <= bpm <= 105: scores["Hip-Hop / Trap"] += .38; scores["Phonk"] += .22
    if low_ratio > .3: scores["Phonk"] += .3; scores["Hip-Hop / Trap"] += .2; scores["House"] += .12
    if energy > .48 and centroid > 2400: scores["EDM"] += .2
    if energy > .4 and onset_density > .25: scores["Techno"] += .14
    if energy < .16: scores["Ambient / Downtempo"] += .6
    if 90 <= bpm < 125 and .18 < energy < .55: scores["Pop"] += .25
    best = max(scores, key=scores.get)
    total = sum(max(0.0, v) for v in scores.values()) or 1.0
    confidence = max(.2, min(.72, scores[best] / (total * .75)))
    return {"genre": best, "confidence": round(confidence, 2),
            "alternatives": [{"genre": k, "score": round(v, 3)} for k, v in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]]}


def _ml_genre_guess(y: Any, sr: int) -> dict[str, Any] | None:
    """Optional broad-genre classifier, run on short segments to control memory use."""
    global _GENRE_PIPELINE
    if not installed("transformers"):
        return None
    try:
        import librosa
        import numpy as np
        from transformers import pipeline
        if _GENRE_PIPELINE is None:
            model_name = os.getenv("NEON_GENRE_MODEL", "dima806/music_genres_classification")
            device_setting = os.getenv("NEON_GENRE_DEVICE", "cpu").strip().lower()
            device = -1 if device_setting in {"cpu", "-1"} else int(device_setting)
            _GENRE_PIPELINE = pipeline("audio-classification", model=model_name, device=device)
        target_sr = 16000
        chunk_size = 10 * target_sr
        clip = y[:30 * sr]
        if len(clip) < sr * 4:
            clip = y
        resampled = librosa.resample(clip, orig_sr=sr, target_sr=target_sr)
        score_sum: dict[str, float] = {}
        n_chunks = 0
        for start in range(0, len(resampled), chunk_size):
            segment = resampled[start:start + chunk_size]
            if len(segment) < target_sr * 3:
                continue
            predictions = _GENRE_PIPELINE({"raw": segment, "sampling_rate": target_sr}, top_k=10)
            for item in predictions:
                label = str(item.get("label", "unknown")).strip()
                score_sum[label] = score_sum.get(label, 0.0) + float(item.get("score", 0.0))
            n_chunks += 1
        if not score_sum or not n_chunks:
            return None
        ranked = sorted(((k, v / n_chunks) for k, v in score_sum.items()), key=lambda x: x[1], reverse=True)
        return {"model": os.getenv("NEON_GENRE_MODEL", "dima806/music_genres_classification"),
                "genre": ranked[0][0], "confidence": round(ranked[0][1], 3),
                "alternatives": [{"genre": k, "score": round(v, 3)} for k, v in ranked[:5]],
                "scope": "broad genre classes; use subgenre heuristic for electronic styles"}
    except Exception as exc:
        # Classification is optional; don't block BPM/key/structure analysis if it fails.
        return {"error": str(exc), "available": False}


def analyze_audio(path: Path) -> dict[str, Any]:
    try:
        import librosa
        import numpy as np
    except Exception as exc:
        raise RuntimeError("Для server-side audio analysis установи librosa и numpy.") from exc

    y, sr = librosa.load(str(path), sr=22050, mono=True, duration=MAX_ANALYZE_SECONDS)
    if y.size < sr * 2:
        raise RuntimeError("Трек слишком короткий для надёжного анализа.")
    hop = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo_values = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop, aggregate=np.median)
    bpm = float(np.ravel(tempo_values)[0]) if np.size(tempo_values) else 0.0
    if bpm > 0:
        # Keep a conventional DJ tempo estimate; octave ambiguity remains possible.
        while bpm < 68: bpm *= 2
        while bpm > 178: bpm /= 2
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop, n_fft=4096)
    key = _key_estimate(chroma)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=hop)[0]
    spec = np.abs(librosa.stft(y=y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    total = spec.sum(axis=0) + 1e-9
    low_ratio = float((spec[freqs < 150].sum(axis=0) / total).mean())
    centroid_avg = float(centroid.mean())
    rms_avg = float(rms.mean())
    onset_density = float(np.mean(onset_env > (onset_env.mean() + onset_env.std() * .8)))
    energy = max(0.0, min(100.0, rms_avg * 380))
    # Eight-second blocks are explainable feature windows, not ground-truth verse/chorus labels.
    block_frames = max(1, round(8 * sr / hop))
    chunks = []
    for start in range(0, len(rms), block_frames):
        end = min(len(rms), start + block_frames)
        if end <= start: continue
        chunks.append({
            "start": round(start * hop / sr),
            "end": round(min(len(y) / sr, end * hop / sr)),
            "energy": float(rms[start:end].mean()),
            "low": float(low_ratio),
            "onset": float(onset_env[start:min(end, len(onset_env))].mean()) if start < len(onset_env) else 0,
            "centroid": float(centroid[start:end].mean()) if start < len(centroid) else centroid_avg,
        })
    energies = np.array([c["energy"] for c in chunks]) if chunks else np.array([0.0])
    med = float(np.median(energies)); peak = float(energies.max()) or 1e-9
    onset_mean = float(np.mean(onset_env)) if onset_env.size else 0
    sections = []
    for i, c in enumerate(chunks):
        prev = chunks[max(0, i - 1)]["energy"]
        nxt = chunks[min(len(chunks) - 1, i + 1)]["energy"]
        trend = nxt - prev
        label = "Groove"
        if c["start"] < 20 and c["energy"] < med * .9: label = "Intro"
        elif c["energy"] > med * 1.28 and c["onset"] > onset_mean * 1.08: label = "High energy"
        elif c["energy"] < med * .72: label = "Breakdown"
        elif trend > peak * .12: label = "Build-up"
        elif trend < -peak * .12: label = "Break / release"
        sections.append({"start": c["start"], "end": c["end"], "label": label,
                         "energy": round(min(100, c["energy"] * 380), 1),
                         "low": round(c["low"] * 100, 1), "onset": round(c["onset"], 3),
                         "centroid": round(c["centroid"])})
    genre = _genre_guess(bpm, rms_avg * 3.8, low_ratio, centroid_avg, onset_density)
    genre_model = _ml_genre_guess(y, sr)
    duration = float(len(y) / sr)
    return {
        "source": "server", "duration": round(duration, 1), "bpm": round(bpm, 1),
        "bpmConfidence": None, "key": key, "genre": genre, "genreModel": genre_model, "energy": round(energy),
        "lowRatio": round(low_ratio * 100), "spectralCentroid": round(centroid_avg),
        "spectralFlatness": None, "onsetDensity": round(onset_density * 100),
        "beatOffset": None, "sections": sections,
    }


def run_demucs(input_path: Path, job_id: str) -> tuple[dict[str, str], str | None, Path | None]:
    if not (installed("demucs") and installed("torch")):
        return {}, "Demucs/PyTorch не установлены: полноценное source separation недоступно.", None
    out_dir = JOBS_ROOT / job_id / "separated"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = os.getenv("NEON_DEMUCS_DEVICE", "cpu")
    timeout = int(os.getenv("NEON_DEMUCS_TIMEOUT", "1800"))
    command = [sys.executable, "-m", "demucs.separate", "-n", os.getenv("NEON_DEMUCS_MODEL", "htdemucs"), "-d", device, "-o", str(out_dir), str(input_path)]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {}, f"Demucs превысил лимит времени {timeout} секунд.", None
    except Exception as exc:
        return {}, f"Не удалось запустить Demucs: {exc}", None
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "no logs")[-900:]
        return {}, f"Demucs завершился с ошибкой: {tail}", None
    stem_dir = out_dir / os.getenv("NEON_DEMUCS_MODEL", "htdemucs") / input_path.stem
    stems: dict[str, str] = {}
    for name in ("vocals", "drums", "bass", "other"):
        p = stem_dir / f"{name}.wav"
        if p.exists():
            stems[name] = f"/outputs/jobs/{job_id}/separated/{os.getenv('NEON_DEMUCS_MODEL', 'htdemucs')}/{input_path.stem}/{name}.wav"
    if not stems:
        return {}, "Demucs выполнился, но ожидаемые WAV-stems не найдены.", None
    # Reconstruct a convenient no-vocals instrumental from the three aligned non-vocal stems.
    instrumental_warning = None
    try:
        import numpy as np
        import soundfile as sf
        non_vocal_paths = [stem_dir / f"{name}.wav" for name in ("drums", "bass", "other")]
        if all(path.exists() for path in non_vocal_paths):
            arrays = []
            rates = []
            for path in non_vocal_paths:
                arr, rate = sf.read(str(path), dtype="float32", always_2d=True)
                arrays.append(arr); rates.append(rate)
            if len(set(rates)) == 1 and len({arr.shape[1] for arr in arrays}) == 1:
                length = min(len(arr) for arr in arrays)
                instrumental = arrays[0][:length] + arrays[1][:length] + arrays[2][:length]
                peak = float(np.max(np.abs(instrumental))) if instrumental.size else 0.0
                if peak > 0.99:
                    instrumental *= (0.99 / peak)
                inst_path = stem_dir / "instrumental.wav"
                sf.write(str(inst_path), instrumental, rates[0], subtype="PCM_24")
                stems["instrumental"] = f"/outputs/jobs/{job_id}/separated/{os.getenv('NEON_DEMUCS_MODEL', 'htdemucs')}/{input_path.stem}/instrumental.wav"
            else:
                instrumental_warning = "Не удалось собрать instrumental: несовпадающие форматы non-vocal stems."
        else:
            instrumental_warning = "Instrumental mix не создан: отсутствуют отдельные drums/bass/other stems."
    except Exception as exc:
        instrumental_warning = f"Не удалось собрать instrumental mix: {exc}"
    vocal = stem_dir / "vocals.wav"
    return stems, instrumental_warning, vocal if vocal.exists() else None


def run_transcription(audio_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not installed("faster_whisper"):
        return None, "faster-whisper не установлен: транскрипция текста недоступна."
    try:
        from faster_whisper import WhisperModel
        model_name = os.getenv("NEON_WHISPER_MODEL", "base")
        device = os.getenv("NEON_WHISPER_DEVICE", "cpu")
        compute_type = os.getenv("NEON_WHISPER_COMPUTE_TYPE", "int8")
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        segments_iter, info = model.transcribe(str(audio_path), beam_size=5, vad_filter=True, word_timestamps=False)
        segments = []
        full_text = []
        for seg in segments_iter:
            text = seg.text.strip()
            if text:
                segments.append({"start": round(float(seg.start), 2), "end": round(float(seg.end), 2), "text": text})
                full_text.append(text)
        # Repeated phrases can indicate a hook, but this is not formal song-section recognition.
        freq: dict[str, int] = {}
        for seg in segments:
            norm = re.sub(r"[^\w\s]", "", seg["text"].lower()).strip()
            if len(norm) > 8: freq[norm] = freq.get(norm, 0) + 1
        repeated = [k for k, v in freq.items() if v > 1]
        return {"language": getattr(info, "language", None), "language_probability": round(float(getattr(info, "language_probability", 0) or 0), 3),
                "text": " ".join(full_text), "segments": segments, "repeated_lines": repeated[:20],
                "note": "Распознавание пения может ошибаться; текст/таймкоды необходимо проверить на слух."}, None
    except Exception as exc:
        return None, f"Whisper transcription failed: {exc}"


@app.post("/api/deep-analyze")
async def deep_analyze(
    file: UploadFile = File(...),
    separate: bool = Form(True),
    transcribe: bool = Form(True),
) -> dict[str, Any]:
    original_name = Path(file.filename or "uploaded-audio.wav").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".mp4", ".webm"}:
        suffix = ".audio"
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    source_path = job_dir / f"source{suffix}"
    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Пустой файл.")
    if len(contents) > MAX_BYTES:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(413, f"Файл превышает лимит {MAX_BYTES // (1024 * 1024)} MB.")
    source_path.write_bytes(contents)
    warnings: list[str] = []
    try:
        analysis = analyze_audio(source_path)
    except Exception as exc:
        analysis = None
        warnings.append(f"Audio analysis failed: {exc}")
    stems: dict[str, str] = {}
    vocal_path: Path | None = None
    if separate:
        stems, stem_warning, vocal_path = run_demucs(source_path, job_id)
        if stem_warning: warnings.append(stem_warning)
    transcript = None
    if transcribe:
        transcription_source = vocal_path or source_path
        transcript, transcript_warning = run_transcription(transcription_source)
        if transcript_warning: warnings.append(transcript_warning)
    messages = ["Server analysis finished."]
    if stems: messages.append(f"Separated {len(stems)} stems with Demucs.")
    if transcript: messages.append(f"Whisper transcribed {len(transcript.get('segments', []))} vocal segments.")
    if not stems and not transcript: messages.append("Для полной обработки установи необязательные AI-зависимости из requirements-ai.txt; см. README.")
    return {"ok": analysis is not None, "job_id": job_id, "analysis": analysis, "stems": stems,
            "transcript": transcript, "warnings": warnings, "message": " ".join(messages),
            "source_name": original_name}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")), reload=False)
