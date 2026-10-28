# NEON DJ PRO — Complete Starter Build

A browser-based two-deck mixing studio with local audio decoding, playback, crossfader, three-band EQ, pitch/rate control, and approximate BPM/level analysis in a Web Worker.

## Run locally

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

Open `http://127.0.0.1:8000`.

## Deploy on Render

- Runtime: Python
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`
- Root directory: repository root

The app is static-first: playback and analysis happen in the browser. The backend only serves files and a health endpoint. No music files are uploaded to the server.

## How to use

1. Add audio files in the library.
2. Use the A and B buttons beside each track to load each deck.
3. Press Play on the decks. On mobile Safari, the first playback must be started by a user gesture.
4. Move the crossfader to blend the decks.
5. Use the 3-band EQ and rate controls to adjust the sound.
6. Select a track and run the approximate local analysis; export its JSON report if needed.

## Important limitations

- BPM is an estimate based on signal changes, not a studio-grade beatgrid.
- Rate changes playback speed and pitch together. This is not key-locked time stretching.
- No automatic phase-accurate beatmatching, cue-point memory, streaming music, stem separation, or AI transcription is included in this build.
- Supported audio formats vary by browser. MP3 and WAV are the safest choices.
- Large uncompressed audio files can consume substantial memory, especially on phones. Use moderate-size files and close other tabs if playback stutters.
- Browser playback requires user interaction; check the phone's output route and browser sound permissions if audio is silent.

## Project layout

```text
neon-dj-pro-complete/
├── index.html
├── server.py
├── requirements.txt
├── render.yaml
├── README.md
└── static/
    └── analysis-worker.js
```
