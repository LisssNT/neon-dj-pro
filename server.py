from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="NEON DJ PRO", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")

@app.get("/api/health")
def health():
    return JSONResponse({"ok": True, "service": "NEON DJ PRO", "analysis": "browser-local-worker", "heavy_ai": False})

@app.get("/api/info")
def info():
    return {
        "name": "NEON DJ PRO",
        "version": "2.0.0",
        "capabilities": ["two-deck playback", "crossfader", "3-band EQ", "approximate local BPM estimate"],
        "limitations": ["No server-side stem separation or speech transcription in this build"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
