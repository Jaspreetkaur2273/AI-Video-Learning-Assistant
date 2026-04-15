"""
whisper_service.py — Speech-to-text fallback using OpenAI Whisper.
Downloads audio from YouTube via yt-dlp and transcribes it.
"""

import os
import tempfile
import subprocess
import torch
import whisper

# ---------------------------------------------------------------------------
# Eager Model Loading (Initialize once on startup for speed)
# ---------------------------------------------------------------------------
_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[whisper_service] Initializing Whisper 'tiny' model on {_DEVICE}...")

try:
    _whisper_model = whisper.load_model("tiny", device=_DEVICE)
except Exception as e:
    print(f"[whisper_service] Error loading Whisper model: {e}")
    _whisper_model = None

def transcribe_video(video_id: str) -> str:
    """Download audio from a YouTube video and transcribe it using Whisper."""
    if _whisper_model is None:
        raise RuntimeError("Whisper model not initialized. Check logs.")

    url = f"https://www.youtube.com/watch?v={video_id}"

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")

        # Download LOW-QUALITY audio for faster processing
        # -f 'ba[abr<64]' targets lowest bitrate audio
        result = subprocess.run(
            [
                "yt-dlp",
                "-f", "ba[abr<64]/ba*",
                "--no-playlist",
                "-o", f"{tmpdir}/audio.%(ext)s",
                "--quiet",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )

        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr}")

        # Find the actual audio file
        actual_path = audio_path
        if not os.path.exists(actual_path):
            for f in os.listdir(tmpdir):
                if f.startswith("audio"):
                    actual_path = os.path.join(tmpdir, f)
                    break

        if not os.path.exists(actual_path):
            raise RuntimeError("Audio file not found after download.")

        # Transcribe with the persistent model
        result = _whisper_model.transcribe(actual_path)
        
        # Format segments for our service
        segments = [
            {"text": s.get("text", "").strip(), "start": s.get("start", 0)}
            for s in result.get("segments", [])
        ]
        
        return {
            "text": result.get("text", ""),
            "segments": segments
        }
