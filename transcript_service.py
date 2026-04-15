"""
transcript_service.py — Fetches YouTube video transcripts.
Primary: youtube-transcript-api
Fallback: Whisper speech-to-text (via whisper_service)
"""

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter


import time
from deep_translator import GoogleTranslator
import subprocess
import os
import tempfile
import re
from services.whisper_service import transcribe_video

def _fetch_yt_dlp_segments(video_id: str) -> str:
    """Fallback: use yt-dlp to get auto-generated subtitles."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use yt-dlp to get auto-generated captions in srt/vtt format
        cmd = [
            "yt-dlp",
            "--write-auto-subs",
            "--sub-lang", "en",
            "--skip-download",
            "--output", f"{tmpdir}/sub",
            url
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=30)
            # Find the subtitle file
            for f in os.listdir(tmpdir):
                if f.endswith(".vtt") or f.endswith(".srt"):
                    with open(os.path.join(tmpdir, f), 'r') as file:
                        content = file.read()
                        # Simple cleanup of VTT/SRT tags
                        text = re.sub(r'<[^>]+>', '', content) # Remove HTML tags
                        text = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3}.*', '', text) # Remove timestamps
                        text = "\n".join([line.strip() for line in text.split('\n') if line.strip()])
                        return text
        except:
            pass
    return None

def fetch_transcript(video_id: str, target_language: str = "en") -> dict:
    """
    Attempt to fetch a transcript. 
    Order: 1. Manual/Auto Captions (API), 2. yt-dlp auto-subs (Quick Fallback), 3. Whisper (Slow Fallback)
    """
    overall_t0 = time.time()
    
    # --- Attempt 1: YouTube Transcript API ---
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        
        transcript = None
        translation_needed = False
        
        try:
            transcript = transcript_list.find_transcript([target_language])
        except:
            try:
                transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
            except:
                for t in transcript_list:
                    transcript = t
                    translation_needed = True
                    break
                
        if transcript:
            print(f"[transcript_service] Found {transcript.language_code} transcript")
            if transcript.language_code == target_language:
                translation_needed = False

            fetched = transcript.fetch()
            # Preserve segments for timestamp mapping
            segments = [
                {"text": s.get("text", "").strip(), "start": s.get("start", 0)} 
                for s in fetched if isinstance(s, dict)
            ]
            text = " ".join([s["text"] for s in segments])
            
            if text and len(text.strip()) > 50:
                # Basic translation if needed (skipping full segment translation for speed)
                if translation_needed and target_language == "en" and transcript.language_code != 'en':
                    import concurrent.futures
                    translator = GoogleTranslator(source='auto', target='en')
                    words = text.split()
                    chunks = [" ".join(words[i:i+800]) for i in range(0, len(words), 800)]
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        text = " ".join(list(executor.map(translator.translate, chunks)))
                
                print(f"[transcript_service] API captions fetched in {time.time() - overall_t0:.2f}s")
                return {"text": text.strip(), "segments": segments, "source": "captions"}
    except Exception as e:
        print(f"[transcript_service] API captions failed: {e}")

    # --- Attempt 2: yt-dlp fast fallback ---
    try:
        t0_ytdlp = time.time()
        # For simplicity, yt-dlp fallback currently returns raw text without timestamps
        # In a real app, we'd parse the VTT, but Whisper is our high-fidelity fallback.
        text = _fetch_yt_dlp_segments(video_id)
        if text and len(text.strip()) > 50:
            print(f"[transcript_service] yt-dlp auto-subs fetched in {time.time() - t0_ytdlp:.2f}s")
            return {"text": text.strip(), "segments": [], "source": "yt-dlp auto-subs"}
    except Exception as e:
        print(f"[transcript_service] yt-dlp fallback failed: {e}")

    # --- Attempt 3: Whisper fallback ---
    try:
        t0_whisper = time.time()
        # Update whisper_service to return segments
        whisper_data = transcribe_video(video_id) # Now returns {text, segments}
        text = whisper_data.get("text", "")
        segments = whisper_data.get("segments", [])
        
        if text and len(text.strip()) > 50:
            print(f"[transcript_service] Whisper transcription took {time.time() - t0_whisper:.2f}s")
            return {"text": text.strip(), "segments": segments, "source": "whisper"}
    except Exception as e:
        print(f"[transcript_service] Whisper fallback failed: {e}")

    return {"error": "Could not retrieve transcript."}

