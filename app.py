"""
AI Video Learning Assistant — Flask Backend
============================================
Main application entry point with REST API endpoints.
"""

import os
import sys
import json
import traceback
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS

# Add parent to path so imports work
sys.path.insert(0, os.path.dirname(__file__))

from utils.video_utils import extract_video_id, get_video_metadata
from services.transcript_service import fetch_transcript
from services.llm_service import process_transcript

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Caching setup
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(video_id, language):
    return os.path.join(CACHE_DIR, f"{video_id}_{language}.json")

def get_cached_result(video_id, language):
    path = get_cache_path(video_id, language)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            return None
    return None

def save_to_cache(video_id, language, result):
    path = get_cache_path(video_id, language)
    try:
        with open(path, "w") as f:
            json.dump(result, f)
    except Exception as e:
        print(f"[app] Cache save failed: {e}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.route("/api/summarize-video", methods=["POST"])
def summarize_video():
    """
    Main endpoint: accepts a YouTube URL and returns learning content.

    Request JSON:
        {"url": "https://www.youtube.com/watch?v=..."}

    Response JSON:
        {
            "video": { "title", "channel", "thumbnail", "video_id" },
            "summary": "...",
            "key_points": ["...", "..."],
            "notes": "...",
            "quiz": [{"question", "options": [], "correct": int}, ...],
            "transcript_source": "captions" | "whisper"
        }
    """
    data = request.get_json(force=True, silent=True) or {}
    url = data.get("url", "").strip()
    language = data.get("language", "en").strip()

    if not url:
        return jsonify({"error": "Please provide a YouTube video URL."}), 400

    # 1. Extract video ID
    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL. Please check the link and try again."}), 400

    # 2. Check Cache
    cached = get_cached_result(video_id, language)
    if cached:
        print(f"[app] Serving cached results for {video_id} ({language})")
        return jsonify(cached)

    # 3. Fetch video metadata
    metadata = get_video_metadata(video_id)
    metadata["video_id"] = video_id

    # 4. Fetch transcript
    transcript_result = fetch_transcript(video_id)
    if "error" in transcript_result:
        return jsonify({"error": transcript_result["error"]}), 422

    transcript_text = transcript_result["text"]
    transcript_segments = transcript_result.get("segments", [])
    transcript_source = transcript_result["source"]

    # 5. Run AI pipeline
    try:
        ai_result = process_transcript(transcript_text, transcript_segments, target_language=language)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"AI processing failed: {str(e)}"}), 500

    # 6. Build and Cache result
    final_result = {
        "video": metadata,
        "summary": ai_result["summary"],
        "key_points": ai_result["key_points"],
        "notes": ai_result["notes"],
        "quiz": ai_result["quiz"],
        "transcript_source": transcript_source,
    }
    save_to_cache(video_id, language, final_result)

    # 7. Return combined result
    return jsonify(final_result)



@app.route("/api/transcript", methods=["POST"])
def get_transcript():
    """Endpoint to just fetch the transcript without AI processing."""
    data = request.get_json(force=True, silent=True) or {}
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "Please provide a YouTube video URL."}), 400

    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL."}), 400

    transcript_result = fetch_transcript(video_id)
    if "error" in transcript_result:
        return jsonify({"error": transcript_result["error"]}), 422

    return jsonify({
        "video_id": video_id,
        "transcript": transcript_result["text"],
        "source": transcript_result["source"],
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  AI Video Learning Assistant — Backend Server")
    print("=" * 60)
    print("  Starting on http://localhost:5001")
    print("  Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=5001, debug=True)
