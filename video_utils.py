"""
video_utils.py — Utility functions for extracting YouTube video IDs and metadata.
"""

import re
from googleapiclient.discovery import build

YOUTUBE_API_KEY = "AIzaSyBOkOTx6cLKCR_wjW9de6VwAXcFwLAIpL8"

# Regex patterns covering all common YouTube URL formats
YOUTUBE_URL_PATTERNS = [
    r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
    r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
    r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})',
    r'(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})',
    r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
]


def extract_video_id(url: str) -> str | None:
    """Extract the 11-character YouTube video ID from a URL."""
    if not url:
        return None
    url = url.strip()
    for pattern in YOUTUBE_URL_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # Fallback: if the input is already a raw 11-char ID
    if re.fullmatch(r'[a-zA-Z0-9_-]{11}', url):
        return url
    return None


def get_video_metadata(video_id: str) -> dict:
    """Fetch video title, channel, duration, and thumbnail using YouTube Data API v3."""
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        request = youtube.videos().list(part="snippet,contentDetails", id=video_id)
        response = request.execute()

        if not response.get("items"):
            return {"title": "Unknown", "channel": "Unknown", "thumbnail": ""}

        item = response["items"][0]
        snippet = item["snippet"]
        return {
            "title": snippet.get("title", "Unknown"),
            "channel": snippet.get("channelTitle", "Unknown"),
            "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
            "published_at": snippet.get("publishedAt", ""),
            "duration": item.get("contentDetails", {}).get("duration", ""),
        }
    except Exception as e:
        print(f"[video_utils] Error fetching metadata: {e}")
        return {"title": "Unknown", "channel": "Unknown", "thumbnail": ""}
