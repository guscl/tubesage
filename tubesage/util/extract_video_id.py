import re


def extract_video_id(video_url: str) -> str:
    match = re.search(r"v=([^&]+)", video_url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid YouTube URL")
