import re


def extract_video_id(video_url: str) -> str:
    if "youtube.com" in video_url:
        match = re.search(r"(?<=v=)[^&#]+", video_url)
    elif "youtu.be" in video_url:
        match = re.search(r"(?<=youtu\.be/)[^?&]+", video_url)
    else:
        match = None

    if match:
        return match.group(0)
    else:
        raise ValueError("Invalid YouTube URL")
