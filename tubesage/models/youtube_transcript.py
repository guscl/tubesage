from dataclasses import dataclass

@dataclass
class YoutubeTranscript():
    text: str
    start: float
    duration: float

