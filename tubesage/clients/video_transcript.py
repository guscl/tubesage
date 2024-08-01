import os
import logging
from typing import List, Tuple
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, VideoUnavailable
from tubesage.models.youtube_transcript import YoutubeTranscript

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class VideoTranscriptClient:
    def get_transcript(self, video_id: str):
        pass


class YoutubeTranscriptClient(VideoTranscriptClient):
    """
    A client class for retrieving transcripts from YouTube videos.
    """

    def get_transcript(self, video_id: str) -> Tuple[List[YoutubeTranscript], str]:
        """
        Retrieves the transcript and full text of a YouTube video.

        Args:
            video_id (str): The ID of the YouTube video.

        Returns:
            Tuple[List[YoutubeTranscript], str]: A tuple containing a list of YoutubeTranscript objects
            representing the individual parts of the transcript, and a string containing the full transcript.
        """
        transcripts = []
        full_transcript_parts = []

        try:
            response_list = YouTubeTranscriptApi.get_transcript(video_id)
        except TranscriptsDisabled:
            logger.error(f"Transcripts are disabled for video ID: {video_id}")
            raise
        except VideoUnavailable:
            logger.error(f"Video is unavailable for video ID: {video_id}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise

        for response in response_list:
            full_transcript_parts.append(f"{response['text']} ")
            transcripts.append(
                YoutubeTranscript(
                    text=response["text"],
                    start=response["start"],
                    duration=response["duration"],
                )
            )

        full_transcript = "".join(full_transcript_parts)
        return transcripts, full_transcript


if __name__ == "__main__":
    client = YoutubeTranscriptClient()
    key = "VMj-3S1tku0&t"

    transcripts, full_transcript = client.get_transcript(key)
    try:
        data_dir = "../data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info(f"Created directory: {data_dir}")

        transcript_path = os.path.join(data_dir, f"transcript_{key}.txt")
        with open(transcript_path, "w") as f:
            f.write(full_transcript)

        logger.info(f"Transcript saved to {transcript_path}")
    except Exception as e:
        logger.error(f"Failed to save transcript for video ID: {key} - {e}")
