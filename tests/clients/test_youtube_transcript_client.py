import pytest
from unittest.mock import patch
from youtube_transcript_api._errors import TranscriptsDisabled, VideoUnavailable
from tubesage.clients.video_transcript import YoutubeTranscriptClient, YoutubeTranscript


@pytest.fixture
def mock_transcript_data():
    return [
        {"text": "Hello", "start": 0.0, "duration": 1.0},
        {"text": "world", "start": 1.0, "duration": 1.0},
    ]


def test_get_transcript_success(mock_transcript_data):
    expected_transcripts = [
        YoutubeTranscript(text="Hello", start=0.0, duration=1.0),
        YoutubeTranscript(text="world", start=1.0, duration=1.0),
    ]

    client = YoutubeTranscriptClient()
    video_id = "valid_video_id"

    with patch(
        "tubesage.clients.video_transcript.YouTubeTranscriptApi.get_transcript", return_value=mock_transcript_data
    ):
        transcripts, full_transcript = client.get_transcript(video_id)

    assert full_transcript == "Hello world "
    assert len(transcripts) == len(mock_transcript_data)
    assert transcripts == expected_transcripts


def test_get_transcript_transcripts_disabled():
    client = YoutubeTranscriptClient()
    video_id = "disabled_video_id"

    with patch(
        "tubesage.clients.video_transcript.YouTubeTranscriptApi.get_transcript",
        side_effect=TranscriptsDisabled(video_id),
    ):
        with pytest.raises(TranscriptsDisabled):
            client.get_transcript(video_id)


def test_get_transcript_video_unavailable():
    client = YoutubeTranscriptClient()
    video_id = "unavailable_video_id"

    with patch(
        "tubesage.clients.video_transcript.YouTubeTranscriptApi.get_transcript", side_effect=VideoUnavailable(video_id)
    ):
        with pytest.raises(VideoUnavailable):
            client.get_transcript(video_id)


def test_get_transcript_unexpected_error():
    client = YoutubeTranscriptClient()
    video_id = "error_video_id"

    with patch(
        "tubesage.clients.video_transcript.YouTubeTranscriptApi.get_transcript",
        side_effect=Exception("Unexpected error"),
    ):
        with pytest.raises(Exception, match="Unexpected error"):
            client.get_transcript(video_id)
