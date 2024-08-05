import pytest
from tubesage.util.extract_video_id import extract_video_id


def test_correct_desktop_extraction():
    input_urls = [
        "https://www.youtube.com/watch?v=w3-Opzwv-VM",
        "https://www.youtube.com/watch?v=w3-Opzwv-Vk",
        "https://www.youtube.com/watch?v=test-hello",
        "https://www.youtube.com/watch?v=w3-",
        "https://www.youtube.com/watch?v=w3-Opzwv-VM&ab_channel=Example",
        "https://www.youtube.com/watch?v=w3-Opzwv-VM&t=123s",
    ]

    expected_ids = ["w3-Opzwv-VM", "w3-Opzwv-Vk", "test-hello", "w3-", "w3-Opzwv-VM", "w3-Opzwv-VM"]

    result_ids = [extract_video_id(url) for url in input_urls]

    assert result_ids == expected_ids


def test_correct_mobile_extraction():
    input_urls = [
        "https://youtu.be/w3-Opzwv-VM?si=v9L1UfSg0MkpkWyA",
    ]

    expected_ids = ["w3-Opzwv-VM"]

    result_ids = [extract_video_id(url) for url in input_urls]

    assert result_ids == expected_ids


def test_invalid_urls():
    invalid_urls = [
        "https://www.google.com",
        "https://www.youtube.com",
        "https://www.foo.bar",
        "https://www.cnn.com/watch?v=w3-",
        "https://www.youtube.com/watch?v=",
        "https://www.youtube.com/watch",
        "https://www.youtu.be",
    ]

    for url in invalid_urls:
        with pytest.raises(ValueError, match="Invalid YouTube URL"):
            extract_video_id(url)
