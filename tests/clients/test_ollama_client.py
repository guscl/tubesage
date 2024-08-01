import pytest
from unittest.mock import patch
from tubesage.clients.embbeding_client import OllamaEmbeddingClient


@pytest.fixture
def mock_embeddings():
    return [
        0.47720080614089966,
        1.183258295059204,
        -0.482545405626297,
    ]


def test_get_emebeddings(mock_embeddings):
    expected_embeddings = [
        0.47720080614089966,
        1.183258295059204,
        -0.482545405626297,
    ]

    client = OllamaEmbeddingClient("any_model")
    text = "any text"

    with patch("tubesage.clients.ollama.OllamaEmbeddingClient.get_embedding", return_value=mock_embeddings):
        result_embeddings = client.get_embedding(text)

    assert result_embeddings == expected_embeddings


def test_get_embeddings_unexpected_error():
    client = OllamaEmbeddingClient("any_model")
    text = "any_text"

    with patch(
        "tubesage.clients.ollama.OllamaEmbeddingClient.get_embedding",
        side_effect=Exception("Unexpected error"),
    ):
        with pytest.raises(Exception, match="Unexpected error"):
            _ = client.get_embedding(text)
