import logging
from typing import List
from langchain_community.embeddings import OllamaEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EmbeddingClient:
    def get_embedding(self, text: str) -> List[float]:
        pass

    def get_embedding_llm(self):
        pass


class OllamaEmbeddingClient(EmbeddingClient):
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        """
        Initialize the OllamaClient with a specific model.

        Args:
            model (str): The model name to be used for embeddings.
            base_url (str): The base URL of the Ollama service
        """
        try:
            self.embeddings = OllamaEmbeddings(base_url=base_url, model=model)
            logger.info(f"Initialized OllamaEmbeddings with model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize OllamaEmbeddings: {e}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        """
        Get the embedding for a given text.

        Args:
            text (str): The input text to embed.

        Returns:
            list: The embedding of the input text.
        """
        try:
            embedding = self.embeddings.embed_query(text)
            logger.info("Successfully retrieved embedding")
            return embedding
        except Exception as e:
            logger.error(f"An error occurred while getting embedding: {e}")
            raise

    def get_embedding_llm(self):
        return self.embeddings


if __name__ == "__main__":
    client = OllamaEmbeddingClient("llama3.1")
    text = "Hello, World!"
    embedding = client.get_embedding(text)
    logger.info(f"Embedding: {embedding}")
