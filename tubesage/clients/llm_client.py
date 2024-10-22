import logging
from langchain_ollama.chat_models import ChatOllama

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LLMClient:
    def get_llm(self):
        pass


class OllamaLLMClient(LLMClient):
    """
    A client class for interacting with the OllamaLLM language model.

    Args:
        model (str): The name of the language model to use.
        base_url (str, optional): The base URL of the OllamaLLM server.
    """

    # This odd base_url is because I'm running both images with the same docker-compose file sharing the same network
    def __init__(self, model: str, base_url: str = "http://ollama:11434"):
        try:
            self.llm = ChatOllama(base_url=base_url, model=model, temperature=0)
            logger.info(f"Initialized OllamaLLM with model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize OllamaLLM: {e}")
            raise

    def invoke(self, input: str) -> str:
        """
        Invoke the OllamaLLM language model with the given input.

        Args:
            input (str): The input text to be processed by the language model.

        Returns:
            str: The output generated by the language model.
        """
        return self.llm.invoke(input)

    def get_llm(self) -> ChatOllama:
        """
        Get the underlying OllamaLLM instance.

        Returns:
            OllamaLLM: The OllamaLLM instance used by the client.
        """
        return self.llm


if __name__ == "__main__":
    llm_client = OllamaLLMClient(model="llama3.1")
    text = "I got catch them all"
    response = llm_client.invoke(text)
    logger.info(f"LLM response: {response}")
