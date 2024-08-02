import logging
from typing import Dict
from langchain_core.vectorstores import VectorStoreRetriever
from tubesage.clients.llm_client import LLMClient
from tubesage.services.qa_engine import LangChainQA
from tubesage.util.extract_video_id import extract_video_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class InvokeLLMController:
    def __init__(
        self,
        chroma_retriever_map: Dict[str, VectorStoreRetriever],
        llm: LLMClient,
    ):
        self.chroma_retriever_map = chroma_retriever_map
        self.llm = llm

    def run(self, input: str, video_url: str):
        video_id = extract_video_id(video_url)
        if video_id not in self.chroma_retriever_map:
            return "Video not transcribed yet"

        qa_engine = LangChainQA(self.llm, self.chroma_retriever_map[video_id])
        return qa_engine.invoke({"input": input})
