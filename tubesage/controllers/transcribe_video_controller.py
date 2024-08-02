import logging
import re
from typing import Dict
from langchain_core.vectorstores import VectorStoreRetriever
from tubesage.services.text_splitter import TextSplitter
from tubesage.clients.embbeding_client import EmbeddingClient
from tubesage.clients.video_transcript import YoutubeTranscriptClient
from tubesage.clients.vector_db import ChromaClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TranscribeVideoController:
    def __init__(
        self,
        chroma_retriever_map: Dict[str, VectorStoreRetriever],
        text_splitter: TextSplitter,
        embedding_client: EmbeddingClient,
    ):
        self.chroma_retriever_map = chroma_retriever_map
        self.transcription_client = YoutubeTranscriptClient()
        self.text_splitter = text_splitter
        self.embedding_client = embedding_client

    def run(self, video_url: str):
        video_id = self._extract_video_id(video_url)
        if video_id in self.chroma_retriever_map:
            return

        full_transcript = self._transcribe_video(video_id)
        retriever = self._add_text_to_chroma(full_transcript)
        self.chroma_retriever_map[video_id] = retriever
        return full_transcript

    def _add_text_to_chroma(self, text: str) -> VectorStoreRetriever:
        try:
            db = ChromaClient(self.text_splitter, self.embedding_client)
            db.add_concurrent_texts(text)
            return db.get_retriever()
        except Exception as e:
            logger.error(f"Failed to add video transcript to Chroma: {e}")
            raise e

    def _transcribe_video(self, video_id: str):
        try:
            _, full_transcript = self.transcription_client.get_transcript(video_id)
            return full_transcript
        except Exception as e:
            logger.error(f"Failed to transcribe video: {e}")
            raise e

    def _extract_video_id(self, video_url: str) -> str:
        match = re.search(r"v=([^&]+)", video_url)
        if match:
            return match.group(1)
        else:
            raise ValueError("Invalid YouTube URL")
