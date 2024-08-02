import logging
import os

from flask import Flask, request
from flask_restful import Api

from tubesage.clients.embbeding_client import OllamaEmbeddingClient
from tubesage.services.text_splitter import LangChainSmartTextSplitter
from tubesage.controllers.transcribe_video_controller import TranscribeVideoController
from tubesage.routes.transcribe_video_route import TranscribeVideoRouteV1

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def handle_exception(e):
    logger.error(f"Error in {request.path}: {e}", exc_info=True)
    return {"message": str(e)}, 500


def create_app():
    app = Flask(__name__)
    api = Api(app)

    app.register_error_handler(Exception, handle_exception)

    chroma_retriever_map = {}
    text_splitter = LangChainSmartTextSplitter()
    embedding_client = OllamaEmbeddingClient("llama3.1")

    transcribe_video_controller = TranscribeVideoController(chroma_retriever_map, text_splitter, embedding_client)

    api.add_resource(
        TranscribeVideoRouteV1,
        "/v1/transcribe-video",
        resource_class_kwargs={"controller": transcribe_video_controller},
    )

    return app


if __name__ == "__main__":
    app = create_app()
    port = os.environ.get("SERVICE_PORT", 5000)
    logger.info(f"Using port {port}")
    app.run(debug=True, host="0.0.0.0", port=port)
