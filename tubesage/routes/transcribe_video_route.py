from werkzeug.exceptions import BadRequest
from flask_restful import Resource
from flask import request
from tubesage.controllers.transcribe_video_controller import TranscribeVideoController
from tubesage.decorators.require_auth import require_auth
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TranscribeVideoRouteV1(Resource):
    def __init__(
        self,
        controller: TranscribeVideoController,
    ):
        self.controller = controller

    @require_auth
    def post(self):
        request_data = request.get_json()

        required_fields = ["video_url"]

        if not all([field in request_data for field in required_fields]):
            missing_fields = [field for field in required_fields if field not in request_data]
            raise BadRequest(f"Missing required fields: {', '.join(missing_fields)}")

        video_url = request_data["video_url"]

        try:
            text = self.controller.run(video_url)
            return {"transcription": text}, 200
        except Exception as e:
            logger.error(f"Unable to transcribe video {video_url}. {e}", exc_info=True)
            return {"error": f"Unable to transcribe video {video_url}"}, 500
