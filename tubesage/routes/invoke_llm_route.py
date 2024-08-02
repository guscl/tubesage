from werkzeug.exceptions import BadRequest
from flask_restful import Resource
from flask import request
from tubesage.controllers.invoke_llm_controller import InvokeLLMController
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class InvokeLLMRouteV1(Resource):
    def __init__(
        self,
        controller: InvokeLLMController,
    ):
        self.controller = controller

    def post(self):
        request_data = request.get_json()

        required_fields = ["input", "video_url"]

        if not all([field in request_data for field in required_fields]):
            missing_fields = [field for field in required_fields if field not in request_data]
            raise BadRequest(f"Missing required fields: {', '.join(missing_fields)}")

        input = request_data["input"]
        video_url = request_data["video_url"]

        try:
            text = self.controller.run(input, video_url)
            return {"response": text}, 200
        except Exception as e:
            logger.error(f"Unable to invoke llm {video_url}, input {input}. {e}", exc_info=True)
            return {"error": f"Unable to invoke llm {video_url}, input {input}"}, 500
