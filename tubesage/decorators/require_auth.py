import os
from flask import request, jsonify
from functools import wraps

# Change this to only fetch the real one if deploying somewhere
# Using a default value to help with Docker Compose and local development
API_KEY = os.getenv("TUBESAGE_API_KEY", "tubesage_api_key")


def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header or auth_header != f"ApiKey {API_KEY}":
            response = jsonify({"message": "Forbidden"})
            response.status_code = 403
            return response
        return f(*args, **kwargs)

    return decorated_function
