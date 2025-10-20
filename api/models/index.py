import sys
import os
import json
from http.server import BaseHTTPRequestHandler

# Ensure backend modules are importable
backend_path = os.path.join(os.path.dirname(__file__), "..", "..", "backend")
backend_path = os.path.abspath(backend_path)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from llm_service import LLMService

llm_service = LLMService()


class handler(BaseHTTPRequestHandler):
    def _set_common_headers(self, status_code=200):
        self.send_response(status_code)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_common_headers()

    def do_GET(self):
        try:
            data = {
                "default_model": llm_service.get_default_model(),
                "models": llm_service.list_available_models(),
            }
            self._set_common_headers()
            self.wfile.write(json.dumps(data).encode())
        except Exception as exc:
            error = {"error": str(exc)}
            self._set_common_headers(status_code=500)
            self.wfile.write(json.dumps(error).encode())

