"""Local-only Mars-1 Mini API implemented with Python's standard library.

This is a developer experiment, not a public production API.
Run only on your own computer after a checkpoint has been trained.
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from .generate import complete, load_checkpoint


class MarsHandler(BaseHTTPRequestHandler):
    model = None
    inference_lock = threading.Lock()

    def send_json(self, body: dict, status: int = 200) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json({
                "name": "Mars-1 Mini",
                "status": "experimental",
                "model": "original small Transformer",
            })
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            return self.send_json({"error": "Not found"}, 404)
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self.send_json({"error": "Invalid length"}, 400)
        if size <= 0 or size > 4_096:
            return self.send_json({"error": "Request too large"}, 413)
        try:
            request = json.loads(self.rfile.read(size).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeError):
            return self.send_json({"error": "Invalid JSON"}, 400)
        message = request.get("message") if isinstance(request, dict) else None
        if not isinstance(message, str) or not message.strip() or len(message) > 500:
            return self.send_json({"error": "Send a text message under 500 characters"}, 400)

        try:
            with self.inference_lock, torch.inference_mode():
                reply = complete(self.model, message.strip(), max_new_tokens=120)
            return self.send_json({"reply": reply, "model": "mars-1-mini"})
        except Exception:
            return self.send_json({"error": "Model generation failed"}, 500)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local original Mars-1 Mini API")
    parser.add_argument("--checkpoint", default="artifacts/mars1-mini/mars1-mini.pt")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    MarsHandler.model = load_checkpoint(args.checkpoint)
    # Local-only for safety: DO NOT expose this experimental API to the Internet.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), MarsHandler)
    print(f"Mars-1 Mini local API: http://127.0.0.1:{args.port}/health")
    print('POST /api/chat with JSON: {"message": "hello"}')
    print("Not a public backend; press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
