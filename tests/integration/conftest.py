"""Integration test fixtures for local HTTP server."""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class _IntegrationRequestHandler(BaseHTTPRequestHandler):
    """Serve deterministic HTML/JSON responses for crawling tests."""

    server_version = "Crawl4AITestServer/0.1"

    def do_GET(self) -> None:  # noqa: D401
        path = self.path.split("?", 1)[0]
        if path == "/static":
            self._send_html(
                """
                <html><head><title>Static Page</title></head>
                <body>
                    <h1>Integration Static</h1>
                    <p id="body">Hello integration tests.</p>
                </body>
                </html>
                """
            )
        elif path == "/js":
            self._send_html(
                """
                <html><head><title>JS Page</title></head>
                <body>
                    <h1>JS Interaction</h1>
                    <div id="output">Pending</div>
                    <script>
                        async function run() {
                            const response = await fetch('/api/message');
                            const data = await response.json();
                            document.getElementById('output').innerText = data.message;
                        }
                        setTimeout(run, 50);
                    </script>
                </body>
                </html>
                """
            )
        elif path == "/links":
            self._send_html(
                """
                <html><head><title>Links</title></head>
                <body>
                    <h1>Link Hub</h1>
                    <a href="/article/alpha">Alpha topic</a>
                    <a href="/article/beta">Beta topic</a>
                </body>
                </html>
                """
            )
        elif path.startswith("/article/"):
            topic = path.rsplit("/", 1)[-1]
            self._send_html(
                f"""
                <html><head><title>Article {topic}</title></head>
                <body>
                    <article>
                        <h2>Topic {topic.title()}</h2>
                        <p>
                            This article references topic {topic}
                            for best-first tests.
                        </p>
                    </article>
                </body>
                </html>
                """
            )
        elif path == "/api/message":
            self._send_json({"message": "Hello from API"})
        elif path == "/forbidden":
            self.send_error(HTTPStatus.FORBIDDEN)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args):  # noqa: D401
        return

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, str]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="session")
def integration_server() -> Iterator[str]:
    """Start a lightweight HTTP server for integration tests."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()

    httpd = ThreadingHTTPServer((host, port), _IntegrationRequestHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        thread.join()
