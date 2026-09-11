"""Deterministic loopback HTTP origin for the corpus downloader demo and tests.

The server implements the subset of RFC 9110 that the downloader relies on:
byte ranges, strong ETag validators, ``If-Range``, ``206``/``416`` responses,
and a one-shot truncated body that simulates a dropped connection. It binds
only to ``127.0.0.1`` on an ephemeral port and is never used for external
traffic.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import cast
from urllib.parse import urlsplit

_RANGE_RE = re.compile(r"bytes=(\d+)-")


@dataclass(frozen=True)
class ServedRequest:
    """One request as observed by the local origin."""

    method: str
    range_header: str | None
    if_range: str | None


class _RangeHTTPServer(ThreadingHTTPServer):
    owner: LocalRangeServer


class _RangeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_HEAD(self) -> None:
        self._serve(head=True)

    def do_GET(self) -> None:
        self._serve(head=False)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _serve(self, *, head: bool) -> None:
        owner = cast(_RangeHTTPServer, self.server).owner
        if urlsplit(self.path).path != owner.path:
            self.send_error(404, "unknown path")
            return

        range_header = self.headers.get("Range")
        if_range = self.headers.get("If-Range")
        payload, etag, honor_range, fail_after = owner._request_state(
            method=self.command,
            range_header=range_header,
            if_range=if_range,
            consume_failure=not head,
        )

        status = 200
        body = payload
        content_range: str | None = None

        if (
            not head
            and fail_after is None
            and honor_range
            and range_header is not None
            and (if_range is None or if_range == etag)
        ):
            match = _RANGE_RE.fullmatch(range_header.strip())
            if match is not None:
                start = int(match.group(1))
                if start >= len(payload):
                    status = 416
                    body = b""
                    content_range = f"bytes */{len(payload)}"
                elif start > 0:
                    status = 206
                    body = payload[start:]
                    content_range = f"bytes {start}-{len(payload) - 1}/{len(payload)}"

        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", etag)
        self.send_header("Connection", "close")
        if content_range is not None:
            self.send_header("Content-Range", content_range)
        self.end_headers()
        self.close_connection = True

        if head or status == 416:
            return

        if fail_after is not None:
            try:
                self.wfile.write(payload[:fail_after])
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        try:
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class LocalRangeServer:
    """A loopback zstd shard origin with optional deterministic failure."""

    def __init__(
        self,
        payload: bytes,
        *,
        path: str = "/shard.jsonl.zst",
        etag: str = '"v1"',
        honor_range: bool = True,
        fail_first_get_after: int | None = None,
    ) -> None:
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        if not path.startswith("/"):
            raise ValueError("path must start with '/'")
        if not etag:
            raise ValueError("etag must be non-empty")
        if fail_first_get_after is not None:
            if isinstance(fail_first_get_after, bool) or not isinstance(fail_first_get_after, int):
                raise TypeError("fail_first_get_after must be an int")
            if not 0 < fail_first_get_after < len(payload):
                raise ValueError("fail_first_get_after must be between 1 and len(payload) - 1")
        self._payload = payload
        self._path = path
        self._etag = etag
        self._honor_range = honor_range
        self._fail_first_get_after = fail_first_get_after
        self._failure_consumed = False
        self._request_log: list[ServedRequest] = []
        self._lock = threading.Lock()
        self._server: _RangeHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._url: str | None = None

    @property
    def path(self) -> str:
        return self._path

    @property
    def url(self) -> str:
        if self._url is None:
            raise RuntimeError("server has not been started")
        return self._url

    @property
    def requests(self) -> tuple[ServedRequest, ...]:
        with self._lock:
            return tuple(self._request_log)

    def replace(self, payload: bytes, *, etag: str) -> None:
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        if not etag:
            raise ValueError("etag must be non-empty")
        with self._lock:
            self._payload = payload
            self._etag = etag

    def start(self) -> None:
        if self._server is not None:
            return
        server = _RangeHTTPServer(("127.0.0.1", 0), _RangeHandler)
        server.owner = self
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._server = server
        self._thread = thread
        address = cast(tuple[str, int], server.server_address)
        self._url = f"http://{address[0]}:{address[1]}{self._path}"

    def stop(self) -> None:
        server = self._server
        thread = self._thread
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=5.0)
        self._server = None
        self._thread = None

    def __enter__(self) -> LocalRangeServer:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    def _request_state(
        self,
        *,
        method: str,
        range_header: str | None,
        if_range: str | None,
        consume_failure: bool,
    ) -> tuple[bytes, str, bool, int | None]:
        with self._lock:
            self._request_log.append(
                ServedRequest(method=method, range_header=range_header, if_range=if_range)
            )
            fail_after: int | None = None
            if (
                consume_failure
                and self._fail_first_get_after is not None
                and not self._failure_consumed
            ):
                fail_after = self._fail_first_get_after
                self._failure_consumed = True
            return self._payload, self._etag, self._honor_range, fail_after
