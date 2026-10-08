# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Shared Flask-based live-video web UI for Ubuntu-py apps.

A grid of windows (`set_windows`, default 1x1), each showing live video
(`set_frame`) or one still image (`set_image`) with a caption (`set_text`); an
"Info" side panel (`set_sections`); and browser keydown callbacks (`on_keypress`).

All state lives on a `WebUI` object with its own Flask app, so an app creates
one (`ui = WebUI()`, or `WebUI(port=...)`) and drives it (`ui.start_thread()`,
`ui.set_frame(...)`, ...). Several independent servers can coexist.

SECURITY: the server binds 0.0.0.0. The video stream and state are readable by
anyone who can reach the port, and any `on_keypress` callback runs for any JSON
POST to `/keypress`, not just the browser UI. (Cross-origin browser requests are
blocked: `/keypress` requires a JSON content type and no CORS headers are sent.)
Only run this on a trusted, isolated network.
"""

import dataclasses
import inspect
import logging
import os
import queue
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any, cast

import cv2
from flask import Flask, Response, jsonify, render_template, request
from werkzeug import serving

_logger = logging.getLogger(__name__)

# Templates and static assets ship next to this module, in the repo and in a
# bundle alike, so they are located relative to it rather than the CWD.
_ASSET_DIR = "assets/webui"

# Frames wider than this are downscaled before JPEG encoding; see `set_frame`.
_DEFAULT_MAX_WIDTH = 1280

# An Info row is (label, value); a list value renders as an indented sub-group.
InfoRow = tuple[str, "str | list[tuple[str, str]]"]


@dataclasses.dataclass
class _Window:
    latest_jpeg: bytes | None = None
    frame_id: int = 0
    title: str | None = None
    text: str = ""
    # True when the window holds one still image (`set_image`) rather than a
    # live stream, so the client fetches it once instead of opening /stream.
    static: bool = False


def _mjpeg_generator(win: _Window) -> Iterator[bytes]:
    sent = -1
    while True:
        frame = win.latest_jpeg
        if frame is None or win.frame_id == sent:
            time.sleep(0.01)
            continue
        sent = win.frame_id
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


class WebUI:
    """One live-video web UI: its own Flask app, windows, Info panel and callbacks."""

    def __init__(self, port: int = 8080) -> None:
        self.port = port
        self.app = Flask(
            __name__,
            template_folder=_ASSET_DIR,
            static_folder=_ASSET_DIR,
            static_url_path="/assets",
        )
        self.app.config["MAX_CONTENT_LENGTH"] = 4 * 1024
        # State is swapped by single assignments, so no locks are needed.
        self._windows: list[_Window] = [_Window()]
        self._rows = 1
        self._columns_css = "repeat(1, 1fr)"
        self._app_title = "App"
        self._info_sections: list[dict[str, Any]] = []
        self._keypress_callbacks: dict[str, Callable[[], Any]] = {}
        self._server_started = False
        self._callback_queue: queue.Queue[Callable[[], Any]] = queue.Queue()

        self.app.add_url_rule("/", "index", self._index)
        # Bare /stream aliases window 0 for apps that hardcode it.
        self.app.add_url_rule(
            "/stream", "stream_window", self._stream_window, defaults={"window": 0}
        )
        self.app.add_url_rule(
            "/stream/<int:window>", "stream_window", self._stream_window
        )
        self.app.add_url_rule("/image/<int:window>", "image_window", self._image_window)
        self.app.add_url_rule("/state", "state", self._state)
        self.app.add_url_rule(
            "/keypress", "keypress_route", self._keypress_route, methods=["POST"]
        )

    def _get_window(self, window: int) -> _Window | None:
        if 0 <= window < len(self._windows):
            return self._windows[window]
        _logger.warning(
            "window=%d is out of range (have %d window(s)); ignoring.",
            window,
            len(self._windows),
        )
        return None

    def set_windows(
        self,
        rows: int,
        cols: int,
        titles: list[str] | None = None,
        col_widths: list[float] | None = None,
    ) -> None:
        """Lay out `rows` x `cols` windows, matplotlib-`subplots`-style.

        `col_widths` sets each column's relative width (e.g. `[1, 2, 2]`); it needs
        `cols` entries, and defaults to equal widths. Replaces any previous windows
        (frames/text already set are lost). Must be called before `start_thread()`,
        else `RuntimeError`.
        """
        if self._server_started:
            raise RuntimeError("set_windows() must be called before start_thread()")
        if rows < 1 or cols < 1:
            raise ValueError(
                f"rows and cols must be >= 1, got rows={rows}, cols={cols}"
            )
        n = rows * cols
        if titles is not None and len(titles) != n:
            raise ValueError(
                f"titles has {len(titles)} entries, expected {n} (rows*cols)"
            )
        if col_widths is not None and (
            len(col_widths) != cols or any(w <= 0 for w in col_widths)
        ):
            raise ValueError(
                f"col_widths needs {cols} entries, all > 0, got {col_widths}"
            )
        self._rows = rows
        self._columns_css = (
            " ".join(f"{w}fr" for w in col_widths)
            if col_widths
            else f"repeat({cols}, 1fr)"
        )
        self._windows = [_Window(title=titles[i] if titles else None) for i in range(n)]

    def set_title(self, app_name: str) -> None:
        """Set the app-specific name shown in the browser tab title.

        Safe to call at any time, including after `start_thread()`.
        """
        self._app_title = app_name

    def on_keypress(
        self, key: str, callback: Callable[[], Any] | Callable[[str], Any]
    ) -> None:
        """Register `callback` to run when `key` is pressed in the browser.

        `callback` takes no arguments, or the key string. Re-registering a key
        overwrites it. Safe to call at any time. Callbacks run one at a time on a
        background thread; one that raises is logged.

        Per the SECURITY note above, the endpoint is unauthenticated, so a callback
        should not do anything destructive or irreversible unless the app confirms
        it separately.
        """
        if len(inspect.signature(callback).parameters) >= 1:
            one_arg = cast(Callable[[str], Any], callback)
            self._keypress_callbacks[key] = lambda: one_arg(key)
        else:
            self._keypress_callbacks[key] = cast(Callable[[], Any], callback)

    def set_frame(
        self,
        bgr_frame: cv2.typing.MatLike,
        window: int = 0,
        jpeg_quality: int = 80,
        max_width: int | None = _DEFAULT_MAX_WIDTH,
    ) -> None:
        """Publish `bgr_frame` as `window`'s current video frame.

        For a window showing a single result rather than live video, use
        `set_image`: a live window holds an MJPEG stream open and shows a frame
        only once the next one arrives, so one `set_frame` call never paints.

        A frame wider than `max_width` is downscaled to it first, preserving the
        aspect ratio. Pass `max_width=None` to encode full size.
        """
        self._publish(bgr_frame, window, jpeg_quality, max_width, static=False)

    def set_image(
        self,
        bgr_image: cv2.typing.MatLike,
        window: int = 0,
        jpeg_quality: int = 80,
        max_width: int | None = _DEFAULT_MAX_WIDTH,
    ) -> None:
        """Publish `bgr_image` as `window`'s still image."""
        self._publish(bgr_image, window, jpeg_quality, max_width, static=True)

    def _publish(
        self,
        bgr_frame: cv2.typing.MatLike,
        window: int,
        jpeg_quality: int,
        max_width: int | None,
        static: bool,
    ) -> None:
        """Encode `bgr_frame` and make it `window`'s current image."""
        win = self._get_window(window)
        if win is None:
            return
        if max_width is not None and bgr_frame.shape[1] > max_width:
            height = round(bgr_frame.shape[0] * max_width / bgr_frame.shape[1])
            bgr_frame = cv2.resize(
                bgr_frame, (max_width, height), interpolation=cv2.INTER_AREA
            )
        ok, jpg = cv2.imencode(
            ".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        )
        if not ok:
            return
        win.latest_jpeg = jpg.tobytes()
        win.static = static
        win.frame_id += 1

    def set_text(self, text: str, window: int = 0) -> None:
        """Set caption text shown under `window`'s stream (e.g. status, FPS)."""
        win = self._get_window(window)
        if win is None:
            return
        win.text = text

    def set_sections(self, sections: list[dict[str, Any]]) -> None:
        """Set the sections of the "Info" side panel.

        Each section is a dict with a `title` and `rows`, e.g.::

            ui.set_sections([
                {"title": "Model overview", "rows": [
                    ("Model", "easyocr_detector.tflite"),
                    ("Backend", "HTP"),
                ]},
                {"title": "Last capture", "rows": [
                    ("Detector latency", "12.3 ms"),
                ]},
            ])

        Rows are `(label, value)` pairs, rendered as escaped text; a value may be
        a list of pairs, shown as an indented sub-group. An empty list hides the
        panel. Replaces any previous sections; safe to call at any time.
        """
        self._info_sections = sections

    def _index(self) -> str:
        return render_template(
            "index.html",
            app_title=self._app_title,
            rows=self._rows,
            columns_css=self._columns_css,
            windows=self._windows,
        )

    def _stream_window(self, window: int) -> Response:
        win = self._get_window(window)
        if win is None:
            return Response(status=404)
        return Response(
            _mjpeg_generator(win), mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    def _image_window(self, window: int) -> Response:
        """Serve `window`'s current image as a plain JPEG (see `set_image`)."""
        win = self._get_window(window)
        if win is None or win.latest_jpeg is None:
            return Response(status=404)
        return Response(win.latest_jpeg, mimetype="image/jpeg")

    def _state(self) -> Response:
        windows = [
            {
                "text": w.text,
                "has_frame": w.latest_jpeg is not None,
                "frame": w.frame_id,
                "static": w.static,
            }
            for w in self._windows
        ]
        return jsonify({"windows": windows, "sections": self._info_sections})

    def _keypress_route(self) -> dict[str, bool]:
        data = request.get_json(silent=True)
        key = data.get("key") if isinstance(data, dict) else None
        callback = self._keypress_callbacks.get(key) if isinstance(key, str) else None
        if callback is not None:
            self._callback_queue.put(callback)
        return {"handled": callback is not None}

    def _callback_worker(self) -> None:
        while True:
            job = self._callback_queue.get()
            try:
                job()
            except Exception:
                _logger.exception("on_keypress callback raised")

    def run_server(self) -> None:
        # Show the host IP (from launch.sh) in the banner, not the container's.
        host_ip = os.environ.get("QAIHA_HOST_IP")
        if host_ip:
            serving.get_interface_ip = lambda family: host_ip
        server = serving.make_server("0.0.0.0", self.port, self.app, threaded=True)
        server.log_startup()
        # Silence per-request logs after the startup banner.
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        server.serve_forever()

    def start_thread(self) -> None:
        self._server_started = True
        threading.Thread(target=self._callback_worker, daemon=True).start()
        threading.Thread(target=self.run_server, daemon=True).start()
