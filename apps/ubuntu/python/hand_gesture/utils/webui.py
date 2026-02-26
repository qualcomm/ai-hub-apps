# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import logging
import threading
import time

import cv2
from flask import Flask, Response

app = Flask(__name__)

latest_jpeg = None
lock = threading.Lock()


# this is a bottleneck for large images
def set_frame(bgr_frame, jpeg_quality=80):
    global latest_jpeg
    ok, jpg = cv2.imencode(
        ".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    )
    if not ok:
        return
    with lock:
        latest_jpeg = jpg.tobytes()


def mjpeg_generator():
    # Streams the most recent frame repeatedly
    while True:
        with lock:
            frame = latest_jpeg
        if frame is None:
            time.sleep(0.01)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        # max 60fps
        time.sleep(1 / 60)


@app.route("/")
def index():
    return """
    <html>
      <head><title>RB3 Stream</title></head>
      <body style="margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh;">
        <img src="/stream" style="max-width:100%;max-height:100%;" />
      </body>
    </html>
    """


@app.route("/stream")
def stream():
    return Response(
        mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def run_server():
    app.run(host="0.0.0.0", port=8080, threaded=True)


def start_thread():
    # Run this early in your main
    threading.Thread(target=run_server, daemon=True).start()

    # Turn off verbose logging after run_server (so we print the nice intro message)
    time.sleep(0.1)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
