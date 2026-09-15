# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Shared helpers for running ONNX models via the ONNX Runtime QNN Execution Provider.

Supports both onnxruntime-qnn release lines:

- ``>=2.0.0`` (plugin EP): a separate ``onnxruntime_qnn`` package registered
  into a plain ``onnxruntime`` install via ``register_execution_provider_library``
  / ``add_provider_for_devices``.
- ``<2.0.0`` (classic EP): ``onnxruntime-qnn`` *is* the ``onnxruntime`` package
  (a QNN-enabled build of it); ``QNNExecutionProvider`` is selected directly via
  the ``providers=`` argument.

Either way, the QNN backend libraries are bundled with the installed package, so
no external QAIRT SDK install is required. This mirrors how ``qai_hub_models``
creates its sessions -- no DLL search path or ``ADSP_LIBRARY_PATH`` is set,
and the backend is never pointed at a separately installed QAIRT (which would risk
a version mismatch against the bundled runtime).

Precompiled ``*.onnx`` exports (``runtime: precompiled_qnn_onnx``) apps embed a QAIRT/QNN
version at export time via their EPContext wrapper. The installed onnxruntime-qnn's bundled
 QNN backend must be at least that version to load them -- an older bundled backend fails with
``LoadCachedQnnContextFromBuffer`` error 5000/1000. Keep this package pinned to
a recent release if new exports start failing to load or pin qaihm_version for the app.
"""

from __future__ import annotations

import functools

import onnxruntime as ort

try:
    import onnxruntime_qnn as qnn  # plugin EP (onnxruntime-qnn >= 2.0.0)
except ImportError:
    qnn = None  # classic EP (onnxruntime-qnn < 2.0.0): QNN is built into `ort` itself

# EP options mirroring qai_hub_models' aihub_defaults() for the QNN EP.
_QNN_PROVIDER_OPTIONS: dict[str, str] = {
    "enable_htp_fp16_precision": "1",
    "htp_performance_mode": "burst",
    "htp_graph_finalization_optimization_mode": "3",
    "offload_graph_io_quantization": "1",
}


@functools.cache
def _register_qnn_ep() -> None:
    """Register the plugin QNN EP library. Cached so it runs once per process."""
    ort.register_execution_provider_library(qnn.EP_NAME, qnn.get_library_path())


def _qnn_devices() -> list[ort.OrtEpDevice]:
    """Register the plugin QNN EP library (once per process) and return its devices."""
    _register_qnn_ep()
    return [d for d in ort.get_ep_devices() if d.ep_name == qnn.EP_NAME]


def open_qnn_session(model_path: str) -> ort.InferenceSession:
    """Open an ORT session for ``model_path`` on the QNN Execution Provider.

    Works for both a plain graph (compiled for the HTP on load) and an
    AI-Hub-exported EPContext wrapper around a precompiled QNN context binary
    (``runtime: precompiled_qnn_onnx`` apps) -- the QNN EP handles either
    directly, with no separate compile-and-cache step needed.

    Parameters
    ----------
    model_path
        Path to the ONNX model.

    Returns
    -------
    ort.InferenceSession
        An initialized ONNX Runtime session using the QNN Execution Provider,
        backed by the QNN backend bundled with the installed onnxruntime-qnn.
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if qnn is not None:
        # Plugin EP (onnxruntime-qnn >= 2.0.0).
        so.add_provider_for_devices(_qnn_devices(), _QNN_PROVIDER_OPTIONS)
        return ort.InferenceSession(model_path, sess_options=so)

    # Classic EP (onnxruntime-qnn < 2.0.0): QNN is selected directly, with the
    # bundled QnnHtp.dll resolved via the bare filename (no QAIRT path needed).
    return ort.InferenceSession(
        model_path,
        sess_options=so,
        providers=["QNNExecutionProvider"],
        provider_options=[{**_QNN_PROVIDER_OPTIONS, "backend_path": "QnnHtp.dll"}],
    )
