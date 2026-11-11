# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import time


class FpsCounter:
    """Tracks the current frames-per-second of a processing loop.

    Call :meth:`tick` once per frame and read :meth:`fps` for the rate. By
    default :meth:`tick` also prints the rate every ``report_interval`` seconds.
    """

    def __init__(self, smoothing: float = 0.1, report_interval: float = 1.0) -> None:
        """Create a counter.

        Parameters
        ----------
        smoothing
            Weight in (0, 1] applied to each new sample in the moving average.
            Larger values track changes faster; smaller values are smoother.
        report_interval
            How often, in seconds, :meth:`tick` prints the current rate.
            Set to <= 0 to disable printing.
        """
        self.smoothing = smoothing
        self.report_interval = report_interval
        self._last_tick: float | None = None
        self._last_report: float | None = None
        self._fps = 0.0

    def tick(self) -> None:
        """Record one processed frame and print the rate if a report is due."""
        now = time.perf_counter()
        if self._last_tick is not None:
            delta = now - self._last_tick
            if delta > 0:
                instant = 1.0 / delta
                self._fps = (
                    self.smoothing * instant + (1.0 - self.smoothing) * self._fps
                    if self._fps
                    else instant
                )
        self._last_tick = now

        if self.report_interval > 0:
            if self._last_report is None:
                self._last_report = now
            elif now - self._last_report >= self.report_interval:
                self._last_report = now
                print(f"FPS: {self._fps:.1f}", flush=True)

    def fps(self) -> float:
        """Return the current rate in frames per second.

        Returns
        -------
        float
            The smoothed frames-per-second, or ``0.0`` before the second tick.
        """
        return self._fps
