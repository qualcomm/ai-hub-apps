# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import random
import time
import uuid
from collections.abc import Callable, Iterator

import httpx
import qai_hub as hub
import requests
from qualcomm_device_cloud_sdk.api import qdc_api
from qualcomm_device_cloud_sdk.models import (
    ArtifactType,
    JobMode,
    JobState,
    JobSubmissionParameter,
    JobType,
    TestFramework,
)
from qualcomm_device_cloud_sdk.models.job_type_0 import JobType0 as Job
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

# States in which a job is still in progress (not yet terminal)
_RUNNING_STATES = {
    JobState.DISPATCHED.value,
    JobState.RUNNING.value,
    JobState.SETUP.value,
    JobState.SUBMITTED.value,
}

# Map from hub device names to QDC target device names
HUB_DEVICE_TO_QDC_DEVICE_MAP = {
    "Dragonwing IQ-9075 EVK": "QCS9075M",
    "Snapdragon 8 Elite QRD": "SM8750",
    "Snapdragon X Elite CRD": "SC8380XP",
}

QDC_REST_BASE_URL = "https://api.qualcomm.com/deviceloud/v1"
# QDC job limit
QDC_JOB_LIMIT = int(os.getenv("QDC_JOB_LIMIT", "1"))
# Default timeout for job status polling (in seconds)
DEFAULT_JOB_TIMEOUT = 7200  # 2 hours
# Polling interval for job status checks (in seconds)
POLL_INTERVAL = 30
# Backoff schedule for retrying a *failed* QDC call (distinct from the steady
# POLL_INTERVAL cadence used while a job is legitimately still running). Grows
# exponentially from RETRY_BACKOFF_BASE, doubling each attempt, capped at
# RETRY_BACKOFF_MAX. Keeps us from hammering a rate-limited (429) or overloaded
# (5xx) endpoint on tight upload/download loops that have no outer throttle.
RETRY_BACKOFF_BASE = 5
RETRY_BACKOFF_MAX = 300
# QDC submission fail if name exceeds this
QDC_JOB_NAME_LIMIT = 32
# Number of consecutive errors tolerated when polling status. With BASE=5 and
# MAX=300, the schedule (5, 10, 20, 40, 80, 160, 300, 300, 300, 300) sums to
# ~30 minutes — long enough to absorb a ~30-min QDC API outage while still
# probing quickly during the first few seconds.
STATUS_POLL_MAX_RETRIES = 10
# HTTP status codes that the QDC SDK can surface transiently on status polling.
# The SDK raises a bare Exception with the code embedded in the message (e.g.
# "failed with status code 403 and message: Invalid Credentials"), so we match
# on the message. 403s have been observed intermittently on otherwise-valid
# credentials; 429/5xx are the usual rate-limit / server-side blips. 400 is
# NOT included globally: it is a permanent client error for every other QDC
# call (malformed job ID, bad params), and is only retryable for submit_job
# (per-user pending-job cap rejection) — that callsite passes it explicitly
# via ``extra_retryable_codes``.
_RETRYABLE_STATUS_CODES = (403, 429, 500, 502, 503, 504)


def _matched_retryable_status_code(
    err: Exception, extra_retryable_codes: tuple[int, ...] = ()
) -> int | None:
    """Return the retryable HTTP status code embedded in err, else None.

    The QDC SDK raises a bare ``Exception`` whose message embeds the code (e.g.
    "failed with status code 403 and message: Invalid Credentials"). We return
    only the matched code so callers can log it WITHOUT echoing the rest of the
    message, which may contain server-reflected secrets or credential fragments.

    ``extra_retryable_codes`` is matched in addition to the global set; use it
    for codes that are only retryable at a specific callsite.
    """
    message = str(err)
    for code in _RETRYABLE_STATUS_CODES + extra_retryable_codes:
        if f"status code {code}" in message:
            return code
    return None


# Note: We only have to do this hack because the QDC API re-throws as bare
# exceptions. We have asked them not to do this (so revisit after 0.2.3)
# https://jira-dc.qualcomm.com/jira/browse/QDC-5475
def _unwrap_causes(err: BaseException) -> Iterator[BaseException]:
    """Yield ``err`` and every exception in its __cause__/__context__ chain."""
    seen: set[int] = set()
    cur: BaseException | None = err
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


def _transient_network_error_name(err: Exception) -> str | None:
    """Return the underlying transient-network error's type name, else None.

    The QDC SDK's ``try_call`` flattens the real failure into a bare
    ``Exception`` (``raise Exception(msg) from e``), so the network type is only
    visible by walking the cause chain. DNS failures, for instance, surface as
    ``socket.gaierror`` ("Temporary failure in name resolution") — an
    ``OSError`` subclass — wrapped under httpx/httpcore ConnectError. We return
    only the type name so callers can log it WITHOUT echoing the message, which
    may carry server-reflected secrets or credential fragments.

    httpx.TransportError and its subclasses (ConnectError, ReadError, *Timeout,
    RemoteProtocolError, etc.) are all transport-layer errors safe to retry.
    HTTPStatusError is a sibling, not a TransportError — non-retryable status
    codes are still filtered by _matched_retryable_status_code.
    """
    for cause in _unwrap_causes(err):
        if isinstance(cause, (OSError, ConnectionError, TimeoutError)):
            return type(cause).__name__
        if isinstance(cause, httpx.TransportError):
            return type(cause).__name__
    return None


def _retry_reason(err: Exception, extra_retryable_codes: tuple[int, ...]) -> str | None:
    """Return a redacted reason to retry ``err``, or None if it is not retryable.

    The QDC SDK flattens the real failure into a bare Exception, so we inspect
    both the cause chain (transient network errors like a DNS gaierror) and the
    message (embedded retryable status codes). We surface only the matched type
    name / status code -- never the raw message, which the SDK may populate with
    credential fragments or secrets.
    """
    net_err = _transient_network_error_name(err)
    if net_err is not None:
        return f"transient network error ({net_err})"
    code = _matched_retryable_status_code(err, extra_retryable_codes)
    if code is not None:
        return f"transient status code {code}"
    return None


def retry_qdc(
    extra_retryable_codes: tuple[int, ...] = (),
) -> Callable[[Callable], Callable]:
    """Tenacity ``@retry`` policy for QDC SDK calls that are safe to repeat.

    Covers transient network blips (DNS resolution failures, dropped
    connections) as well as transient HTTP status errors the QDC SDK raises as a
    bare Exception (the global ``_RETRYABLE_STATUS_CODES`` plus any callsite-
    specific codes passed via ``extra_retryable_codes``). A genuinely fatal error
    is not retried, and a still-transient error surfaces unchanged after
    ``STATUS_POLL_MAX_RETRIES`` attempts (``reraise=True`` -- callers see the
    original exception, not tenacity's ``RetryError``). Delays use capped
    exponential backoff (``wait_exponential`` from ``RETRY_BACKOFF_BASE``,
    doubling each attempt, capped at ``RETRY_BACKOFF_MAX``) so we don't hammer a
    rate-limited or overloaded endpoint.

    Applied to QDC SDK calls that talk to the network and are idempotent: status
    polling (``get_job_status``), log retrieval/download (``get_job_log_files``,
    ``download_job_log_files``), artifact upload (``upload_file``, whose only
    commit point is the final chunk and which returns the uuid of the successful
    attempt), and job submission (``submit_job``).

    Parameters
    ----------
    extra_retryable_codes
        Additional HTTP status codes that are retryable for this callsite only
        (matched in addition to the global ``_RETRYABLE_STATUS_CODES``).

    Returns
    -------
    Callable[[Callable], Callable]
        A ``tenacity.retry`` decorator carrying the QDC retry policy.
    """

    def _is_retryable(err: BaseException) -> bool:
        return isinstance(err, Exception) and (
            _retry_reason(err, extra_retryable_codes) is not None
        )

    def _log_before_sleep(retry_state: RetryCallState) -> None:
        err = retry_state.outcome.exception()
        assert isinstance(err, Exception)
        # Log only the matched type/status code, never the raw message, which
        # the SDK may populate with credential fragments or secrets.
        reason = _retry_reason(err, extra_retryable_codes)
        name = retry_state.fn.__name__ if retry_state.fn else "QDC call"
        ident = ", ".join(str(arg)[:30] for arg in retry_state.args)
        target = f"{name}({ident})" if ident else name
        print(
            f"[QDC retry] {target} failed with {reason}; attempt "
            f"{retry_state.attempt_number}/{STATUS_POLL_MAX_RETRIES}, retrying "
            f"in {retry_state.next_action.sleep:g}s."
        )

    return retry(
        stop=stop_after_attempt(STATUS_POLL_MAX_RETRIES),
        wait=wait_exponential(
            multiplier=RETRY_BACKOFF_BASE, exp_base=2, max=RETRY_BACKOFF_MAX
        ),
        retry=retry_if_exception(_is_retryable),
        before_sleep=_log_before_sleep,
        reraise=True,
    )


class QDCDevice:
    """Wraps a QAI Hub device and exposes QDC-specific properties."""

    def __init__(self, device: str) -> None:
        """
        Parameters
        ----------
        device
            QAI Hub device name. The latest matching device is selected.
        """
        self.device = hub.get_devices(device)[-1]
        self.device_attributes = getattr(self.device, "attributes", [])

    @property
    def hexagon_version(self) -> str:
        """Hexagon version string parsed from the device's hub attributes."""
        htp_version = None
        for attr in self.device_attributes:
            if "hexagon" in attr:
                htp_version = attr.split(":")[-1]
        assert htp_version is not None, (
            f"Hexagon/HTP version not found in device attributes. "
            f"Device: {getattr(self.device, 'name', 'unknown')!r}. "
            f"Attributes: {self.device_attributes!r}"
        )
        return htp_version

    @property
    def windows_platform(self) -> bool:
        """True if the device runs Windows, based on hub attributes."""
        for attr in self.device_attributes:
            if "os" in attr and attr.endswith("windows"):
                return True
        return False

    @property
    def mobile_platform(self) -> bool:
        """True if the device is a phone form factor, based on hub attributes."""
        for attr in self.device_attributes:
            if "format" in attr and attr.endswith("phone"):
                return True
        return False

    @property
    def iot_platform(self) -> bool:
        """True if the device is an IoT form factor, based on hub attributes."""
        for attr in self.device_attributes:
            if "format" in attr and attr.endswith("iot"):
                return True
        return False

    @property
    def qdc_name(self) -> str:
        """QDC target device name corresponding to the hub device name."""
        return HUB_DEVICE_TO_QDC_DEVICE_MAP[self.device.name]

    @property
    def test_framework(self) -> TestFramework:
        """QDC test framework appropriate for this device's platform."""
        if self.windows_platform:
            return TestFramework.POWERSHELL
        if self.iot_platform:
            return TestFramework.BASH
        return TestFramework.APPIUM


class QDCJobs:
    """
    Base class for QDC job handlers.

    Provides shared functionality for submitting jobs, polling status,
    and retrieving logs. Subclasses implement their own artifact creation
    and metrics computation methods specific to their workload type.
    """

    def __init__(
        self,
        *,
        api_key: str,
        app_name_header: str,
    ) -> None:
        """
        Parameters
        ----------
        api_key
            API key for QDC authentication.
        app_name_header
            Application name header for QDC API client.
        """
        self.client = qdc_api.get_public_api_client_using_api_key(
            api_key_header=api_key,
            app_name_header=app_name_header,
            on_behalf_of_header="ai_hub_models",
            client_type_header="Python",
        )
        self._api_key = api_key
        self._app_name_header = app_name_header
        self._session = requests.Session()
        self._session.headers.update(
            {
                "accept": "application/json",
                "Authorization": api_key,
                "X-QCOM-TokenType": "apikey",
                "X-QCOM-AppName": app_name_header,
                "X-QCOM-ClientType": "appName",
            }
        )

    @retry_qdc()
    def get_job(self, job_id: str) -> Job:
        """Fetch job details from the QDC REST API.

        Parameters
        ----------
        job_id
            ID of the job to retrieve.

        Returns
        -------
        job : Job
            Job object constructed from the ``GET /jobs/{job_id}`` response.

        Raises
        ------
        requests.HTTPError
            If the API returns a non-2xx status code.
        """
        # Currently there is no support for get_job in the QDC Python API
        # This is an interim solution until QDC-5417 is resolved
        # https://jira-dc.qualcomm.com/jira/browse/QDC-5417
        response = self._session.get(
            f"{QDC_REST_BASE_URL}/jobs/{job_id}",
            headers={"X-QCOM-TracingId": str(uuid.uuid4())},
        )
        response.raise_for_status()
        return Job.from_dict(response.json())

    @retry_qdc()
    def get_job_status(self, job_id: str) -> str:
        """Fetch a job's lifecycle state, retrying transient QDC errors."""
        return qdc_api.get_job_status(self.client, job_id)

    def status(self, job_id: str, timeout: int = DEFAULT_JOB_TIMEOUT) -> str:
        """
        Poll and return the terminal status for a job (e.g., Completed/Canceled).

        Parameters
        ----------
        job_id
            ID of the job to monitor.
        timeout
            Maximum time to wait for job completion in seconds.
            Defaults to DEFAULT_JOB_TIMEOUT (2 hours).

        Returns
        -------
        job_status : str
            Final status of the job.

        Raises
        ------
        TimeoutError
            If job does not complete within the timeout period.
        """
        job_status = None
        elapsed = 0
        while elapsed < timeout:
            job_status = self.get_job_status(job_id)
            if job_status not in _RUNNING_STATES:
                time.sleep(POLL_INTERVAL)
                return job_status
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL

        job_status = self.get_job_status(job_id)
        if job_status in {"Completed", "Canceled", "Failed", "Error", "Aborted"}:
            return job_status
        qdc_api.abort_job(self.client, job_id)
        raise TimeoutError(
            f"Job {job_id} did not complete within {timeout} seconds. "
            f"Last status: {job_status}"
        )

    def result(self, job_id: str) -> str | None:
        """Return the terminal result of a job (e.g. "Successful"/"Unsuccessful").

        ``status()`` reports only the lifecycle ``state`` ("Completed"); a job can
        reach "Completed" yet still have failed device-side execution, which shows
        up in the separate ``result`` field.

        Parameters
        ----------
        job_id
            ID of the job to query.

        Returns
        -------
        result : str | None
            The job's terminal result (e.g. "Successful"/"Unsuccessful"), or None
            if the field is absent.
        """
        job = self.get_job(job_id)
        return getattr(job, "result", None)

    def get_active_jobs(self) -> list[Job]:
        """Return all currently active (non-terminal) jobs for this user.

        Returns
        -------
        active_jobs : list[Job]
            Jobs whose state is in ``_RUNNING_STATES``.
        """
        # get_jobs_list returns all submitted jobs (latest first). The service allows at most
        # 3 concurrent jobs; we fetch 10 as a safety buffer to ensure we don't miss any active ones.
        jobs = qdc_api.get_jobs_list(self.client, 0, 10)
        if jobs is None:
            raise ValueError(
                "Failure in `get_jobs_list`. Could not get job lists for user"
            )

        return [job for job in jobs.data if job.state in _RUNNING_STATES]

    def submit_automated_job(
        self,
        qdc_device: QDCDevice,
        job_artifacts: list[str],
        entry_script: str | None,
        job_name: str = "QDC Automated Job",
        timeout: int = DEFAULT_JOB_TIMEOUT,
    ) -> str:
        """
        Submit an automated application job to QDC and return its job_id.

        Parameters
        ----------
        qdc_device
            QDCDevice instance for the target device.
        job_artifacts
            List of artifact IDs/descriptors to attach to the job.
        entry_script
            Optional entry script path for the job.
        job_name
            Name of the job to submit.
        timeout
            Maximum time to wait for a job slot to become available in seconds.
            Defaults to DEFAULT_JOB_TIMEOUT (2 hours).

        Returns
        -------
        job_id : str
            The submitted job's ID.
        """
        elapsed = 0
        while elapsed < timeout:
            if len(self.get_active_jobs()) < QDC_JOB_LIMIT:
                # jitter: wait POLL_INTERNAL + random(0, 10) to avoid TOCTOU race condition
                time.sleep(POLL_INTERVAL + random.randint(0, 10))
                if len(self.get_active_jobs()) < QDC_JOB_LIMIT:
                    break
            print(
                f"Job is waiting as the service is at capacity, "
                f"waiting for {POLL_INTERVAL} seconds."
            )
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL

        if elapsed >= timeout:
            raise TimeoutError(
                f"Job {job_name} did not start within {timeout}s because the service is at capacity (>={QDC_JOB_LIMIT} active jobs). "
            )

        target_id = qdc_api.get_target_id(self.client, qdc_device.qdc_name)
        return self._submit_job(
            target_id, job_name, qdc_device.test_framework, entry_script, job_artifacts
        )

    # 400 is retryable only here: it is how QDC rejects submission when the
    # per-user pending-job cap is hit (a transient condition), unlike the
    # permanent 400s other calls get for malformed IDs/params.
    @retry_qdc(extra_retryable_codes=(400,))
    def _submit_job(
        self,
        target_id: str,
        job_name: str,
        test_framework: TestFramework,
        entry_script: str | None,
        job_artifacts: list[str],
    ) -> str:
        """Submit a job to QDC, retrying transient errors (incl. the 400 cap)."""
        return qdc_api.submit_job(
            public_api_client=self.client,
            target_id=target_id,
            job_name=job_name[:QDC_JOB_NAME_LIMIT],
            external_job_id="ExJobId001",
            job_type=JobType.AUTOMATED,
            job_mode=JobMode.APPLICATION,
            timeout=600,
            test_framework=test_framework,
            entry_script=entry_script,
            job_artifacts=job_artifacts,
            monkey_events=None,
            monkey_session_timeout=None,
            job_parameters=[JobSubmissionParameter.WIFIENABLED],
        )

    @retry_qdc()
    def get_job_log_upload_status(self, job_id: str) -> str:
        """Fetch a job's log-upload status, retrying transient QDC errors."""
        return qdc_api.get_job_log_upload_status(self.client, job_id)

    def log_upload_status(
        self, job_id: str, timeout: int = DEFAULT_JOB_TIMEOUT
    ) -> None:
        """
        Poll until job logs are uploaded (completed/failed).

        Parameters
        ----------
        job_id
            ID of the job to monitor.
        timeout
            Maximum time to wait for log upload in seconds.
            Defaults to DEFAULT_JOB_TIMEOUT (2 hours).

        Raises
        ------
        TimeoutError
            If logs are not uploaded within the timeout period.
        """
        status = None
        elapsed = 0
        while elapsed <= timeout:
            status = self.get_job_log_upload_status(job_id).lower()
            if status not in {"completed", "failed"}:
                print(
                    f"Job is completed and the server is uploading logs, "
                    f"waiting for {POLL_INTERVAL} seconds."
                )
                time.sleep(POLL_INTERVAL)
                elapsed += POLL_INTERVAL
            else:
                print("Job logs are uploaded.")
                return

        raise TimeoutError(
            f"Log upload for job {job_id} did not complete within {timeout} seconds. "
            f"Last status: {status}"
        )

    @retry_qdc()
    def get_job_log_files(self, job_id: str) -> list:
        """Wrapper to get job log files using the QDC API.

        Parameters
        ----------
        job_id
            ID of the job to retrieve logs for.

        Returns
        -------
        job_log_files: list
            List of job log files.
        """
        return qdc_api.get_job_log_files(self.client, job_id)

    @retry_qdc()
    def download_job_log_files(self, filename: str, target_path: str) -> None:
        """Download job log files from QDC.

        Parameters
        ----------
        filename
            Name of the log file to download.
        target_path
            Local path to save the downloaded file.
        """
        # Safe to retry as-is: the SDK fetches the whole file into memory inside
        # its own request, and only opens target_path (mode 'wb', single write)
        # after a 200 response. A failed attempt never touches the file, and a
        # later success truncates it -- so no partial/corrupt file can persist.
        qdc_api.download_job_log_files(self.client, filename, target_path)

    @retry_qdc()
    def upload_file(self, file_path: str, artifact_type: ArtifactType) -> str:
        """Upload a file to QDC.

        Parameters
        ----------
        file_path
            Path to the file to upload.
        artifact_type
            Type of artifact being uploaded.

        Returns
        -------
        artifact_id: str
            ID of the uploaded artifact.
        """
        # The SDK uploads large bundles in chunks (start -> continue -> end); the
        # artifact only commits at the final `end_upload`, and upload_file returns
        # the uuid of the *successful* attempt -- so a retry after a mid-transfer
        # drop at worst leaves an orphaned, unreferenced partial artifact server-
        # side (a benign leak), never a duplicate job. We therefore still retry
        # network errors here. The 403 actually observed in CI lands on the first
        # call (post_artifacts_start_upload), before anything is committed.
        return qdc_api.upload_file(self.client, file_path, artifact_type)
