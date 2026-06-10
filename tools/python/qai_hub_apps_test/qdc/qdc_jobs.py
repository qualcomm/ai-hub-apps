# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import random
import time
import uuid

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

# Default timeout for job status polling (in seconds)
DEFAULT_JOB_TIMEOUT = 7200  # 2 hours
# Polling interval for job status checks (in seconds)
POLL_INTERVAL = 30
# QDC submission fail if name exceeds this
QDC_JOB_NAME_LIMIT = 32
# QDC job limit
QDC_JOB_LIMIT = int(os.getenv("QDC_JOB_LIMIT", "1"))


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
        """Hexagon DSP version string parsed from the device's hub attributes."""
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
            job_status = qdc_api.get_job_status(self.client, job_id)
            if job_status not in _RUNNING_STATES:
                time.sleep(POLL_INTERVAL)
                return job_status
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL

        job_status = qdc_api.get_job_status(self.client, job_id)
        if job_status in {"Completed", "Canceled", "Failed", "Error", "Aborted"}:
            return job_status
        qdc_api.abort_job(self.client, job_id)
        raise TimeoutError(
            f"Job {job_id} did not complete within {timeout} seconds. "
            f"Last status: {job_status}"
        )

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

        return qdc_api.submit_job(
            public_api_client=self.client,
            target_id=qdc_api.get_target_id(self.client, qdc_device.qdc_name),
            job_name=job_name[:QDC_JOB_NAME_LIMIT],
            external_job_id="ExJobId002",
            job_type=JobType.AUTOMATED,
            job_mode=JobMode.APPLICATION,
            timeout=600,
            test_framework=qdc_device.test_framework,
            entry_script=entry_script,
            job_artifacts=job_artifacts,
            monkey_events=None,
            monkey_session_timeout=None,
            job_parameters=[JobSubmissionParameter.WIFIENABLED],
        )

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
            status = qdc_api.get_job_log_upload_status(self.client, job_id).lower()
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

    def download_job_log_files(self, filename: str, target_path: str) -> None:
        """Download job log files from QDC.

        Parameters
        ----------
        filename
            Name of the log file to download.
        target_path
            Local path to save the downloaded file.
        """
        qdc_api.download_job_log_files(self.client, filename, target_path)

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
        return qdc_api.upload_file(self.client, file_path, artifact_type)
