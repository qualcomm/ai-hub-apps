# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import time

import qai_hub as hub
from qualcomm_device_cloud_sdk.api import qdc_api
from qualcomm_device_cloud_sdk.models import (
    ArtifactType,
    JobMode,
    JobState,
    JobSubmissionParameter,
    JobType,
    TestFramework,
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
    "Samsung Galaxy S25": "SM8750",
}

# Default timeout for job status polling (in seconds)
DEFAULT_JOB_TIMEOUT = 7200  # 2 hours
# Polling interval for job status checks (in seconds)
POLL_INTERVAL = 30
# QDC submission fail if name exceeds this
QDC_JOB_NAME_LIMIT = 32


class QDCDevice:
    def __init__(self, device: str) -> None:
        self.device = hub.get_devices(device)[-1]
        self.device_attributes = getattr(self.device, "attributes", [])

    @property
    def hexagon_version(self) -> str:
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
        for attr in self.device_attributes:
            if "os" in attr and attr.endswith("windows"):
                return True
        return False

    @property
    def mobile_platform(self) -> bool:
        for attr in self.device_attributes:
            if "format" in attr and attr.endswith("phone"):
                return True
        return False

    @property
    def iot_platform(self) -> bool:
        for attr in self.device_attributes:
            if "format" in attr and attr.endswith("iot"):
                return True
        return False

    @property
    def qdc_name(self) -> str:
        return HUB_DEVICE_TO_QDC_DEVICE_MAP[self.device.name]

    @property
    def test_framework(self) -> TestFramework:
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

    def status(self, job_id: str, timeout: int = DEFAULT_JOB_TIMEOUT) -> str:
        """
        Poll and return the terminal status for a job (e.g., Completed/Canceled).

        Parameters
        ----------
        job_id
            ID of the job to monitor.
        timeout
            Maximum time to wait for job completion in seconds.
            Defaults to DEFAULT_JOB_TIMEOUT (1 hour).

        Returns
        -------
        job_status: str
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

    def submit_automated_job(
        self,
        qdc_device: QDCDevice,
        job_artifacts: list[str],
        entry_script: str | None,
        job_name: str = "QDC Automated Job",
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

        Returns
        -------
        job_id: str
            The submitted job's ID.
        """
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
            Defaults to DEFAULT_JOB_TIMEOUT (1 hour).

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
