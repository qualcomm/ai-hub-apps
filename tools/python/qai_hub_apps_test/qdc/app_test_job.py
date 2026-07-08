# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
import time
import zipfile
from abc import ABC, abstractmethod

from qualcomm_device_cloud_sdk.models import ArtifactType

from qai_hub_apps_test.qdc.qdc_jobs import (
    HUB_DEVICE_TO_QDC_DEVICE_MAP,
    POLL_INTERVAL,
    QDCDevice,
    QDCJobs,
)

TEXT_LOG_EXTENSIONS = (".log", ".stdout", ".txt", ".json")


def create_zip(zip_path: str, source_dir: str | os.PathLike) -> None:
    """Create a zip archive from source_dir at zip_path."""
    if isinstance(source_dir, os.PathLike):
        source_dir = str(source_dir)

    files_to_zip = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, source_dir)
            files_to_zip.append((file_path, arcname))

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path, arcname in files_to_zip:
            zf.write(file_path, arcname)


class AppTestArtifactHandler(ABC):
    """Abstract base class for app-test artifact handlers."""

    @abstractmethod
    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        app_dir: os.PathLike | str,
        dest_dir: os.PathLike | str,
    ) -> str:
        """Create artifact bundle and return path to the zip file."""
        raise NotImplementedError

    @property
    @abstractmethod
    def entry_script(self) -> str | None:
        raise NotImplementedError


class AppTestLinuxArtifactHandler(AppTestArtifactHandler):
    def __init__(self, use_docker: bool = False) -> None:
        self.use_docker = use_docker

    @property
    def entry_script(self) -> str:
        return "/bin/bash /data/local/tmp/TestContent/run_linux.sh"

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        app_dir: os.PathLike | str,
        dest_dir: os.PathLike | str,
    ) -> str:
        """Build the test bundle directory and return the path to the zip archive.

        Copies ``run_linux.sh`` from ``device_scripts/`` into ``dest_dir``,
        substituting placeholders in the shell script. The app directory (which
        must contain a ``Dockerfile`` when ``use_docker=True``) is copied in as
        an ``app/`` subdirectory. The whole ``dest_dir`` is then zipped into
        ``test.zip`` one level above it.

        Parameters
        ----------
        curr_dirname
            Directory containing the ``device_scripts/`` folder.
        app_dir
            Directory of the fetched app (output of ``qai-hub-apps fetch``).
        dest_dir
            Staging directory where the bundle contents are assembled.

        Returns
        -------
        str
            Absolute path to the created ``test.zip`` archive.
        """
        dest_script = os.path.join(dest_dir, "run_linux.sh")
        shutil.copy(
            os.path.join(curr_dirname, "device_scripts", "run_linux.sh"),
            dest_script,
        )
        with open(dest_script, encoding="utf-8") as f:
            content = f.read()
        with open(dest_script, "w", encoding="utf-8") as f:
            f.write(
                content.replace("<<RUN_COMMAND>>", "bash test.sh").replace(
                    "<<USE_DOCKER>>", "true" if self.use_docker else "false"
                )
            )

        shutil.copytree(app_dir, os.path.join(dest_dir, "app"))

        if self.use_docker and not os.path.isfile(
            os.path.join(dest_dir, "app", "Dockerfile")
        ):
            raise FileNotFoundError(
                "use_docker=True but no 'Dockerfile' found in the app bundle. "
                "Ensure the app declares 'base_docker' in info.yaml and was "
                "bundled with bundle_app() before submission."
            )

        zip_path = os.path.join(os.path.dirname(dest_dir), "test.zip")
        create_zip(zip_path, dest_dir)
        return zip_path


class AppTestAndroidArtifactHandler(AppTestArtifactHandler):
    @property
    def entry_script(self) -> str | None:
        return None

    def get_instrumentation_runner(self, app_dir: str | os.PathLike) -> str:
        """Parse applicationId and testInstrumentationRunner from build.gradle."""
        build_gradle = os.path.join(app_dir, "build.gradle")
        with open(build_gradle, encoding="utf-8") as f:
            content = f.read()

        app_id = re.search(r"""applicationId\s+["']([^"']+)["']""", content)
        runner = re.search(r"""testInstrumentationRunner\s+["']([^"']+)["']""", content)

        if not app_id or not runner:
            raise RuntimeError(
                f"Could not parse applicationId or testInstrumentationRunner from {build_gradle}"
            )
        return f"{app_id.group(1)}.test/{runner.group(1)}"

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        app_dir: os.PathLike | str,
        dest_dir: os.PathLike | str,
    ) -> str:
        test_folder = os.path.join(dest_dir, "tests")
        os.makedirs(test_folder, exist_ok=True)

        # Parse instrumentation runner from build.gradle
        instrumentation_runner = self.get_instrumentation_runner(app_dir)

        dest_script = os.path.join(test_folder, "test_app.py")

        # Copy 'run_android.py' and rename it to 'test_app.py' since pytest looks for files starting with 'test_'.
        shutil.copy(
            os.path.join(curr_dirname, "device_scripts", "run_android.py"),
            dest_script,
        )

        with open(dest_script, encoding="utf-8") as f:
            content = f.read()
        with open(dest_script, "w", encoding="utf-8") as f:
            f.write(
                content.replace("<<INSTRUMENTATION_RUNNER>>", instrumentation_runner)
            )

        # Create an empty requirements.txt
        open(os.path.join(dest_dir, "requirements.txt"), "w").close()

        copied_app_dir = os.path.join(dest_dir, "app")
        shutil.copytree(app_dir, copied_app_dir)

        # Keep only build/outputs/
        for item in os.listdir(copied_app_dir):
            item_path = os.path.join(copied_app_dir, item)
            if item == "build":
                for build_item in os.listdir(item_path):
                    if build_item != "outputs":
                        build_item_path = os.path.join(item_path, build_item)
                        shutil.rmtree(build_item_path) if os.path.isdir(
                            build_item_path
                        ) else os.unlink(build_item_path)
            else:
                shutil.rmtree(item_path) if os.path.isdir(item_path) else os.unlink(
                    item_path
                )

        zip_path = os.path.join(os.path.dirname(dest_dir), "test.zip")
        create_zip(zip_path, dest_dir)
        return zip_path


class AppTestWindowsArtifactHandler(AppTestArtifactHandler):
    def __init__(self, use_docker: bool = False) -> None:
        self.use_docker = use_docker

    @property
    def entry_script(self) -> str:
        return "C:\\Temp\\TestContent\\run_windows.ps1"

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        app_dir: os.PathLike | str,
        dest_dir: os.PathLike | str,
    ) -> str:
        dest_script = os.path.join(dest_dir, "run_windows.ps1")
        shutil.copy(
            os.path.join(curr_dirname, "device_scripts", "run_windows.ps1"),
            dest_script,
        )
        with open(dest_script, encoding="utf-8") as f:
            content = f.read()
        with open(dest_script, "w", encoding="utf-8") as f:
            f.write(
                content.replace(
                    "<<USE_DOCKER>>", "true" if self.use_docker else "false"
                ).replace("<<RUN_COMMAND>>", "powershell -File test.ps1")
            )

        shutil.copytree(app_dir, os.path.join(dest_dir, "app"))

        zip_path = os.path.join(os.path.dirname(dest_dir), "test.zip")
        create_zip(zip_path, dest_dir)
        return zip_path


class AppTestQDCJobs(QDCJobs):
    """QDC job handler for generic app on-device testing."""

    def _get_artifact_handler(
        self, qdc_device: QDCDevice, use_docker: bool = False
    ) -> AppTestArtifactHandler:
        if qdc_device.iot_platform:
            return AppTestLinuxArtifactHandler(use_docker=use_docker)
        if qdc_device.mobile_platform:
            return AppTestAndroidArtifactHandler()
        if qdc_device.windows_platform:
            # windows containers are not supported on arm64
            return AppTestWindowsArtifactHandler(use_docker=False)
        raise NotImplementedError(
            f"On-device app testing is not yet supported for this platform. "
            f"Device: {qdc_device.device.name!r}"
        )

    def add_job_artifacts(
        self,
        qdc_device: QDCDevice,
        app_dir: str | os.PathLike,
        use_docker: bool = False,
        save_bundle_dir: str | os.PathLike | None = None,
    ) -> tuple[list[str], str | None]:
        """Prepare and upload app artifacts for job submission.

        Parameters
        ----------
        qdc_device
            QDCDevice instance for the target device.
        app_dir
            Directory of the fetched app (output of ``qai-hub-apps fetch``).
        use_docker
            If True, run the app inside a Docker container on the device.
        save_bundle_dir
            If set, copy the test.zip bundle to this directory before uploading.

        Returns
        -------
        job_artifacts : list[str]
            List of artifact IDs returned by QDC upload.
        entry_script : str | None
            Entry script path used by the test framework.
        """
        curr_dirname = os.path.dirname(os.path.abspath(__file__))
        artifact_handler = self._get_artifact_handler(qdc_device, use_docker)

        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = artifact_handler.create_artifact(
                curr_dirname,
                app_dir,
                tmpdirname,
            )
            upload_response = self.upload_file(zip_path, ArtifactType.TESTSCRIPT)
            if save_bundle_dir is not None:
                os.makedirs(save_bundle_dir, exist_ok=True)
                shutil.copy(zip_path, save_bundle_dir)
            if os.path.exists(zip_path):
                os.unlink(zip_path)

        return [upload_response], artifact_handler.entry_script


def submit_app_bundle_to_qdc_device(
    api_token: str,
    device: str,
    app_dir: str | os.PathLike,
    job_name: str = "App Test",
    use_docker: bool = False,
    save_bundle_dir: str | os.PathLike | None = None,
) -> bool:
    """Submit a fetched app bundle to QDC for on-device execution.

    Parameters
    ----------
    api_token
        API token for QDC authentication.
    device
        Hub device name to run the job on (must be a key in HUB_DEVICE_TO_QDC_DEVICE_MAP).
    app_dir
        Directory of the fetched app (output of ``qai-hub-apps fetch``).
    job_name
        Name for the QDC job.
    use_docker
        If True, build and run the app inside a Docker container on the device
        using the platform specific ``Dockerfile`` base image.
    save_bundle_dir
        If set, copy the test.zip bundle to this directory before uploading.

    Returns
    -------
    success : bool
        True if the job completed successfully, False otherwise.
    """
    qdc_device = QDCDevice(device)
    app_job = AppTestQDCJobs(
        api_key=api_token,
        app_name_header="AppTestQDCJobApp",
    )

    job_artifacts, entry_script = app_job.add_job_artifacts(
        qdc_device, app_dir, use_docker, save_bundle_dir
    )

    job_id = app_job.submit_automated_job(
        qdc_device, job_artifacts, entry_script, job_name=job_name
    )
    if job_id is None:
        raise RuntimeError("Job submission failed.")

    print(f"Submitted QDC job with ID: {job_id}")
    job_status = app_job.status(job_id)
    print(f"QDC job {job_id} completed with status: {job_status}")

    job = app_job.get_job(job_id)
    print(f"QDC job {job_id} test finished with status: {job.result.value}")
    succeeded = job.result == "Successful"

    if not succeeded and job_status == "Completed":
        app_job.log_upload_status(job_id)
        job_log_files = app_job.get_job_log_files(job_id)
        time.sleep(POLL_INTERVAL)
        if job_log_files:
            with tempfile.TemporaryDirectory() as tmpdirname:
                for job_log in job_log_files:
                    filename = job_log.filename
                    if not filename.lower().endswith(TEXT_LOG_EXTENSIONS):
                        continue  # skip .mp4 and other non-text artifacts

                    base_name = os.path.basename(filename)
                    zip_path = os.path.join(tmpdirname, f"{base_name}.zip")
                    try:
                        app_job.download_job_log_files(filename, zip_path)
                        with zipfile.ZipFile(zip_path) as zf:
                            contents = "\n".join(
                                zf.read(name).decode("utf-8", errors="replace")
                                for name in zf.namelist()
                            )
                        print(f"::group::QDC log: {base_name}")
                        print(contents)
                        print("::endgroup::")
                    except Exception as e:  # don't let a bad log mask the failure
                        print(f"::warning::Could not read QDC log {base_name}: {e}")

    return succeeded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit a fetched app bundle to QDC for on-device testing."
    )
    parser.add_argument(
        "--api-token",
        type=str,
        required=True,
        help="API token for QDC authentication.",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        choices=HUB_DEVICE_TO_QDC_DEVICE_MAP.keys(),
        help="Hub device name to run the job on.",
    )
    parser.add_argument(
        "--app-dir",
        type=str,
        required=True,
        help="Directory of the fetched app (output of 'qai-hub-apps fetch').",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="App Test",
        help="QDC job name.",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        default=False,
        help=(
            "Run the app inside a Docker container on the device using the "
            "Dockerfile bundled with the app."
        ),
    )
    parser.add_argument(
        "--save-bundle",
        type=str,
        default=None,
        metavar="DIR",
        help="If set, copy the test.zip bundle to this directory before uploading.",
    )

    args = parser.parse_args()
    if not os.path.isdir(args.app_dir):
        raise NotADirectoryError(
            f"app-dir '{args.app_dir}' does not exist or is not a directory."
        )
    success = submit_app_bundle_to_qdc_device(
        args.api_token,
        args.device,
        args.app_dir,
        args.job_name,
        args.docker,
        args.save_bundle,
    )
    raise SystemExit(0 if success else 1)
