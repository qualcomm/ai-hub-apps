# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from abc import ABC, abstractmethod

from qualcomm_device_cloud_sdk.models import ArtifactType, JobResult, JobState

from qai_hub_apps_test.qdc.qdc_jobs import (
    HUB_DEVICE_TO_QDC_DEVICE_MAP,
    POLL_INTERVAL,
    QDCDevice,
    QDCJobs,
)
from qai_hub_apps_test.utils.paths import REPOSITORY_ROOT

TEXT_LOG_EXTENSIONS = (".log", ".stdout", ".txt", ".json")

# The nightly publishes the CLI wheel to this self-hosted (plain-HTTP) dev index;
# see tools/ci/install_cli.sh.
_S3_WHEEL_INDEX_HOST = "qaihub-public-python-wheels.s3-website-us-west-2.amazonaws.com"


def _cli_spec(version: str | None) -> str:
    """Pip spec for the CLI on the device.

    ``_stage_cli_artifacts`` overrides this with the bundled wheel's on-device path;
    dependencies resolve from PyPI, which every QDC device can reach. No QDC device
    needs the S3 wheel index.
    """
    return f"qai-hub-apps=={version.lstrip('v')}" if version else "qai-hub-apps"


def _resolve_registry(source: str, wheel_path: str, out_dir: str) -> str:
    """Resolve the registry.yaml matching a CLI wheel and return its local path.

    The wheel excludes ``registry.yaml``. For a ``source`` build it is the generated
    test-scope registry already in the checkout. For ``s3``/``prod`` it is fetched from
    S3 by installing the wheel in a throwaway venv and calling ``ensure_registry`` there
    (keyed on the installed CLI's own version), then copied next to the wheel.
    """
    if source == "source":
        registry = str(REPOSITORY_ROOT / "cli" / "qai_hub_apps" / "registry.yaml")
        print(f"[qdc] registry (source): using checkout registry {registry}")
        return registry

    venv_dir = os.path.join(out_dir, "registry-venv")
    print(f"[qdc] registry ({source}): installing {wheel_path} into venv {venv_dir}")
    subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    venv_python = os.path.join(venv_dir, bin_dir, "python")
    subprocess.check_call(
        [venv_python, "-m", "pip", "install", "--no-input", wheel_path]
    )
    resolved = subprocess.check_output(
        [
            venv_python,
            "-c",
            "from qai_hub_apps import __version__; "
            "from qai_hub_apps.registry.remote import ensure_registry; "
            "print(__version__); print(ensure_registry(__version__))",
        ],
        text=True,
    )
    installed_version, registry_src = resolved.strip().splitlines()[-2:]
    dest = os.path.join(out_dir, "registry.yaml")
    shutil.copy(registry_src, dest)
    print(
        f"[qdc] registry ({source}): installed CLI __version__={installed_version}; "
        f"ensure_registry -> {registry_src}; copied to {dest}"
    )
    return dest


def obtain_cli_bundle(
    source: str, version: str | None, out_dir: str, wheel: str | None = None
) -> tuple[str, str]:
    """Provide a qai-hub-apps wheel and its matching registry for the device bundle.

    ``source`` selects where the host gets the wheel when ``wheel`` isn't supplied:
    ``source`` builds it from this checkout, ``s3`` downloads the nightly wheel from the
    dev index, ``prod`` downloads the release wheel from PyPI. The registry is resolved
    by :func:`_resolve_registry`.

    Parameters
    ----------
    source
        One of ``source``, ``s3``, ``prod``.
    version
        CLI version to download (ignored for ``source`` and when ``wheel`` is given).
    out_dir
        Directory to write the wheel/registry into.
    wheel
        A prebuilt wheel to use instead of building/downloading.

    Returns
    -------
    tuple[str, str]
        ``(wheel_path, registry_path)``.
    """
    os.makedirs(out_dir, exist_ok=True)
    if wheel is not None:
        print(f"[qdc] cli bundle: using provided wheel {wheel} (source={source})")
        return wheel, _resolve_registry(source, wheel, out_dir)

    print(
        f"[qdc] cli bundle: obtaining wheel (source={source}, "
        f"version={version or 'latest'}) into {out_dir}"
    )
    if source == "source":
        built = subprocess.check_output(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "tools" / "ci" / "build_cli_wheel.py"),
                out_dir,
            ],
            cwd=REPOSITORY_ROOT,
            text=True,
        ).strip()
        print(f"[qdc] cli bundle: wheel ready {built}")
        return built, _resolve_registry(source, built, out_dir)
    spec = f"qai-hub-apps=={version.lstrip('v')}" if version else "qai-hub-apps"
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--no-deps",
        "--only-binary=:all:",
        "--dest",
        out_dir,
    ]
    if source == "s3":
        cmd += [
            "--pre",
            "--index-url",
            f"http://{_S3_WHEEL_INDEX_HOST}/",
            "--extra-index-url",
            "https://pypi.org/simple/",
            "--trusted-host",
            _S3_WHEEL_INDEX_HOST,
        ]
    cmd.append(spec)
    subprocess.check_call(cmd)

    wheels = glob.glob(os.path.join(out_dir, "qai_hub_apps-*.whl"))
    if not wheels:
        raise RuntimeError(f"No qai-hub-apps wheel produced in {out_dir}")
    print(f"[qdc] cli bundle: wheel ready {wheels[0]}")
    return wheels[0], _resolve_registry(source, wheels[0], out_dir)


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

    _DEVICE_ROOT: str  # bundle root on the device; set by each subclass
    _SEP: str  # device path separator (subclasses on Windows override)

    def __init__(self, device_name: str, model_id: str, cli_spec: str) -> None:
        self.device_name = device_name
        self.model_id = model_id
        self.cli_spec = cli_spec

    def _stage_cli_artifacts(
        self,
        dest_dir: os.PathLike | str,
        cli_wheel: str,
        registry_path: str | None = None,
    ) -> tuple[str, str]:
        """Copy the source wheel (and optional registry) into the bundle root.

        Returns the spec to install (the bundled wheel's on-device path) and the
        on-device registry path (empty when no registry is bundled).
        """
        name = os.path.basename(cli_wheel)
        shutil.copy(cli_wheel, os.path.join(dest_dir, name))
        cli_spec = f"{self._DEVICE_ROOT}{self._SEP}{name}"
        registry_device = ""
        if registry_path:
            shutil.copy(registry_path, os.path.join(dest_dir, "registry.yaml"))
            registry_device = f"{self._DEVICE_ROOT}{self._SEP}registry.yaml"
        print(
            f"[qdc] staged CLI artifacts: wheel {cli_wheel} -> device spec "
            f"{cli_spec!r}; registry {registry_path} -> {registry_device!r}"
        )
        return cli_spec, registry_device

    def _log_substitutions(
        self, script: str, cli_spec: str, registry_device: str
    ) -> None:
        """Log the placeholder values substituted into a device script."""
        print(
            f"[qdc] {script} substitutions: DEVICE_NAME={self.device_name!r} "
            f"MODEL_ID={self.model_id!r} CLI_SPEC={cli_spec!r} "
            f"REGISTRY_PATH={registry_device!r}"
        )

    @abstractmethod
    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        app_dir: os.PathLike | str,
        dest_dir: os.PathLike | str,
        cli_wheel: str | None = None,
        registry_path: str | None = None,
    ) -> str:
        """Create artifact bundle and return path to the zip file."""
        raise NotImplementedError

    @property
    @abstractmethod
    def entry_script(self) -> str | None:
        raise NotImplementedError


class AppTestLinuxArtifactHandler(AppTestArtifactHandler):
    _DEVICE_ROOT = "/data/local/tmp/TestContent"
    _SEP = "/"

    def __init__(
        self,
        device_name: str,
        model_id: str,
        cli_spec: str,
        use_docker: bool = False,
    ) -> None:
        super().__init__(device_name, model_id, cli_spec)
        self.use_docker = use_docker

    @property
    def entry_script(self) -> str:
        return f"/bin/bash {self._DEVICE_ROOT}/run_linux.sh"

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        app_dir: os.PathLike | str,
        dest_dir: os.PathLike | str,
        cli_wheel: str | None = None,
        registry_path: str | None = None,
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
        cli_wheel
            Local CLI wheel to bundle and install on the device (source installs).
        registry_path
            Local registry.yaml to bundle and pass via ``--registry`` (source installs).

        Returns
        -------
        str
            Absolute path to the created ``test.zip`` archive.
        """
        cli_spec, registry_device = self.cli_spec, ""
        if cli_wheel:
            cli_spec, registry_device = self._stage_cli_artifacts(
                dest_dir, cli_wheel, registry_path
            )
        self._log_substitutions("run_linux.sh", cli_spec, registry_device)
        print(f"[qdc] run_linux.sh substitutions: USE_DOCKER={self.use_docker}")
        dest_script = os.path.join(dest_dir, "run_linux.sh")
        shutil.copy(
            os.path.join(curr_dirname, "device_scripts", "run_linux.sh"),
            dest_script,
        )
        with open(dest_script, encoding="utf-8") as f:
            content = f.read()
        with open(dest_script, "w", encoding="utf-8") as f:
            f.write(
                content.replace(
                    "<<USE_DOCKER>>", "true" if self.use_docker else "false"
                )
                .replace("<<DEVICE_NAME>>", self.device_name)
                .replace("<<MODEL_ID>>", self.model_id)
                .replace("<<CLI_SPEC>>", cli_spec)
                .replace("<<REGISTRY_PATH>>", registry_device)
                .replace(
                    "<<PYTHON_VERSION>>",
                    f"{sys.version_info.major}.{sys.version_info.minor}",
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
    _DEVICE_ROOT = "/qdc/appium"
    _SEP = "/"

    @property
    def entry_script(self) -> str | None:
        return None

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        app_dir: os.PathLike | str,
        dest_dir: os.PathLike | str,
        cli_wheel: str | None = None,
        registry_path: str | None = None,
    ) -> str:
        test_folder = os.path.join(dest_dir, "tests")
        os.makedirs(test_folder, exist_ok=True)

        cli_spec, registry_device = self.cli_spec, ""
        if cli_wheel:
            cli_spec, registry_device = self._stage_cli_artifacts(
                dest_dir, cli_wheel, registry_path
            )
        self._log_substitutions("run_android.py", cli_spec, registry_device)
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
                content.replace("<<DEVICE_NAME>>", self.device_name)
                .replace("<<MODEL_ID>>", self.model_id)
                .replace("<<CLI_SPEC>>", cli_spec)
                .replace("<<REGISTRY_PATH>>", registry_device)
            )

        # Create an empty requirements.txt
        open(os.path.join(dest_dir, "requirements.txt"), "w").close()

        # The launch scripts are bash, but the appium host is Alpine (sh/ash only), so
        # bundle a static bash for run_android.py to put on PATH. See bash_static/.
        bin_dir = os.path.join(dest_dir, "bin")
        os.makedirs(bin_dir, exist_ok=True)
        shutil.copy(
            os.path.join(curr_dirname, "bash_static", "bash-static-x86_64"),
            os.path.join(bin_dir, "bash"),
        )
        print(f"[qdc] staged static bash -> {self._DEVICE_ROOT}{self._SEP}bin/bash")

        copied_app_dir = os.path.join(dest_dir, "app")
        shutil.copytree(app_dir, copied_app_dir)

        # The CLI resolves the app from info.yaml and execs launch.sh (which installs
        # the APKs from build/outputs); keep only those to keep the upload small.
        keep = {"info.yaml", "launch.sh", "build"}
        for item in os.listdir(copied_app_dir):
            item_path = os.path.join(copied_app_dir, item)
            if item in keep:
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
    _DEVICE_ROOT = "C:\\Temp\\TestContent"
    _SEP = "\\"

    @property
    def entry_script(self) -> str:
        return f"{self._DEVICE_ROOT}\\run_windows.ps1"

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        app_dir: os.PathLike | str,
        dest_dir: os.PathLike | str,
        cli_wheel: str | None = None,
        registry_path: str | None = None,
    ) -> str:
        cli_spec, registry_device = self.cli_spec, ""
        if cli_wheel:
            cli_spec, registry_device = self._stage_cli_artifacts(
                dest_dir, cli_wheel, registry_path
            )
        self._log_substitutions("run_windows.ps1", cli_spec, registry_device)
        dest_script = os.path.join(dest_dir, "run_windows.ps1")
        shutil.copy(
            os.path.join(curr_dirname, "device_scripts", "run_windows.ps1"),
            dest_script,
        )
        with open(dest_script, encoding="utf-8") as f:
            content = f.read()
        with open(dest_script, "w", encoding="utf-8") as f:
            f.write(
                content.replace("<<DEVICE_NAME>>", self.device_name)
                .replace("<<MODEL_ID>>", self.model_id)
                .replace("<<CLI_SPEC>>", cli_spec)
                .replace("<<REGISTRY_PATH>>", registry_device)
                .replace(
                    "<<PYTHON_VERSION>>",
                    f"Python.Python.{sys.version_info.major}.{sys.version_info.minor}",
                )
            )

        shutil.copytree(app_dir, os.path.join(dest_dir, "app"))

        zip_path = os.path.join(os.path.dirname(dest_dir), "test.zip")
        create_zip(zip_path, dest_dir)
        return zip_path


class AppTestQDCJobs(QDCJobs):
    """QDC job handler for generic app on-device testing."""

    def _get_artifact_handler(
        self,
        qdc_device: QDCDevice,
        device_name: str,
        model_id: str,
        cli_spec: str,
        use_docker: bool = False,
    ) -> AppTestArtifactHandler:
        if qdc_device.iot_platform:
            return AppTestLinuxArtifactHandler(
                device_name, model_id, cli_spec, use_docker=use_docker
            )
        if qdc_device.mobile_platform:
            return AppTestAndroidArtifactHandler(device_name, model_id, cli_spec)
        if qdc_device.windows_platform:
            # windows containers are not supported on arm64
            return AppTestWindowsArtifactHandler(device_name, model_id, cli_spec)
        raise NotImplementedError(
            f"On-device app testing is not yet supported for this platform. "
            f"Device: {qdc_device.device.name!r}"
        )

    def add_job_artifacts(
        self,
        qdc_device: QDCDevice,
        app_dir: str | os.PathLike,
        device_name: str,
        model_id: str,
        cli_spec: str,
        use_docker: bool = False,
        save_bundle_dir: str | os.PathLike | None = None,
        cli_wheel: str | None = None,
        registry_path: str | None = None,
    ) -> tuple[list[str], str | None]:
        """Prepare and upload app artifacts for job submission.

        Parameters
        ----------
        qdc_device
            QDCDevice instance for the target device.
        app_dir
            Directory of the fetched app (output of ``qai-hub-apps fetch``).
        device_name
            Hub device name passed to ``qai-hub-apps test --device`` on the device.
        model_id
            Model id passed to ``qai-hub-apps test --model-id`` on the device.
        cli_spec
            Pip spec used to install the ``qai-hub-apps`` CLI on the device; replaced
            with the bundled wheel's on-device path when ``cli_wheel`` is given.
        use_docker
            If True, run the app inside a Docker container on the device.
        save_bundle_dir
            If set, copy the test.zip bundle to this directory before uploading.
        cli_wheel
            Local path to a CLI wheel to bundle and install on the device (for
            source installs, where no published wheel exists on an index).
        registry_path
            Local path to a registry.yaml to bundle and pass via ``--registry`` on
            the device (for source installs, where the device cannot fetch it).

        Returns
        -------
        job_artifacts : list[str]
            List of artifact IDs returned by QDC upload.
        entry_script : str | None
            Entry script path used by the test framework.
        """
        curr_dirname = os.path.dirname(os.path.abspath(__file__))
        artifact_handler = self._get_artifact_handler(
            qdc_device, device_name, model_id, cli_spec, use_docker
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = artifact_handler.create_artifact(
                curr_dirname,
                app_dir,
                tmpdirname,
                cli_wheel,
                registry_path,
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
    model_id: str,
    job_name: str = "App Test",
    use_docker: bool = False,
    cli_version: str | None = None,
    cli_wheel: str | None = None,
    registry_path: str | None = None,
    save_bundle_dir: str | os.PathLike | None = None,
) -> bool:
    """Submit a fetched app bundle to QDC for on-device execution.

    The device installs the ``qai-hub-apps`` CLI from the bundled ``cli_wheel`` (deps
    from PyPI) and runs
    ``qai-hub-apps test --app-path <bundle> --device <device> --model-id <model_id>``.

    Parameters
    ----------
    api_token
        API token for QDC authentication.
    device
        Hub device name to run the job on (must be a key in HUB_DEVICE_TO_QDC_DEVICE_MAP).
    app_dir
        Directory of the fetched app (output of ``qai-hub-apps fetch``).
    model_id
        Model id the app was fetched with; passed to ``qai-hub-apps test --model-id``.
    job_name
        Name for the QDC job.
    use_docker
        If True, run the app inside a Docker container on the device.
    cli_version
        Version of the ``qai-hub-apps`` CLI being tested; if None, latest.
    cli_wheel
        Local path to the CLI wheel to bundle and install on the device (required).
    registry_path
        Local path to a registry.yaml to bundle and pass via ``--registry``.
    save_bundle_dir
        If set, copy the test.zip bundle to this directory before uploading.

    Returns
    -------
    success : bool
        True if the job completed successfully, False otherwise.
    """
    if cli_wheel is None:
        raise ValueError(
            "cli_wheel is required: the device installs a bundled wheel rather than "
            "fetching the CLI from an index."
        )

    qdc_device = QDCDevice(device)
    app_job = AppTestQDCJobs(
        api_key=api_token,
        app_name_header="AppTestQDCJobApp",
    )

    job_artifacts, entry_script = app_job.add_job_artifacts(
        qdc_device,
        app_dir,
        device,
        model_id,
        _cli_spec(cli_version),
        use_docker,
        save_bundle_dir,
        cli_wheel=cli_wheel,
        registry_path=registry_path,
    )

    job_id = app_job.submit_automated_job(
        qdc_device, job_artifacts, entry_script, job_name=job_name
    )
    if job_id is None:
        raise RuntimeError("Job submission failed.")

    print(f"Submitted QDC job with ID: {job_id}")
    job_status = app_job.status(job_id)
    print(f"QDC job {job_id} completed with status: {job_status}")

    job_result = app_job.result(job_id)
    print(f"QDC job {job_id} test finished with result: {job_result}")
    succeeded = job_result == JobResult.SUCCESSFUL

    if not succeeded and job_status == JobState.COMPLETED:
        app_job.log_upload_status(job_id)
        job_log_files = app_job.get_job_log_files(job_id)
        time.sleep(POLL_INTERVAL)
        if job_log_files:
            with tempfile.TemporaryDirectory() as tmpdirname:
                for job_log in job_log_files:
                    filename = getattr(job_log, "filename", None)
                    if not filename or not filename.lower().endswith(
                        TEXT_LOG_EXTENSIONS
                    ):
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
        "--model-id",
        type=str,
        required=True,
        help="Model id the app should run (was fetched) with (passed to 'qai-hub-apps test').",
    )
    parser.add_argument(
        "--cli-version",
        type=str,
        default=None,
        help="qai-hub-apps CLI version to install on the device (default: latest).",
    )
    parser.add_argument(
        "--cli-source",
        type=str,
        default="source",
        choices=["source", "s3", "prod"],
        help="Where to install the CLI from on the device: source (bundled --cli-wheel, "
        "default), s3 (nightly index), or prod (PyPI).",
    )
    parser.add_argument(
        "--cli-wheel",
        type=str,
        default=None,
        help="Local CLI wheel to bundle and install; obtained per --cli-source if omitted.",
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

    with tempfile.TemporaryDirectory() as wheel_dir:
        cli_wheel, registry_path = obtain_cli_bundle(
            args.cli_source, args.cli_version, wheel_dir, wheel=args.cli_wheel
        )
        success = submit_app_bundle_to_qdc_device(
            api_token=args.api_token,
            device=args.device,
            app_dir=args.app_dir,
            model_id=args.model_id,
            job_name=args.job_name,
            use_docker=args.docker,
            cli_version=args.cli_version,
            cli_wheel=cli_wheel,
            registry_path=registry_path,
            save_bundle_dir=args.save_bundle,
        )
    raise SystemExit(0 if success else 1)
