# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from collections.abc import ValuesView
from pathlib import Path
from typing import Any

from qai_hub_models_cli.fetch import get_asset_url
from qai_hub_models_cli.proto_helpers.release_assets import AssetNotFoundError
from qai_hub_models_cli.utils import download, extract_zip_file, get_next_free_path
from qai_hub_models_cli.versions import CURRENT_VERSION as QAIHM_VERSION

from qai_hub_apps import __version__, _is_dev
from qai_hub_apps.registry.remote import ensure_registry

try:
    from qai_hub_apps_test.bundlers import bundle_app as _bundle_app
except ImportError:  # pragma: no cover
    _bundle_app = None
from qai_hub_apps.configs.app_yaml import AppInfo, AppLanguage, AppStatus
from qai_hub_apps.configs.model_asset import ModelAsset
from qai_hub_apps.configs.registry_yaml import AppRegistry
from qai_hub_apps.errors import (
    AppIncompatibleError,
    AppNotFoundError,
    InvalidArgumentError,
    ModelAssetNotFoundError,
    QAIHubAppsError,
)
from qai_hub_apps.utils.github import make_issue_url
from qai_hub_apps.validate import is_app_supported

DEFAULT_DEPRECATION_MESSAGE = (
    "This app is deprecated and may be removed in a future release."
)


class App:
    """CLI-layer wrapper around AppInfo. Owns presentation logic."""

    def __init__(self, info: AppInfo) -> None:
        self._info = info

    def __getattr__(self, name: str) -> Any:
        # Transparently delegate field access to the underlying AppInfo.
        return getattr(self._info, name)

    def deprecation_message(self) -> str | None:
        """Return the deprecation message for this app, or None if not deprecated.

        Uses the app's ``deprecation_notice`` when set, otherwise a default message.
        """
        if self.status != AppStatus.DEPRECATED:
            return None
        return self.deprecation_notice or DEFAULT_DEPRECATION_MESSAGE

    def _ensure_model_supported(self, model_id: str) -> None:
        """Raise if *model_id* is not one of the app's supported models."""
        if model_id not in self.related_models:
            available = ", ".join(self.related_models) or "none"
            raise AppIncompatibleError(
                f"Model '{model_id}' is not supported for this app. Supported models: {available}"
            )

    def _stage_app(self, tmp: Path) -> Path:
        """Stage the app source into *tmp* and return the staged directory.

        Uses the registry source URL when present; otherwise (dev install only)
        bundles the app from local source.
        """
        staged = tmp / self.id
        if self.url is not None:
            print(f"Fetching from: {self.url.source}")
            return download(self.url.source, staged, extract=True)
        if not _is_dev():
            raise QAIHubAppsError(
                "No source URL found in registry. "
                "The registry may be outdated. Please upgrade: pip install -U qai-hub-apps"
            )
        # Dev install + no URL: bundle from source on-the-fly.
        if _bundle_app is None:
            raise QAIHubAppsError(
                "Dev install detected but qai_hub_apps_test is not installed. "
                "Install it with: pip install -e tools/python/"
            )
        print(f"Dev install: bundling '{self.id}' from source...")
        # bundle_app with make_zip=False writes to tmp/<app_id>/ == staged
        _bundle_app(self.id, tmp, make_zip=False)
        return staged

    def _stage_model(self, model_asset: ModelAsset, dest: Path) -> None:
        """Stage the model asset into *dest*.

        Copies/extracts a locally-exported model (directory or .zip), or
        resolves and downloads the model asset.
        """
        if model_asset.path is not None:
            src = model_asset.path
            if zipfile.is_zipfile(src):
                extract_zip_file(src, dest)
            elif src.is_dir():
                shutil.copytree(src, dest)
            else:
                raise AppIncompatibleError(
                    f"--model path must be a directory or a .zip file: {src}"
                )
            return

        assert model_asset.model_id is not None  # set when path is None
        try:
            model_download_url = get_asset_url(
                model=model_asset.model_id,
                runtime=self.runtime,
                precision=self.precisions[0],
                version=QAIHM_VERSION,
                chipset=model_asset.chipset,
            )
        except AssetNotFoundError as e:
            reason = str(e)
            if e.model_sharing_restricted:
                reason += (
                    " After exporting, point to the assets when fetching the app:\n"
                    f"  qai-hub-apps fetch {self.id} --model <exported_model_path>"
                )
            raise ModelAssetNotFoundError(
                model_asset.model_id, model_asset.chipset, reason=reason
            ) from e
        except KeyError as e:
            raise ModelAssetNotFoundError(
                model_asset.model_id, model_asset.chipset
            ) from e
        download(model_download_url, dest, extract=True)

    def _read_model_metadata(self, model_dir: Path, model_asset: ModelAsset) -> dict:
        """Read and validate metadata.json for a model in *model_dir*."""
        is_local = model_asset.path is not None
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            if is_local:
                raise AppIncompatibleError(
                    f"No metadata.json found in '{model_asset.path}'. Point --model at "
                    "the exported model directory (or .zip) that contains metadata.json. "
                    "Ensure you exported the model using the AI Hub Models package "
                    "(https://github.com/qualcomm/ai-hub-models)."
                )
            issue_url = make_issue_url(
                title=f"Model asset missing metadata.json for app '{self.id}'",
                body=(f"App: {self.id}\nModel ID: {model_asset.model_id}"),
            )
            raise AppIncompatibleError(
                f"The model '{model_asset.model_id}' downloaded for '{self.id}' is missing metadata.json. "
                f"This is likely a bug - please file an issue and we'll look into it:\n"
                f"  {issue_url}"
            )
        with open(metadata_path) as f:
            metadata = json.load(f)

        missing = [key for key in ("model_id", "model_files") if key not in metadata]
        if missing:
            if is_local:
                raise AppIncompatibleError(
                    f"metadata.json at '{model_asset.path}' is missing required "
                    f"field(s): {', '.join(missing)}. Ensure you exported the model "
                    "using the AI Hub Models package "
                    "(https://github.com/qualcomm/ai-hub-models)."
                )
            issue_url = make_issue_url(
                title=f"Model asset metadata missing field(s) for app '{self.id}'",
                body=(
                    f"App: {self.id}\nModel ID: {model_asset.model_id}\n"
                    f"AI Hub Models version: {QAIHM_VERSION}\n"
                    f"Missing field(s): {', '.join(missing)}"
                ),
            )
            raise AppIncompatibleError(
                f"metadata.json for model '{model_asset.model_id}' downloaded for '{self.id}' "
                f"is missing required field(s): {', '.join(missing)}. "
                f"This is likely a bug - please file an issue and we'll look into it:\n"
                f"  {issue_url}"
            )

        return metadata

    def _place_model_in_app(
        self, model_dir: Path, app_dir: Path, metadata: dict
    ) -> None:
        """Move a model asset's files into the app's source tree.

        Uses ``model_file_paths`` (renaming each file to its destination name) or,
        if unset, ``model_file_dir`` (dropping files in as-is).
        """
        model_id = metadata["model_id"]
        src_names = list(metadata["model_files"].keys())
        if self.model_file_paths:
            dst_paths = self.model_file_paths
            if len(src_names) != len(dst_paths):
                issue_url = make_issue_url(
                    title=f"Model file count mismatch for app '{self.id}'",
                    body=(
                        f"App: {self.id}\n"
                        f"Version: {__version__}\n"
                        f"Model ID: {model_id}\n"
                        f"Expected files: {len(dst_paths)}, Got: {len(src_names)}\n"
                        f"Available files in model asset: {', '.join(src_names)}"
                    ),
                )
                raise AppIncompatibleError(
                    f"The model '{model_id}' for '{self.id}' ({__version__}) has {len(src_names)} "
                    f"file(s) but {len(dst_paths)} were expected. "
                    f"This is likely a bug - please file an issue and we'll look into it:\n"
                    f"  {issue_url}"
                )
            dst_parents = {Path(p).parent for p in dst_paths}
            if len(dst_parents) > 1:
                issue_url = make_issue_url(
                    title=f"model_file_paths directory mismatch for app '{self.id}'",
                    body=(
                        f"App: {self.id}\n"
                        f"model_file_paths: {[str(p) for p in dst_paths]}"
                    ),
                )
                raise AppIncompatibleError(
                    f"All model_file_paths for '{self.id}' must share the same parent directory. "
                    f"This is likely a bug - please file an issue and we'll look into it:\n"
                    f"  {issue_url}"
                )
            # Build rename map: original filename -> desired destination filename
            rename_map = {
                src_name: Path(dst_rel).name
                for src_name, dst_rel in zip(src_names, dst_paths, strict=True)
            }
            # Move entire asset into its destination directory
            models_dest = app_dir / Path(dst_paths[0]).parent
            models_dest.mkdir(parents=True, exist_ok=True)
            for item in model_dir.iterdir():
                dest_name = rename_map.get(item.name, item.name)
                if item.name == "metadata.json":
                    # Update model_files keys to reflect renames, then write
                    updated_files = {
                        rename_map.get(k, k): v
                        for k, v in metadata["model_files"].items()
                    }
                    metadata["model_files"] = updated_files
                    (models_dest / "metadata.json").write_text(
                        json.dumps(metadata, indent=2)
                    )
                else:
                    shutil.move(str(item), models_dest / dest_name)
        else:
            # model_file_dir: drop all files as-is into the target directory
            models_dest = app_dir / self.model_file_dir
            models_dest.mkdir(parents=True, exist_ok=True)
            for item in model_dir.iterdir():
                shutil.move(str(item), models_dest / item.name)

    def fetch(
        self,
        dest: Path,
        model_asset: ModelAsset | None = None,
    ) -> Path:
        """Download and extract app source. Returns the extraction path."""
        app_dest = dest / self.id

        if app_dest.exists():
            new_dest = get_next_free_path(app_dest)
            print(f"Warning: {app_dest} already exists, saving to {new_dest} instead.")
            app_dest = new_dest

        is_model_required = model_asset is not None
        is_model_local = model_asset is not None and model_asset.path is not None

        with tempfile.TemporaryDirectory() as _tmp:
            tmp = Path(_tmp)

            if is_model_required:
                assert model_asset is not None
                if self.disable_cli_model_fetch:
                    raise AppIncompatibleError(
                        f"App '{self.id}' downloads its model at runtime and bundles no model files. "
                        f"Re-run without --model:\n  qai-hub-apps fetch {self.id}"
                    )

                if not self.model_file_paths and not self.model_file_dir:
                    raise AppIncompatibleError(
                        f"No model_file_paths or model_file_dir configured for app '{self.id}'."
                    )

                # An auto-resolved `--model` whose value is both a supported model
                # and an existing path is ambiguous.
                if is_model_local and str(model_asset.path) in self.related_models:
                    raise InvalidArgumentError(
                        f"'{model_asset.path}' is both a supported model id and a local path. "
                        "Use --model-id or --model-path to disambiguate."
                    )

                if not is_model_local:
                    assert (
                        model_asset.model_id is not None
                    )  # must be set when path is None
                    self._ensure_model_supported(model_asset.model_id)

            staged = self._stage_app(tmp)

            if is_model_required:
                assert model_asset is not None
                model_tmp = tmp / "model_asset"
                self._stage_model(model_asset, model_tmp)
                metadata = self._read_model_metadata(model_tmp, model_asset)
                meta_model_id = metadata["model_id"]
                if is_model_local:
                    # check the local model is supported by the app
                    self._ensure_model_supported(meta_model_id)
                elif meta_model_id != model_asset.model_id:
                    # the downloaded asset's model should match the requested one
                    issue_url = make_issue_url(
                        title=f"Model asset id mismatch for app '{self.id}'",
                        body=(
                            f"App: {self.id}\n"
                            f"Requested model ID: {model_asset.model_id}\n"
                            f"metadata.json model ID: {meta_model_id}\n"
                            f"AI Hub Models version: {QAIHM_VERSION}"
                        ),
                    )
                    raise AppIncompatibleError(
                        f"The downloaded model asset for '{self.id}' reports model id "
                        f"'{meta_model_id}', but '{model_asset.model_id}' was requested. "
                        f"This is likely a bug - please file an issue and we'll look into it:\n"
                        f"  {issue_url}"
                    )
                self._place_model_in_app(model_tmp, staged, metadata)

            shutil.move(staged, app_dest)

        return app_dest

    def __repr__(self) -> str:
        lines = [self.name, "\u2550" * 50, ""]
        if deprecation := self.deprecation_message():
            lines.append(f"\u26a0  DEPRECATED: {deprecation}\n")
        for label, value in self.detail_fields():
            lines.append(f"{label + ':':<12}{value}")
        lines.append("")
        if self.headline:
            lines.append(f"{self.headline}\n")
        if self.description:
            lines.append(f"{self.description}\n")
        if self.app_repo_url:
            lines.append(f"Repo:  {self.app_repo_url}")
        return "\n".join(lines)

    def detail_fields(self) -> list[tuple[str, str]]:
        fields: list[tuple[str, str]] = [
            ("ID", self.id or "-"),
            ("Type", self.app_type.value),
        ]
        if self.runtime:
            fields.append(("Runtime", self.runtime))
        if self.domain:
            fields.append(("Domain", self.domain))
        if self.use_case:
            fields.append(("Use Case", self.use_case))
        if self.precisions:
            fields.append(("Precisions", ", ".join(self.precisions)))
        if self.related_models:
            fields.append(("Models", ", ".join(str(m) for m in self.related_models)))
        return fields


class Registry:
    """CLI-layer wrapper around AppRegistry. Singleton — one instance per process."""

    _instance: Registry | None = None

    def __init__(self, raw: AppRegistry) -> None:
        self._raw = raw
        self._apps = {a.id: _make_app(a) for a in raw.apps}

    @classmethod
    def load(cls, path: str | Path | None = None) -> Registry:
        """Load and return the singleton Registry instance.

        Parameters
        ----------
        path:
            Path to registry.yaml. If None, resolves via ensure_registry()
            (bundled file -> local cache -> S3 download).

        Returns
        -------
        Registry
            The singleton Registry instance.

        Note
        ----
        This is a process-level singleton. If called a second time with a
        different path, the cached instance is returned unchanged. In practice
        load() is called exactly once per process — either with no path (default
        registry) or with an explicit path from ``--registry`` CLI arg.
        """
        if cls._instance is None:
            if path is None:
                path = ensure_registry(__version__)
            cls._instance = cls(AppRegistry.from_yaml(Path(path)))
        return cls._instance

    @property
    def apps(self) -> ValuesView[App]:
        return self._apps.values()

    def find_by_id(self, app_id: str) -> App:
        app = self._apps.get(app_id.lower())
        if app is None:
            raise AppNotFoundError(app_id)
        return app

    @property
    def version(self) -> str:
        return self._raw.version or "dev"

    def fetch_app(
        self,
        app_id: str,
        dest: Path,
        model_asset: ModelAsset | None = None,
    ) -> Path:
        """Find app by ID and download + extract it. Returns the extraction path."""
        app = self.find_by_id(app_id)

        deprecation = app.deprecation_message()
        if deprecation:
            print(f"Warning: {deprecation}")

        if not is_app_supported(app):
            print("Warning: This app may not be supported on the current device.")

        return app.fetch(dest, model_asset=model_asset)


def _make_app(info: AppInfo) -> App:
    """Return the appropriate App subclass based on the app's languages."""
    if AppLanguage.PYTHON in info.languages:
        from qai_hub_apps.registry.python_app import PythonApp

        return PythonApp(info)
    return App(info)
