# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import sys
import tempfile
from collections import Counter
from enum import Enum, unique
from pathlib import Path
from typing import TYPE_CHECKING

from tap import Tap

if TYPE_CHECKING:
    from mypy_boto3_s3.service_resource import Bucket

from qai_hub_apps_test.bundlers import bundle_app
from qai_hub_apps_test.configs.asset_bases_yaml import AssetBases
from qai_hub_apps_test.configs.info_yaml import (
    AppLanguage,
    AppStatus,
    AppType,
    AppUrl,
    QAIHAAppInfo,
    QAIHACLIAppInfo,
)
from qai_hub_apps_test.configs.registry_yaml import AppRegistry
from qai_hub_apps_test.utils.aws import (
    QAIHM_PUBLIC_S3_BUCKET,
    attempt_with_s3_credentials_warning,
    get_qaihm_s3,
)
from qai_hub_apps_test.utils.paths import REPOSITORY_ROOT, get_all_apps


@unique
class RegistryScope(Enum):
    """Which apps to include when generating registry.yaml.

    - PRODUCTION: apps that ship in the released CLI registry (PUBLISHED +
      DEPRECATED). The only scope permitted with --build_and_upload.
    - TEST: the CI test set — everything exercised by tests (UNPUBLISHED,
      PUBLISHED, DEPRECATED); excludes PUBLISHED_WEBSITE_ONLY.
    - ALL: every app regardless of status
    """

    PRODUCTION = "production"
    TEST = "test"
    ALL = "all"

    def includes(self, info: QAIHAAppInfo) -> bool:
        """Whether an app with this info belongs to this scope."""
        if self is RegistryScope.PRODUCTION:
            return info.status in (AppStatus.PUBLISHED, AppStatus.DEPRECATED)
        if self is RegistryScope.TEST:
            return info.status in (
                AppStatus.UNPUBLISHED,
                AppStatus.PUBLISHED,
                AppStatus.DEPRECATED,
            )
        if self is RegistryScope.ALL:
            return True
        raise NotImplementedError(f"Unhandled registry scope: {self}")


def _is_supported_app(info: QAIHAAppInfo) -> bool:
    """Whether an app can be bundled/registered by the CLI.

    Supported: Android apps, any Python app, and Windows C++ apps.
    """
    if info.app_type == AppType.ANDROID:
        return True
    if AppLanguage.PYTHON in info.languages:
        return True
    return info.app_type == AppType.WINDOWS and AppLanguage.CPP in info.languages


def _read_cli_version() -> str:
    from setuptools_scm import get_version

    return get_version(root=str(REPOSITORY_ROOT))


class GenerateRegistryParser(Tap):
    output_dir: Path  # Directory where registry.yaml will be written
    schema_version: str = "1.1"  # Schema version to embed in the registry
    # Bump this when a schema change requires a newer CLI to parse the registry correctly.
    # The check is exclusive (cli > min_cli_version), so set this to the last version
    # that does NOT support the new schema.
    min_cli_version: str = "0.29.0"
    ref: str = "main"  # Git ref (branch or tag) used to construct GitHub URLs

    cli_version: str = _read_cli_version()  # CLI version used for S3 path
    build_and_upload: bool = False  # Build app zips and upload to S3; without this, list apps without bundling
    # Which apps to include: Only 'production' may be combined with --build_and_upload.
    scope: RegistryScope = RegistryScope.PRODUCTION


def _resolve_repo_url(info: QAIHAAppInfo, repo_base: str, ref: str) -> str:
    """Return the full GitHub URL for an app's source.

    Uses app_repo_url directly if set (e.g. external repos); otherwise
    constructs the URL from repo_base, ref, and app_repo_relative_path.
    """
    if info.app_repo_url:
        return info.app_repo_url
    return f"{repo_base}/tree/{ref}/apps/{info.app_repo_relative_path}"


def upload_registry(
    registry_path: Path, bucket: Bucket, s3_prefix: str, cli_version: str
) -> None:
    """Upload registry.yaml to S3."""
    s3_key = f"{s3_prefix}/{cli_version}/registry.yaml"

    def _upload(key: str = s3_key, path: Path = registry_path) -> None:
        bucket.upload_file(str(path), key, ExtraArgs={"ACL": "public-read"})

    attempt_with_s3_credentials_warning(_upload)
    print(f"Uploaded to s3://{QAIHM_PUBLIC_S3_BUCKET}/{s3_key}")


def upload_app(
    zip_path: Path, app_id: str, bucket: Bucket, s3_prefix: str, cli_version: str
) -> None:
    """Upload an app zip to S3."""
    s3_key = f"{s3_prefix}/{cli_version}/{app_id}/source.zip"

    def _upload(key: str = s3_key, path: Path = zip_path) -> None:
        bucket.upload_file(str(path), key, ExtraArgs={"ACL": "public-read"})

    attempt_with_s3_credentials_warning(_upload)
    print(f"Uploaded to s3://{QAIHM_PUBLIC_S3_BUCKET}/{s3_key}")


def generate_registry(
    output_dir: Path,
    all_apps: list[tuple[QAIHAAppInfo, Path]],
    repo_base: str,
    ref: str,
    cli_version: str,
    schema_version: str = "1.0",
    min_cli_version: str = "0.0.1",
    build_and_upload: bool = False,
    scope: RegistryScope = RegistryScope.PRODUCTION,
) -> None:
    """Generate registry.yaml from a list of (info, app_dir) pairs.

    Parameters
    ----------
    output_dir:
        Directory where registry.yaml will be written.
    all_apps:
        Every (QAIHAAppInfo, app_dir) pair in the repo, unfiltered.
    repo_base:
        GitHub repo base URL (no ref), e.g. https://github.com/qualcomm/ai-hub-apps
    ref:
        Git ref (branch or tag) used to construct GitHub source URLs.
    cli_version:
        CLI version string embedded in the registry and used as the S3 path component.
    schema_version:
        Schema version to embed in the registry.
    min_cli_version:
        Minimum CLI version required to consume this registry.
    build_and_upload:
        If True, bundle Python and Android apps and upload zips + registry to S3.
    scope:
        Which apps to include (see RegistryScope). Only 'production' may be combined with build_and_upload.
    """
    if build_and_upload and scope is not RegistryScope.PRODUCTION:
        raise SystemExit(
            f"build_and_upload requires scope='production' (got '{scope.value}')."
        )

    mode = "build + upload" if build_and_upload else "list only"
    print(f"\n{'  Generating registry.yaml  ':=^60}")
    print(f"Scope:     {scope.value} ({mode})")
    print(f"CLI ver:   {cli_version}")
    print(f"Ref:       {ref} (repo base: {repo_base})")
    output_dir.mkdir(parents=True, exist_ok=True)

    public_apps: list[tuple[QAIHAAppInfo, Path]] = []
    excluded: list[QAIHAAppInfo] = []
    for info, app_dir in all_apps:
        if info.id != app_dir.name:
            raise SystemExit(
                f"Error: app ID '{info.id}' in {app_dir / 'info.yaml'} "
                f"does not match directory name '{app_dir.name}'."
            )
        if scope.includes(info):
            resolved_url = _resolve_repo_url(info, repo_base, ref)
            public_apps.append(
                (info.model_copy(update={"app_repo_url": resolved_url}), app_dir)
            )
        else:
            excluded.append(info)

    # Report which apps are in/out of this scope, grouped by status.
    included_by_status = Counter(info.status.value for info, _ in public_apps)
    print(
        f"\nIncluded {len(public_apps)}/{len(all_apps)} app(s) for scope "
        f"'{scope.value}': "
        + (
            ", ".join(f"{n} {s}" for s, n in sorted(included_by_status.items()))
            or "none"
        )
    )
    if excluded:
        excluded_by_status = Counter(info.status.value for info in excluded)
        print(
            f"Excluded {len(excluded)} app(s): "
            + ", ".join(f"{n} {s}" for s, n in sorted(excluded_by_status.items()))
        )

    ids = [info.id for info, _ in public_apps]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise SystemExit(f"Error: duplicate app IDs found: {sorted(dupes)}")

    s3_prefix = "qai-hub-apps/releases"
    S3_REGION = "us-west-2"
    s3_base = f"https://{QAIHM_PUBLIC_S3_BUCKET}.s3.{S3_REGION}.amazonaws.com"

    if build_and_upload:
        if "dev" in cli_version:
            print(
                f"\nWarning: version '{cli_version}' looks like a development build.\n"
                f"Uploading dev versions to S3 is not recommended."
            )
            answer = input("Continue? [y/N]: ").strip().lower()
            if answer != "y":
                sys.exit("Aborted.")
        bucket, _ = get_qaihm_s3(QAIHM_PUBLIC_S3_BUCKET, requires_admin=False)

    bundled_apps: list[QAIHACLIAppInfo] = []

    if build_and_upload:
        with tempfile.TemporaryDirectory() as build_dir:
            build_path = Path(build_dir)
            for info, app_dir in public_apps:
                print(f"\n{f' {info.id} ':─^60}")
                if not _is_supported_app(info):
                    print(
                        f"Skipping: unsupported app "
                        f"(type={info.app_type.value}, languages={[l.value for l in info.languages]})"
                    )
                    continue
                bundle_app(app_dir, build_path, make_zip=True)
                zip_path = build_path / f"{app_dir.name}.zip"
                upload_app(zip_path, info.id, bucket, s3_prefix, cli_version)
                source_url = f"{s3_base}/{s3_prefix}/{cli_version}/{info.id}/source.zip"
                bundled_apps.append(
                    info.model_copy(update={"url": AppUrl(source=source_url)})
                )
    else:
        for info, _ in public_apps:
            print(f"\n{f' {info.id} ':─^60}")
            if not _is_supported_app(info):
                print(
                    f"Skipping: unsupported app "
                    f"(type={info.app_type.value}, languages={[l.value for l in info.languages]})"
                )
                continue
            bundled_apps.append(info)
            print("Registered (no bundle)")

    registry = AppRegistry(
        schema_version=schema_version,
        min_cli_version=min_cli_version,
        version=cli_version if build_and_upload else None,
        apps=bundled_apps,
    )
    registry.to_yaml(output_dir / "registry.yaml", write_if_empty=True)

    if build_and_upload:
        upload_registry(
            output_dir / "registry.yaml",
            bucket,
            s3_prefix,
            cli_version,
        )

    action = "Uploaded" if build_and_upload else "Registered"
    print(f"\n{'  Summary  ':=^60}")
    print(
        f"{action} {len(bundled_apps)} app(s) out of {len(public_apps)} "
        f"in-scope ('{scope.value}') app(s) to {output_dir / 'registry.yaml'}"
    )
    if build_and_upload:
        print(
            f"Registry uploaded to: {s3_base}/{s3_prefix}/{cli_version}/registry.yaml"
        )


def main() -> None:
    args = GenerateRegistryParser().parse_args()
    repo_base = AssetBases.load().app_repo_base
    all_apps = [QAIHAAppInfo.from_app(rel_path) for rel_path in get_all_apps()]
    generate_registry(
        args.output_dir,
        all_apps,
        repo_base,
        args.ref,
        args.cli_version,
        args.schema_version,
        args.min_cli_version,
        args.build_and_upload,
        args.scope,
    )


if __name__ == "__main__":
    main()
