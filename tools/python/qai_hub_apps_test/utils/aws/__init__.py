# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

import boto3
import tqdm
from botocore.exceptions import ClientError, NoCredentialsError

if TYPE_CHECKING:
    from mypy_boto3_s3.service_resource import Bucket


QAIHM_PUBLIC_S3_BUCKET = "qaihub-public-assets"
QAIHM_PRIVATE_S3_BUCKET = "qai-hub-models-private-assets"

CallableRetT = TypeVar("CallableRetT")


def _is_on_ci() -> bool:
    """Whether we're running in CI, per the QAIHA_CI environment variable."""
    return os.environ.get("QAIHA_CI", "").lower() in {"true", "1", "on", "yes"}


def attempt_with_s3_credentials_warning(
    s3_call: Callable[[], CallableRetT],
) -> CallableRetT:
    """
    Attempt to call the given function. Wrap the failure with a helpful warning about missing credentials.

    Typically you would call this like so:
        list_s3_files_in_folder_recursive(lambda: get_s3_url(args))
    """
    try:
        return s3_call()
    except (ClientError, NoCredentialsError) as e:
        if (
            isinstance(e, NoCredentialsError)
            or e.response.get("Error", {}).get("Code", None) == "400"
        ):
            raise ValueError(
                "S3 credentials not found or expired. Run `python -m qai_hub_apps_test.utils.aws.validate_credentials` and retry."
            ) from None
        raise


def s3_download(bucket: Bucket, key: str, local_file_path: str) -> None:
    """Download file at s3://<bucket>/<key> to local_file_path."""
    obj = bucket.Object(key)
    with tqdm.tqdm(total=obj.content_length, unit="B", unit_scale=True) as t:
        attempt_with_s3_credentials_warning(
            lambda: obj.download_file(
                local_file_path, Callback=t.update if _is_on_ci() else None
            )
        )
        if _is_on_ci():
            t.update(obj.content_length)


def get_qaihm_s3(bucket_name: str, requires_admin: bool = False) -> tuple[Bucket, bool]:
    """
    Get boto3 objects for interacting with the given bucket using QAIHM credentials.
    Throws if credentials do not exist

    Parameters
    ----------
    bucket_name
        Name of the s3 bucket to get objects for.
    requires_admin
        If True, uses the admin AWS profile.

    Returns
    -------
    tuple[Bucket, bool]
        Bucket object and a bool indicating whether the session has admin permissions.
    """
    if requires_admin:
        profile_name = "qaihm-admin"
        exception_msg = "This action requires adminsitrator permissions. Administrator permissions are not available."
    else:
        profile_name = "qaihm"
        exception_msg = "Could not find valid AWS profile. Run <repo_root>/scripts/build_and_test.py --task validate_aws_credentials"

    try:
        session: boto3.Session = boto3.Session(profile_name=profile_name)
        bucket = session.resource("s3").Bucket(bucket_name)
        return bucket, False
    except Exception as e:
        raise ValueError(exception_msg) from e
