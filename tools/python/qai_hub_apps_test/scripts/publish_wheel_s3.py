# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Upload a built wheel to S3 and regenerate the self-hosted PEP 503 wheel index."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from packaging.utils import parse_wheel_filename
from packaging.version import Version
from tap import Tap

if TYPE_CHECKING:
    from mypy_boto3_s3.service_resource import Bucket

from qai_hub_apps_test.utils.aws import (
    ASSETS_S3_BASE,
    QAIHM_PUBLIC_S3_BUCKET,
    QAIHM_PUBLIC_WHEELS_S3_BUCKET,
    RELEASES_S3_PREFIX,
    attempt_with_s3_credentials_warning,
    get_qaihm_s3,
    s3_prefix_for,
    upload_public_file,
)

# Single object in the wheels bucket served for the `.../qai-hub-apps/` path.
_INDEX_KEY = "qaiha-index.html"


class PublishParser(Tap):
    wheel_path: Path  # Path to the built .whl to upload
    cli_version: str  # CLI version used for S3 path


def upload_wheel(
    wheel_path: Path, bucket: Bucket, s3_prefix: str, cli_version: str
) -> None:
    """Upload a built wheel to S3, co-located with the registry + app zips."""
    upload_public_file(
        bucket, wheel_path, f"{s3_prefix}/{cli_version}/{wheel_path.name}"
    )


def list_release_wheels(assets_bucket: Bucket) -> list[str]:
    """Return assets-bucket keys of every qai-hub-apps wheel (prod + dev)."""

    def _list() -> list[str]:
        return [
            obj.key
            for obj in assets_bucket.objects.filter(Prefix=f"{RELEASES_S3_PREFIX}/")
            if obj.key.endswith(".whl")
        ]

    return attempt_with_s3_credentials_warning(_list)


def _anchor(key: str) -> str:
    """An <a> tag for a wheel key: %2B-encoded href, verbatim filename as text."""
    name = key.rsplit("/", 1)[-1]
    # quote() encodes '+' -> '%2B'; keep path slashes readable.
    return f'  <a href="{ASSETS_S3_BASE}/{quote(key)}">{name}</a><br>'


def _wheel_version(key: str) -> Version:
    """Parse the PEP 440 version from a wheel key, for sorting."""
    _, version, _, _ = parse_wheel_filename(key.rsplit("/", 1)[-1])
    return version


def render_pep503_index(wheel_keys: list[str]) -> str:
    """Render a PEP 503 project page linking to the given assets-bucket wheel keys.

    Wheels are grouped under "Stable" / "Nightly" headings, newest version first. pip
    ignores the headings and only follows the <a href> anchors.
    """
    dev_marker = f"{RELEASES_S3_PREFIX}/dev/"
    stable = sorted(
        (k for k in wheel_keys if dev_marker not in k),
        key=_wheel_version,
        reverse=True,
    )
    nightly = sorted(
        (k for k in wheel_keys if dev_marker in k), key=_wheel_version, reverse=True
    )

    sections = []
    for heading, keys in (("Stable", stable), ("Nightly", nightly)):
        if not keys:
            continue
        anchors = "\n".join(_anchor(k) for k in keys)
        sections.append(f"  <h2>{heading}</h2>\n{anchors}")
    body = "\n".join(sections)

    return (
        "<!DOCTYPE html>\n"
        '<html><head><meta name="pypi:repository-version" content="1.0">'
        "<title>qai-hub-apps</title></head>\n"
        f"<body>\n  <h1>qai-hub-apps</h1>\n{body}\n</body></html>\n"
    )


def write_index(wheels_bucket: Bucket, html: str) -> None:
    """Overwrite the single qaiha-index.html object in the wheels bucket."""

    def _put() -> None:
        wheels_bucket.put_object(
            Key=_INDEX_KEY,
            Body=html.encode("utf-8"),
            ContentType="text/html",
            ACL="public-read",
        )

    attempt_with_s3_credentials_warning(_put)
    print(f"Wrote index to s3://{QAIHM_PUBLIC_WHEELS_S3_BUCKET}/{_INDEX_KEY}")


def main() -> None:
    args = PublishParser().parse_args()
    assets_bucket, _ = get_qaihm_s3(QAIHM_PUBLIC_S3_BUCKET, requires_admin=False)

    upload_wheel(
        args.wheel_path,
        assets_bucket,
        s3_prefix_for(args.cli_version),
        args.cli_version,
    )

    wheel_keys = list_release_wheels(assets_bucket)
    print(f"Detected {len(wheel_keys)} wheel(s) to index:")
    for key in sorted(wheel_keys):
        print(f"  {key}")

    html = render_pep503_index(wheel_keys)
    wheels_bucket, _ = get_qaihm_s3(QAIHM_PUBLIC_WHEELS_S3_BUCKET, requires_admin=False)
    write_index(wheels_bucket, html)


if __name__ == "__main__":
    main()
