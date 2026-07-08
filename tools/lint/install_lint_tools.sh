#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Download the JVM lint toolchain (JDK, google-java-format, ktlint, checkstyle)
# used by the pre-commit Java/Kotlin hooks into <repo>/.lint-tools/; skipped if already present.

set -euo pipefail

# shellcheck source=tools/lint/lint_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/lint_common.sh"


JDK_VERSION="$(get_version JDK_VERSION)"
GOOGLE_JAVA_FORMAT_VERSION="$(get_version GOOGLE_JAVA_FORMAT_VERSION)"
KTLINT_VERSION="$(get_version KTLINT_VERSION)"
CHECKSTYLE_VERSION="$(get_version CHECKSTYLE_VERSION)"

mkdir -p "$TOOLS_DIR"

# Microsoft Build of OpenJDK
case "$(uname -m)" in
    x86_64)          JDK_ARCH="x64";     JDK_SHA256="ef7ccf007efc473d3a19ffd84464da19905e2c11eaa31e96886c46dda27f94eb" ;;
    aarch64 | arm64) JDK_ARCH="aarch64"; JDK_SHA256="54bbc1873710d8ecd993af23b4122410fa532bffa40271b227bdf15861414ab3" ;;
    *) echo "error: unsupported architecture $(uname -m) for JDK download" >&2; exit 1 ;;
esac
JDK_DIR="$TOOLS_DIR/jdk"
if [ ! -x "$JDK_DIR/bin/java" ]; then
    echo "Installing JDK $JDK_VERSION ($JDK_ARCH)"
    jdk_tar="$(mktemp --suffix=.tar.gz)"
    download_and_verify \
        "https://aka.ms/download-jdk/microsoft-jdk-${JDK_VERSION}-linux-${JDK_ARCH}.tar.gz" \
        "$jdk_tar" \
        "$JDK_SHA256"
    rm -rf "$JDK_DIR"
    mkdir -p "$JDK_DIR"
    tar xzf "$jdk_tar" -C "$JDK_DIR" --strip-components=1
    rm -f "$jdk_tar"
else
    echo "JDK already present at $JDK_DIR"
fi

# google-java-format
GJF_JAR="$TOOLS_DIR/google-java-format.jar"
if [ ! -f "$GJF_JAR" ]; then
    download_and_verify \
        "https://github.com/google/google-java-format/releases/download/v${GOOGLE_JAVA_FORMAT_VERSION}/google-java-format-${GOOGLE_JAVA_FORMAT_VERSION}-all-deps.jar" \
        "$GJF_JAR" \
        "32342e7c1b4600f80df3471da46aee8012d3e1445d5ea1be1fb71289b07cc735"
else
    echo "google-java-format already present at $GJF_JAR"
fi

# ktlint
KTLINT_JAR="$TOOLS_DIR/ktlint.jar"
if [ ! -f "$KTLINT_JAR" ]; then
    download_and_verify \
        "https://github.com/pinterest/ktlint/releases/download/${KTLINT_VERSION}/ktlint" \
        "$KTLINT_JAR" \
        "a3fd620207d5c40da6ca789b95e7f823c54e854b7fade7f613e91096a3706d75"
else
    echo "ktlint already present at $KTLINT_JAR"
fi

# checkstyle
CHECKSTYLE_JAR="$TOOLS_DIR/checkstyle.jar"
if [ ! -f "$CHECKSTYLE_JAR" ]; then
    download_and_verify \
        "https://github.com/checkstyle/checkstyle/releases/download/checkstyle-${CHECKSTYLE_VERSION}/checkstyle-${CHECKSTYLE_VERSION}-all.jar" \
        "$CHECKSTYLE_JAR" \
        "b88646a3bf32840d8c33f196fec89d7a379c8a142014206444d0aa0092fdb06e"
else
    echo "checkstyle already present at $CHECKSTYLE_JAR"
fi

echo "Lint toolchain ready in $TOOLS_DIR"
