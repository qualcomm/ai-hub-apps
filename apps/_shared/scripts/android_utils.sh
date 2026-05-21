#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Android build toolchain installation utilities.
#
# Functions:
#   install_android_sdk [--force]
#       Install SDKMAN, Java, Gradle, Android SDK command-line tools,
#       and NDK. Skips each component if already present. After sourcing this script,
#       $ANDROID_HOME, $JAVA_HOME and $GRADLE_HOME holds the SDK root directories. $PATH is updated automatically.
#
# Usage: source android_utils.sh
# ---------------------------------------------------------------------
_ANDROID_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_ANDROID_UTILS_DIR/load_versions.sh"
# shellcheck disable=SC1091
source "$_ANDROID_UTILS_DIR/apt_utils.sh"

ANDROID_HOME="${ANDROID_HOME:-$HOME/android-sdk}"
SDKMAN_DIR="${SDKMAN_DIR:-$HOME/.sdkman}"
export JAVA_HOME="$SDKMAN_DIR/candidates/java/current"
export GRADLE_HOME="$SDKMAN_DIR/candidates/gradle/current"
export PATH="$JAVA_HOME/bin:$GRADLE_HOME/bin:$PATH"
export ANDROID_HOME

install_android_sdk() {
    local force=0
    if [ "${1:-}" = "--force" ]; then force=1; fi

    $SUDO apt-get update -q
    install_apt_pkgs unzip zip wget

    # SDKMAN
    if [ ! -d "$SDKMAN_DIR" ] || [ "$force" -eq 1 ]; then
        echo "::step::Installing SDKMAN"
        local sdkman_script
        sdkman_script="$(wget -qO- "https://get.sdkman.io?ci=true")" \
            || { echo "error: failed to download SDKMAN installer" >&2; return 1; }
        bash <<< "$sdkman_script"
        echo "::done::SDKMAN"
    else
        echo "::skip::SDKMAN"
    fi
    # sdkman-init.sh and sdk subcommands reference positional params ($3) and vars
    # before assigning them, and internally run commands that return non-zero as part
    # of normal operation. Suspend errexit (-e) and nounset (-u) around all sdk calls.
    set +eu
    # shellcheck disable=SC1091
    source "$SDKMAN_DIR/bin/sdkman-init.sh"
    set -eu

    # Java
    if [ ! -d "$SDKMAN_DIR/candidates/java/$JAVA_SDK_VERSION" ] || [ "$force" -eq 1 ]; then
        echo "::step::Installing Java $JAVA_SDK_VERSION"
        set +eu; sdk install java "$JAVA_SDK_VERSION"; set -eu
        echo "::done::Java $JAVA_SDK_VERSION"
    else
        echo "::skip::Java $JAVA_SDK_VERSION"
        set +eu; sdk use java "$JAVA_SDK_VERSION"; set -eu
    fi

    # Gradle
    if [ ! -d "$SDKMAN_DIR/candidates/gradle/$GRADLE_VERSION" ] || [ "$force" -eq 1 ]; then
        echo "::step::Installing Gradle $GRADLE_VERSION"
        set +eu; sdk install gradle "$GRADLE_VERSION"; set -eu
        echo "::done::Gradle $GRADLE_VERSION"
    else
        echo "::skip::Gradle $GRADLE_VERSION"
        set +eu; sdk use gradle "$GRADLE_VERSION"; set -eu
    fi

    # Android command-line tools
    if [ ! -d "$ANDROID_HOME/cmdline-tools/latest" ] || [ "$force" -eq 1 ]; then
        echo "::step::Installing Android command-line tools"
        local tmp_zip arch
        tmp_zip="$(mktemp --suffix=.zip)"
        arch="$(uname -m)"
        if [ "$arch" = "aarch64" ] || [ "$arch" = "arm64" ]; then
            # Android build tools (AAPT2, etc.) are x86_64 only. On ARM64 hosts,
            # install QEMU user-space emulation and the x86_64 runtime libraries.
            echo "::step::Detected aarch64, installing x86_64 emulation support"
            install_apt_pkgs qemu-user-static libc6-amd64-cross libgcc-s1-amd64-cross
            mkdir -p /lib64
            ln -sf /usr/x86_64-linux-gnu/lib64/ld-linux-x86-64.so.2 /lib64/ld-linux-x86-64.so.2
            ln -sf /usr/x86_64-linux-gnu/lib /lib/x86_64-linux-gnu
            # NDK clang is an x86_64 binary that needs an x86_64 glibc libz.so.1.
            # No cross-package exists for zlib, so extract directly from the Debian amd64 deb.
            local zlib_tmp
            zlib_tmp="$(mktemp -d)"
            wget -q "http://ftp.us.debian.org/debian/pool/main/z/zlib/zlib1g_1.3.dfsg+really1.3.2-3_amd64.deb" \
                -O "${zlib_tmp}/zlib1g_amd64.deb"
            dpkg-deb -x "${zlib_tmp}/zlib1g_amd64.deb" "${zlib_tmp}/zlib1g_amd64"
            $SUDO cp "${zlib_tmp}/zlib1g_amd64"/usr/lib/x86_64-linux-gnu/libz.so.* /usr/x86_64-linux-gnu/lib/
            rm -rf "$zlib_tmp"
            echo "::done::x86_64 emulation support"
        fi
        mkdir -p "$ANDROID_HOME/cmdline-tools"
        wget -q "https://dl.google.com/android/repository/commandlinetools-linux-${ANDROID_CMDLINE_TOOLS}.zip" \
            -O "$tmp_zip"
        unzip -q "$tmp_zip" -d "$ANDROID_HOME/cmdline-tools"
        mv "$ANDROID_HOME/cmdline-tools/cmdline-tools" "$ANDROID_HOME/cmdline-tools/latest"
        rm "$tmp_zip"
        echo "::done::Android command-line tools"
    else
        echo "::skip::Android command-line tools"
    fi


    if [ "${NON_INTERACTIVE:-}" = "true" ]; then
        set +eu; yes | "$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager" --licenses > /dev/null; set -eu
    else
        "$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager" --licenses
    fi
    # SDK packages
    echo "::step::Installing Android SDK packages"
    "$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager" \
        "platform-tools" \
        "build-tools;${ANDROID_COMPILE_API}.0.0" \
        "platforms;android-${ANDROID_TARGET_API}" \
        "ndk;${ANDROID_NDK_VERSION}"
    echo "::done::Android SDK packages"
}
