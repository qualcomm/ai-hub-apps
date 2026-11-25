# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Android build image. Holds the whole build toolchain (SDKMAN, JDK, Gradle,
# Android SDK/NDK) and no app source: the app directory is bind-mounted at /app
# when the container runs, so editing app source never invalidates this image.
ARG REGISTRY_PREFIX=""

FROM ${REGISTRY_PREFIX}ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    software-properties-common \
    sudo \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

ENV NON_INTERACTIVE=true

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt

WORKDIR /app

# set QAIHA_APP_ROOT for shared scripts
ENV QAIHA_APP_ROOT=/app

# These seven files are byte-identical across all android apps, so the layer -- and the SDK install -- is shared.
# The /app bind mount shadows this copy at run time; it is only needed here.
COPY scripts/android_utils.sh \
     scripts/apt_utils.sh \
     scripts/interactive.sh \
     scripts/load_versions.sh \
     scripts/retry.sh \
     scripts/sudo.sh \
     scripts/versions.env \
     /app/scripts/

# The whole Android toolchain. This is what install_build.sh does natively, and
# it is app-agnostic, so it belongs in the image rather than in the container.
RUN source /app/scripts/android_utils.sh && install_android_sdk

CMD ["bash"]
