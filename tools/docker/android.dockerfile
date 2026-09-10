# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Android build image. Holds the whole build toolchain (SDKMAN, JDK, Gradle,
# Android SDK/NDK) and no app source: the app directory is bind-mounted at /app
# when the container runs, so editing app source never invalidates this image.
ARG REGISTRY_PREFIX=""

FROM ${REGISTRY_PREFIX}ubuntu:24.04

ARG INSTALL_QUALCOMM_CA="false"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN if [ "$INSTALL_QUALCOMM_CA" = "true" ]; then \
        mkdir -p /usr/local/share/ca-certificates/qualcomm.com \
        && wget --no-check-certificate -P /usr/local/share/ca-certificates/qualcomm.com \
            https://pki.qualcomm.com/qc_root_g2_cert.crt \
            https://pki.qualcomm.com/ssl_v2_cert.crt \
            https://pki.qualcomm.com/ssl_v4_cert.crt \
        && update-ca-certificates \
        && wget --no-check-certificate \
            -O /usr/local/share/ca-certificates/qualcomm.com/nscacert.crt \
            https://github.qualcomm.com/raw/netskope-ssl/download/main/nscacert.cer \
        && update-ca-certificates; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    software-properties-common \
    sudo \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

ENV NON_INTERACTIVE=true

# Point Python/pip/requests at the system CA bundle, which carries the Qualcomm
# roots when INSTALL_QUALCOMM_CA=true. Harmless otherwise; the path always exists.
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

# Import the Qualcomm root into the JDK truststore so Gradle can reach internal
# hosts. Runs after the SDK install so $JAVA_HOME exists.
RUN if [ "$INSTALL_QUALCOMM_CA" = "true" ]; then \
        source /app/scripts/android_utils.sh \
        && keytool -import -noprompt -trustcacerts -alias qualcommroot \
            -file /usr/local/share/ca-certificates/qualcomm.com/nscacert.crt \
            -keystore "$JAVA_HOME/lib/security/cacerts" \
            -storepass changeit; \
    fi

CMD ["bash"]
