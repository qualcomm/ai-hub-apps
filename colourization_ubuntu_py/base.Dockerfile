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

COPY scripts/apt_utils.sh \
     scripts/interactive.sh \
     scripts/load_versions.sh \
     scripts/python_utils.sh \
     scripts/retry.sh \
     scripts/sudo.sh \
     scripts/versions.env \
     /app/scripts/

RUN source /app/scripts/python_utils.sh && install_python

RUN source /app/scripts/apt_utils.sh && install_apt_pkgs \
    libcairo2-dev \
    pkg-config \
    libgirepository1.0-dev \
    gir1.2-gstreamer-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-libav \
    v4l-utils \
    v4l2loopback-utils \
    libsndfile1 \
    libportaudio2 \
    ffmpeg \
    unzip \
    && rm -rf /var/lib/apt/lists/*

CMD ["bash"]
