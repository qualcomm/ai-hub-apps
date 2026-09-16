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
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
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

CMD ["bash"]
