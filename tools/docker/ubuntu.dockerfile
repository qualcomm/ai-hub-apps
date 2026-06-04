ARG REGISTRY_PREFIX=""

FROM ${REGISTRY_PREFIX}ubuntu:24.04

ARG INSTALL_QUALCOMM_CA="false"
ARG BUILD_TYPE="runtime"

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

WORKDIR /app

COPY . /app

# When INSTALL_QUALCOMM_CA is true.
# Set SSL env vars before install scripts so Python/pip requests use the Qualcomm CA.
# Keytool runs after install_build.sh so JAVA_HOME is available for the JDK truststore update.
RUN if [ "$INSTALL_QUALCOMM_CA" = "true" ]; then \
        export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt; \
        export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt; \
        export PIP_CERT=/etc/ssl/certs/ca-certificates.crt; \
    fi \
    && if [ "$BUILD_TYPE" = "build" ] && [ -f install_build.sh ]; then \
        bash install_build.sh; \
        elif [ -f install_runtime.sh ]; then \
            QAIRT_INSTALL_SKIP=true bash install_runtime.sh; \
    fi \
    && if [ "$INSTALL_QUALCOMM_CA" = "true" ] && [ -f scripts/android_utils.sh ]; then \
        source scripts/android_utils.sh; \
        keytool -import -noprompt -trustcacerts -alias qualcommroot \
            -file /usr/local/share/ca-certificates/qualcomm.com/nscacert.crt \
            -keystore "$JAVA_HOME/lib/security/cacerts" \
            -storepass changeit; \
    fi

ENTRYPOINT ["bash", "-c", "if [ -f /app/scripts/qairt_utils.sh ]; then source /app/scripts/qairt_utils.sh && install_qairt; fi && exec \"$@\"", "--"]
CMD ["bash"]
