#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Docker helpers for the generated build.sh / launch.sh scripts. Docker runs
# through $SUDO when the caller has sourced sudo.sh.
#
# Functions:
#   docker_image_tag <prefix> <app_dir>
#       Echo an image tag unique to <app_dir>, so two copies of the same app
#       in different directories never collide.
#   resolve_base_image <pinned>
#       Echo $QAIHA_BASE_IMAGE if set (e.g. a locally built base), else <pinned>.
#   resolve_host_ip
#       Echo the host's outbound IP address, or nothing if it cannot be found.
#   replace_stale_container <container> <image>
#       Remove <container> if it was created from an older build of <image>.
#   start_container <container> <image> [<docker create arg>...]
#       Start <container>, creating it from <image> if needed.
#   install_qc_ca_certs <container> [--java-keystore]
#       When QC_INTERNAL_HOST=1, install the Qualcomm CA certificates in
#       <container>; --java-keystore also imports them into the JDK truststore.
#
# Usage: source docker_utils.sh
# ---------------------------------------------------------------------

docker_image_tag() {
    local prefix="$1" app_dir="$2" hash
    hash="$(printf '%s' "$app_dir" | sha1sum | cut -c1-12)"
    echo "$prefix-$(basename "$app_dir")-$hash"
}

resolve_base_image() {
    echo "${QAIHA_BASE_IMAGE:-$1}"
}

resolve_host_ip() {
    ip route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i < NF; i++) if ($i == "src") {print $(i + 1); exit}}'
}

# A container is pinned to the image id it was created from, not to the tag, so
# an existing container built from an older image would silently keep running
# the stale one.
replace_stale_container() {
    local container="$1" image="$2" built_image_id container_image_id
    built_image_id="$(${SUDO:-} docker image inspect -f '{{.Id}}' "$image")"
    container_image_id="$(${SUDO:-} docker container inspect -f '{{.Image}}' "$container" 2>/dev/null || true)"
    if [ -n "$container_image_id" ] && [ "$container_image_id" != "$built_image_id" ]; then
        echo "::step::Replacing container $container built from a stale image"
        ${SUDO:-} docker rm -f "$container" >/dev/null 2>&1 || true
        echo "::done::replace"
    fi
}

start_container() {
    local container="$1" image="$2"
    shift 2
    if ! ${SUDO:-} docker start "$container" >/dev/null 2>&1; then
        # A create or install that died earlier can leave a container behind under
        # this name in a state docker start rejects; replace it.
        ${SUDO:-} docker rm -f "$container" >/dev/null 2>&1 || true
        echo "::step::Creating container $container"
        # --init reaps the daemons that docker exec children reparent to PID 1.
        ${SUDO:-} docker create --name "$container" --init "$@" \
            "$image" sleep infinity >/dev/null
        ${SUDO:-} docker start "$container" >/dev/null
        echo "::done::container"
    fi
}

install_qc_ca_certs() {
    local container="$1" java_keystore="${2:-}"
    if [ "${QC_INTERNAL_HOST:-}" != "1" ]; then
        return 0
    fi
    echo "::step::Installing Qualcomm CA certificates in $container"
    ${SUDO:-} docker exec "$container" bash -c '
        set -euo pipefail
        cert_dir=/usr/local/share/ca-certificates/qualcomm.com
        if [ ! -f "$cert_dir/nscacert.crt" ]; then
            mkdir -p "$cert_dir"
            wget --no-check-certificate -P "$cert_dir" \
                https://pki.qualcomm.com/qc_root_g2_cert.crt \
                https://pki.qualcomm.com/ssl_v2_cert.crt \
                https://pki.qualcomm.com/ssl_v4_cert.crt
            wget --no-check-certificate -O "$cert_dir/nscacert.crt" \
                https://github.qualcomm.com/raw/netskope-ssl/download/main/nscacert.cer
            update-ca-certificates
        fi'

    if [ "$java_keystore" = "--java-keystore" ]; then
        ${SUDO:-} docker exec "$container" bash -c '
            set -euo pipefail
            . /app/scripts/android_utils.sh
            keystore="$JAVA_HOME/lib/security/cacerts"
            if ! keytool -list -alias qualcommroot -keystore "$keystore" -storepass changeit >/dev/null 2>&1; then
                keytool -import -noprompt -trustcacerts -alias qualcommroot \
                    -file /usr/local/share/ca-certificates/qualcomm.com/nscacert.crt \
                    -keystore "$keystore" -storepass changeit
            fi'
    fi
    echo "::done::Qualcomm CA certificates"
}
