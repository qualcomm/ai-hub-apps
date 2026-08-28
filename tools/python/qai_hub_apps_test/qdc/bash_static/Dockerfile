# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Build a fully static bash from GNU source on Alpine (musl), for the QDC Android
# appium host (Alpine, ships only sh/ash). The resulting binary has no shared-library
# dependencies, so it runs on any matching-arch Alpine with nothing else installed.
# build.sh builds this image and extracts /bash into the vendored path.
FROM alpine:3.18

RUN apk add --no-cache build-base gnupg curl tar

ARG BASH_VERSION=5.2.21

WORKDIR /src
# Fetch the GNU bash release and verify it against Chet Ramey's (bash's maintainer)
# public key, served from his canonical page over HTTPS.
RUN curl -fsSLO "https://ftp.gnu.org/gnu/bash/bash-${BASH_VERSION}.tar.gz" \
    && curl -fsSLO "https://ftp.gnu.org/gnu/bash/bash-${BASH_VERSION}.tar.gz.sig" \
    && curl -fsSL "https://tiswww.case.edu/php/chet/gpgkey.asc" | gpg --batch --import \
    && gpg --batch --verify "bash-${BASH_VERSION}.tar.gz.sig" "bash-${BASH_VERSION}.tar.gz" \
    && tar xzf "bash-${BASH_VERSION}.tar.gz"

WORKDIR /src/bash-${BASH_VERSION}
# --enable-static-link: fully static binary; --disable-readline: no readline/ncurses
# dependency (line editing is unused for running scripts); --without-bash-malloc: use
# musl's malloc (bash's internal malloc is unsafe with static musl).
# -Wl,-z,muldefs: bash ships its own strtoimax, which musl's libc also defines; under a
# static link that collides. muldefs resolves to the first definition (bash's own, which
# links before -lc), which is a correct implementation.
RUN LDFLAGS="-Wl,-z,muldefs" ./configure --enable-static-link --disable-readline \
        --disable-nls --without-bash-malloc \
    && make -j"$(nproc)" \
    && strip bash \
    && cp bash /bash
