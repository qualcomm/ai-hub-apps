# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Docker helpers for the generated build.ps1 scripts.
#
# Functions:
#   Get-DockerImageTag -Prefix <str> -AppDir <path>
#       Return an image tag unique to <AppDir>, so two copies of the same app
#       in different directories never collide.
#   Resolve-BaseImage -Pinned <str>
#       Return $env:QAIHA_BASE_IMAGE if set (e.g. a locally built base), else <Pinned>.
#   Start-Container -Name <str> -Image <str> -CreateArgs <string[]>
#       Start container <Name>, creating it from <Image> if needed.
#   Install-QcCaCerts -Container <str> [-JavaKeystore]
#       When QC_INTERNAL_HOST=1, install the Qualcomm CA certificates in
#       <Container>; -JavaKeystore also imports them into the JDK truststore.
#
# Usage: . docker_utils.ps1
# ---------------------------------------------------------------------

function Get-DockerImageTag {
    param([string]$Prefix, [string]$AppDir)
    $Sha1 = [System.Security.Cryptography.SHA1]::Create()
    $Bytes = [System.Text.Encoding]::UTF8.GetBytes($AppDir)
    $Hash = ([System.BitConverter]::ToString($Sha1.ComputeHash($Bytes)) -replace '-', '').ToLower().Substring(0, 12)
    "$Prefix-$(Split-Path $AppDir -Leaf)-$Hash"
}

function Resolve-BaseImage {
    param([string]$Pinned)
    if ($env:QAIHA_BASE_IMAGE) { $env:QAIHA_BASE_IMAGE } else { $Pinned }
}

function Start-Container {
    param([string]$Name, [string]$Image, [string[]]$CreateArgs)
    try { docker start $Name 2>$null | Out-Null } catch {}
    if ($LASTEXITCODE -ne 0) {
        # A create or install that died earlier can leave a container behind under
        # this name in a state docker start rejects; replace it.
        try { docker rm -f $Name 2>$null | Out-Null } catch {}
        Write-Host "::step::Creating container $Name"
        # --init reaps the daemons that docker exec children reparent to PID 1.
        docker create --name $Name --init @CreateArgs $Image sleep infinity | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "::error::docker create failed (exit $LASTEXITCODE)"
            exit $LASTEXITCODE
        }
        docker start $Name | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "::error::docker start failed (exit $LASTEXITCODE)"
            exit $LASTEXITCODE
        }
        Write-Host "::done::container"
    }
}

function Install-QcCaCerts {
    param([string]$Container, [switch]$JavaKeystore)
    if ($env:QC_INTERNAL_HOST -ne "1") { return }
    Write-Host "::step::Installing Qualcomm CA certificates in $Container"
    docker exec $Container bash -c '
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
    if ($LASTEXITCODE -ne 0) {
        Write-Host "::error::Qualcomm CA certificates failed (exit $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
    if ($JavaKeystore) {
        docker exec $Container bash -c '
        set -euo pipefail
        source /app/scripts/android_utils.sh
        keystore="$JAVA_HOME/lib/security/cacerts"
        if ! keytool -list -alias qualcommroot -keystore "$keystore" -storepass changeit >/dev/null 2>&1; then
            keytool -import -noprompt -trustcacerts -alias qualcommroot \
                -file /usr/local/share/ca-certificates/qualcomm.com/nscacert.crt \
                -keystore "$keystore" -storepass changeit
        fi'
        if ($LASTEXITCODE -ne 0) {
            Write-Host "::error::JDK truststore import failed (exit $LASTEXITCODE)"
            exit $LASTEXITCODE
        }
    }
    Write-Host "::done::Qualcomm CA certificates"
}
