# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
FROM mcr.microsoft.com/windows/server:ltsc2025

ARG BUILD_TYPE="runtime"

# Install Visual C++ Redistributable (required by many native packages)
WORKDIR C:\\Downloads
ADD https://aka.ms/vs/16/release/vc_redist.x64.exe C:\\Downloads\\vcredist_x64.exe
RUN C:\\Downloads\\vcredist_x64.exe /install /passive /norestart /log out.txt

# Install winget from GitHub release
RUN powershell -Command \
    "Invoke-WebRequest -Uri 'https://github.com/microsoft/winget-cli/releases/download/v1.29.250/Microsoft.DesktopAppInstaller_8wekyb3d8bbwe.msixbundle' -OutFile C:\\Downloads\\winget.zip; \
     Expand-Archive -Path C:\\Downloads\\winget.zip -DestinationPath C:\\Downloads\\winget_outer -Force; \
     Copy-Item C:\\Downloads\\winget_outer\\AppInstaller_x64.msix C:\\Downloads\\winget_x64.zip; \
     Expand-Archive -Path C:\\Downloads\\winget_x64.zip -DestinationPath C:\\winget -Force; \
     Remove-Item -Recurse -Force C:\\Downloads\\winget.zip, C:\\Downloads\\winget_outer, C:\\Downloads\\winget_x64.zip; \
     [Environment]::SetEnvironmentVariable('PATH', $env:PATH + ';C:\\winget', 'Machine')"

ENV NON_INTERACTIVE=true

WORKDIR C:\\app

# set QAIHA_APP_ROOT for shared scripts
ENV QAIHA_APP_ROOT=C:\\app

COPY . C:\\app

RUN powershell -Command \
    "if ($env:BUILD_TYPE -eq 'build' -and (Test-Path 'install_build.ps1')) { \
        . .\\install_build.ps1 \
    } elseif (Test-Path 'install_runtime.ps1') { \
        . .\\install_runtime.ps1 \
    }"

CMD ["powershell"]
