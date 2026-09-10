# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
FROM mcr.microsoft.com/windows/server:ltsc2025

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

# These six files are byte-identical across all windows apps, so the layer is shared. The
# C:\app bind mount shadows this copy at run time; it is only needed here.
COPY scripts/msvc_utils.ps1 \
     scripts/winget_utils.ps1 \
     scripts/interactive.ps1 \
     scripts/load_versions.ps1 \
     scripts/retry.ps1 \
     scripts/versions.env \
     C:/app/scripts/

RUN powershell -Command \
    ". C:\\app\\scripts\\msvc_utils.ps1; \
     Install-MSVC; \
     Install-WingetPackage -Id 'Microsoft.Git' -ExtraArgs @('--source', 'winget')"
CMD ["powershell"]
