# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Download the JVM lint toolchain (JDK, google-java-format, ktlint, checkstyle)
# used by the pre-commit Java/Kotlin hooks into <repo>\.lint-tools\; skipped if already present.

$ErrorActionPreference = "Stop"

. "$PSScriptRoot\..\ci\common.ps1"


$JdkVersion = Get-Version "JDK_VERSION"
$GjfVersion = Get-Version "GOOGLE_JAVA_FORMAT_VERSION"
$KtlintVersion = Get-Version "KTLINT_VERSION"
$CheckstyleVersion = Get-Version "CHECKSTYLE_VERSION"

$RepoRoot = Get-RepoRoot
$ToolsDir = Join-Path $RepoRoot ".lint-tools"
New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null

# Microsoft Build of OpenJDK
$OsArch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
switch ($OsArch) {
    "X64"   { $JdkArch = "x64";     $JdkSha256 = "53f0c9ec64811a9ab968747076653e5500115db7230d244e4ec53577ca5ec8fc" }
    "Arm64" { $JdkArch = "aarch64"; $JdkSha256 = "709d65d7a24290f10fbe5a310c1f9b1edab12497785f6aef8395d4539c5f83c3" }
    default { throw "Unsupported architecture $OsArch for JDK download" }
}
$JdkDir = Join-Path $ToolsDir "jdk"
if (-not (Test-Path (Join-Path $JdkDir "bin\java.exe"))) {
    Write-Host "Installing JDK $JdkVersion ($JdkArch)"
    $jdkZip = Join-Path $env:TEMP "microsoft-jdk-$JdkVersion.zip"
    Invoke-DownloadAndVerify `
        -Url "https://aka.ms/download-jdk/microsoft-jdk-$JdkVersion-windows-$JdkArch.zip" `
        -Dest $jdkZip `
        -Sha256 $JdkSha256
    if (Test-Path $JdkDir) { Remove-Item -Recurse -Force $JdkDir }
    $tmpExtract = Join-Path $env:TEMP "jdk_extract_$([System.IO.Path]::GetRandomFileName())"
    Expand-Archive -Path $jdkZip -DestinationPath $tmpExtract -Force
    # The archive contains a single top-level jdk-<ver>+<build> directory.
    $inner = (Get-ChildItem -Directory $tmpExtract)[0].FullName
    Move-Item -Path $inner -Destination $JdkDir
    Remove-Item -Recurse -Force $tmpExtract, $jdkZip
} else {
    Write-Host "JDK already present at $JdkDir"
}

# google-java-format
$GjfJar = Join-Path $ToolsDir "google-java-format.jar"
if (-not (Test-Path $GjfJar)) {
    Invoke-DownloadAndVerify `
        -Url "https://github.com/google/google-java-format/releases/download/v$GjfVersion/google-java-format-$GjfVersion-all-deps.jar" `
        -Dest $GjfJar `
        -Sha256 "32342e7c1b4600f80df3471da46aee8012d3e1445d5ea1be1fb71289b07cc735"
} else {
    Write-Host "google-java-format already present at $GjfJar"
}

# ktlint
$KtlintJar = Join-Path $ToolsDir "ktlint.jar"
if (-not (Test-Path $KtlintJar)) {
    Invoke-DownloadAndVerify `
        -Url "https://github.com/pinterest/ktlint/releases/download/$KtlintVersion/ktlint" `
        -Dest $KtlintJar `
        -Sha256 "a3fd620207d5c40da6ca789b95e7f823c54e854b7fade7f613e91096a3706d75"
} else {
    Write-Host "ktlint already present at $KtlintJar"
}

# checkstyle
$CheckstyleJar = Join-Path $ToolsDir "checkstyle.jar"
if (-not (Test-Path $CheckstyleJar)) {
    Invoke-DownloadAndVerify `
        -Url "https://github.com/checkstyle/checkstyle/releases/download/checkstyle-$CheckstyleVersion/checkstyle-$CheckstyleVersion-all.jar" `
        -Dest $CheckstyleJar `
        -Sha256 "b88646a3bf32840d8c33f196fec89d7a379c8a142014206444d0aa0092fdb06e"
} else {
    Write-Host "checkstyle already present at $CheckstyleJar"
}

Write-Host "Lint toolchain ready in $ToolsDir"
