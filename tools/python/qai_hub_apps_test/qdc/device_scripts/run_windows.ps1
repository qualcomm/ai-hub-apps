# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
$ErrorActionPreference = "Stop"

$AppDir = "C:\Temp\TestContent\app"
$LogDir = "C:\Temp\QDC_logs"
# set QAIHA_APP_ROOT for shared utils
$env:QAIHA_APP_ROOT = $AppDir
$ExitCode = 0

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Start-Transcript -Path "$LogDir\script.log" -Append

Set-Location $AppDir

# QDC automated jobs are executed via TShell (analogous to ADB for Android), which
# runs commands on the device as SYSTEM. SYSTEM has no user profile and cannot access
# winget (a per-user app). The fix: register a one-shot Scheduled Task as hcktest
# (the always-auto-logged-in QDC interactive user) and start it immediately, giving
# the CLI install + run full user context (PATH, winget, etc.). Windows ARM64 does not
# support containers, so the app always runs natively via the CLI's launch.ps1.
$UserAccount = "hcktest"
$TempScript  = "C:\Temp\run_as_user.ps1"
$UserLog     = "$LogDir\user_run.log"
$TaskName    = "AIHubAppTest"

# Write the actual install + test logic to a temp file that runs as hcktest.
@"
`$ErrorActionPreference = 'Stop'
`$env:NON_INTERACTIVE = 'true'
`$env:QAI_HUB_APPS_EXPERIMENTAL = '1'
`$env:QAI_HUB_APPS_LOG_LEVEL = 'debug'
Start-Transcript -Path "$UserLog" -Append
Set-Location "$AppDir"
Write-Host "Installing Python (<<PYTHON_VERSION>>) using winget ..."
# The task has no console, so winget must never prompt: accept both agreements and
# fail (rather than ask) on anything else. --silent keeps the installer headless.
`$list = winget list --id '<<PYTHON_VERSION>>' --exact --accept-source-agreements 2>&1
if (`$LASTEXITCODE -eq 0 -and (`$list -match '<<PYTHON_VERSION>>')) {
    Write-Host "<<PYTHON_VERSION>> is already installed."
} else {
    winget install --id '<<PYTHON_VERSION>>' --exact --silent --source winget --accept-package-agreements --accept-source-agreements --disable-interactivity
    if (`$LASTEXITCODE -ne 0) { throw "winget install <<PYTHON_VERSION>> failed with exit code `$LASTEXITCODE." }
}
# winget puts python on the machine PATH, which this already-running process cannot see.
`$env:PATH = [System.Environment]::GetEnvironmentVariable('PATH', 'Machine') + ';' + [System.Environment]::GetEnvironmentVariable('PATH', 'User')
Write-Host "Installing qai-hub-apps CLI ..."
# The CLI is a bundled wheel; its dependencies resolve from PyPI.
`$pipArgs = @('--pre')
# Install the CLI into a dedicated venv.
`$CliVenv = 'C:\Temp\cli-venv'
python -m venv `$CliVenv
. "`$CliVenv\Scripts\Activate.ps1"
python -m pip install @pipArgs '<<CLI_SPEC>>'
Write-Host "Running app test ..."
`$testArgs = @('--app-path', "$AppDir", '--device', '<<DEVICE_NAME>>', '--model-id', '<<MODEL_ID>>')
if ('<<REGISTRY_PATH>>' -ne '') { `$testArgs += @('--registry', '<<REGISTRY_PATH>>') }
qai-hub-apps test @testArgs
`$cmdExitCode = `$LASTEXITCODE
Stop-Transcript
exit `$cmdExitCode
"@ | Out-File -FilePath $TempScript -Encoding UTF8

$action  = New-ScheduledTaskAction -Execute "PowerShell" `
               -Argument "-NoProfile -ExecutionPolicy Bypass -NonInteractive -File `"$TempScript`""
$trigger  = New-ScheduledTaskTrigger -Once -At "1990-01-01T00:00:00"
# Allow the task to run regardless of power state (very important for QDC, task stays 'Queued' without this)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
                -ExecutionTimeLimit (New-TimeSpan -Hours 2)

# Remove any leftover task with the same name from a previous run to avoid conflicts.
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -Action $action -Trigger $trigger -Settings $settings `
    -TaskName $TaskName -User $UserAccount -Force | Out-Null
Write-Host "Registered scheduled task '$TaskName' as $UserAccount."

Start-ScheduledTask -TaskName $TaskName
Write-Host "Started scheduled task '$TaskName'."

# Phase 1: wait up to 10 minutes for the task to start.
$timeout = 600
$elapsed = 0
while ((Get-ScheduledTask -TaskName $TaskName).State -ne "Running") {
    Start-Sleep -Seconds 10
    $elapsed += 10
    if ($elapsed -ge $timeout) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
        Write-Error "Timed out waiting for '$TaskName' to start after $timeout seconds."
        $ExitCode = 1
        break
    }
}

# Phase 2: wait indefinitely for the task to finish.
# The task itself enforces a 2-hour execution limit via ExecutionTimeLimit.
if ($ExitCode -eq 0) {
    Write-Host "Task is running. Waiting for completion ..."
    while ((Get-ScheduledTask -TaskName $TaskName).State -eq "Running") {
        Start-Sleep -Seconds 10
    }
    $ExitCode = (Get-ScheduledTaskInfo -TaskName $TaskName).LastTaskResult
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}

# Print the user session log so QDC captures it in the job output.
if (Test-Path $UserLog) {
    Write-Host "=== Output from user session ==="
    Get-Content $UserLog
    Write-Host "================================"
}
Write-Host "Exiting with $ExitCode"
Stop-Transcript
exit $ExitCode
