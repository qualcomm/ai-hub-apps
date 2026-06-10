# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
$ErrorActionPreference = "Stop"

$AppDir = "C:\Temp\TestContent\app"
$LogDir = "C:\Temp\QDC_logs"
$UseDocker = [System.Convert]::ToBoolean("<<USE_DOCKER>>")
$ExitCode = 0

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Start-Transcript -Path "$LogDir\script.log" -Append

Set-Location $AppDir

if ($UseDocker) {
    Write-Warning "Windows ARM64 devices may not support Docker containers and this may fail. The recommended approach is to run without Docker."
    . "$AppDir\scripts\qairt_utils.ps1"

    $ImageName = "aiha-app-test"

    Write-Host "Building Docker image ..."
    docker build `
        --build-arg BUILD_TYPE=runtime `
        -t $ImageName `
        -f "$AppDir\windows.dockerfile" `
        $AppDir
    if ($LASTEXITCODE -ne 0) {
        $ExitCode = $LASTEXITCODE
    } else {
        Write-Host "Running inside container ..."
        docker run --rm `
            -v "${env:QAIRT_ROOT}:${env:QAIRT_ROOT}" `
            -w "C:\app" `
            $ImageName `
            powershell -Command "<<RUN_COMMAND>>"
        $ExitCode = $LASTEXITCODE
    }
} else {
    # QDC automated jobs are executed via TShell (analogous to ADB for Android), which
    # runs commands on the device as SYSTEM. SYSTEM has no user profile and cannot access
    # winget (a per-user app). The fix: register a one-shot Scheduled Task as hcktest
    # (the always-auto-logged-in QDC interactive user) and start it immediately.
    # The Task Scheduler service dispatches the task into hcktest's active desktop session,
    # giving it full user context (PATH, winget, etc.) without needing a password.
    $UserAccount = "hcktest"
    $TempScript  = "C:\Temp\run_as_user.ps1"
    $UserLog     = "$LogDir\user_run.log"
    $TaskName    = "AIHubAppTest"

    # Write the actual install + test logic to a temp file that runs as hcktest.
    @"
`$ErrorActionPreference = 'Stop'
Start-Transcript -Path "$UserLog" -Append
Set-Location "$AppDir"
if (Test-Path "install_runtime.ps1") {
    Write-Host "Running install_runtime.ps1 ..."
    . .\install_runtime.ps1
}
Write-Host "Running app command ..."
<<RUN_COMMAND>>
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
}
Write-Host "Exiting with $ExitCode"
Stop-Transcript
exit $ExitCode
