$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:QAIHA_APP_ROOT = $ScriptDir

. "$PSScriptRoot\scripts\qairt_utils.ps1"

Install-Qairt
