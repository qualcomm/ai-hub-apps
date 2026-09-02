# ------------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ------------------------------------------------------------------------

function Copy-Genie {
<#
.SYNOPSIS
    Copy enough of QNN to be able to run genie-t2t-run.exe.
#>

    param(
        [Parameter(Mandatory = $true)]
        [string]$QnnSdkRoot,

        [Parameter(Mandatory = $true)]
        [string]$Destination,

        [Parameter(Mandatory = $true)]
        [string]$DspArch,

        [boolean]$Clean = $true
    )

    if ($Clean) {
        Remove-Item -Path $Destination\*.* -Include *.cat, *.dll, *.exe, *.lib, *.so
    }

    $Arch = "aarch64-windows-msvc"

    Copy-Item -Destination $Destination `
        $QnnSdkRoot\bin\$Arch\genie-t2t-run.exe, `
        $QnnSdkRoot\bin\$Arch\qnn-platform-validator.exe, `
        $QnnSdkRoot\lib\$Arch\Genie.dll, `
        $QnnSdkRoot\lib\$Arch\PlatformValidatorShared.dll, `
        $QnnSdkRoot\lib\$Arch\QnnGenAiTransformer.dll, `
        $QnnSdkRoot\lib\$Arch\QnnGenAiTransformerModel.dll, `
        $QnnSdkRoot\lib\$Arch\QnnHtp.dll, `
        $QnnSdkRoot\lib\$Arch\QnnHtpNetRunExtensions.dll, `
        $QnnSdkRoot\lib\$Arch\QnnHtpPrepare.dll, `
        $QnnSdkRoot\lib\$Arch\QnnHtp${DspArch}Stub.dll, `
        $QnnSdkRoot\lib\$Arch\QnnHtp${DspArch}CalculatorStub.dll, `
        $QnnSdkRoot\lib\$Arch\QnnSystem.dll, `
        $QnnSdkRoot\lib\hexagon-$DspArch\unsigned\libCalculator_skel.so, `
        $QnnSdkRoot\lib\hexagon-$DspArch\unsigned\libQnnHtp${DspArch}.cat, `
        $QnnSdkRoot\lib\hexagon-$DspArch\unsigned\libQnnHtp${DspArch}Skel.so `
}

function Get-ModelFileNames {
<#
.SYNOPSIS
    Get a list of a model's context binaries.
#>
    param (
        [Parameter(Mandatory = $true)]
        [string]$GenieConfigPath
    )

    $genieConfig = (Get-Content -Path $GenieConfigPath) | ConvertFrom-Json
    return $genieConfig.dialog.engine.model.binary."ctx-bins"
}

function Test-ModelFiles {
<#
.SYNOPSIS
    Test if all files required to run a model are available.
#>
    param (
        [string]$GenieConfigTemplatePath,
        [string]$BundleRoot,
        [string]$TokenizerPath
    )

    if (-Not (Test-Path $GenieConfigTemplatePath)) {
        Write-Error "Genie config template $GenieConfigTemplatePath does not exist."
        return $false
    }

    $modelFileNames = Get-ModelFileNames -GenieConfigPath $GenieConfigTemplatePath
    foreach ($modelFileName in $modelFileNames) {
        $modelFilePath = Join-Path $BundleRoot $modelFileName
        if (-Not (Test-Path $modelFilePath)) {
            Write-Error "Model file $modelFilePath does not exist."
            return $false
        }
    }

    if (-Not (Test-Path $TokenizerPath)) {
        Write-Error "Tokenizer not found in $TokenizerPath. See LLM on Genie tutorial for more info."
        return $false
    }

    return $true
}

function Test-SdkDirectories {
<#
.SYNOPSIS
    Check if the QNN and QAIHM directories are valid.
#>

    param(
        [Parameter(Mandatory = $true)]
        [string]$QnnSdkRoot,

        [Parameter(Mandatory = $true)]
        [string]$QaihmRoot
    )

    if (-Not $QnnSdkRoot) {
        Write-Error "QNN SDK root has not been set. Set Env[QNN_SDK_ROOT] or pass -QnnSdkRoot argument."
        return $false
    }

    if (-Not (Test-Path -Path $QnnSdkRoot -PathType Container)) {
        Write-Error "QNN SDK root '$QnnSdkRoot' is not a directory."
        return $false
    }

    if (-Not $QaihmRoot) {
        Write-Error "ai-hub-models repo root has not been set. Set Env[QAIHM_ROOT] or pass -QaihmRoot argument."
        return $false
    }

    if (-Not (Test-Path -Path $QaihmRoot -PathType Container)) {
        Write-Error "ai-hub-models repo root '$QaihmRoot' is not a directory."
        return $false
    }

    return $true
}

function ThrowIfLastFailed() {
    param (
        [string]$ErrorMessage = "Last command failed."
    )
    if (-Not $?) {
        throw $ErrorMessage
    }
}

function Update-ModelPaths {
<#
.SYNOPSIS
    Prefix bare file references in the given genie.config text with our bundle root.
#>
    param (
        [Parameter(Mandatory = $true)]
        [string]$GenieConfigTemplatePath,

        [Parameter(Mandatory = $true)]
        [string]$BundleRoot,

        [Parameter(Mandatory = $true)]
        [string]$TokenizerPath,

        [Parameter(ValueFromPipeline = $true, Mandatory = $true)]
        [string]$GenieConfigLine
    )

    BEGIN {
        $BundleRootEscaped = $BundleRoot.Replace("\", "\\")
        $TokenizerPathEscaped = $TokenizerPath.Replace("\", "\\")
        $ModelFileNames = Get-ModelFileNames -GenieConfigPath $GenieConfigTemplatePath
    }

    PROCESS {
        foreach ($line in $GenieConfigLine) {
            $line = $line.Replace("`"path`": `"tokenizer.json`"", "`"path`": `"$TokenizerPathEscaped`"")
            $line = $line.Replace("`"extensions`": `"htp_backend_ext_config.json`"", "`"extensions`": `"$BundleRootEscaped\\htp_backend_ext_config.json`"")

            foreach ($modelFile in $ModelFileNames) {
                $line = $line.Replace("`"$modelFile`"", "`"$BundleRootEscaped\\$modelFile`"")
            }

            $line
        }
    }
}

function Write-Status {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    Write-Host -ForegroundColor Green $Message
}
