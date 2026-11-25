# ------------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ------------------------------------------------------------------------

<#
.SYNOPSIS
    Run large language models exported by Qualcomm AI Hub.

.DESCRIPTION
    This script handles all of the tasks required to run supported LLMs on
    a Windows on Snapdragon device. Use the ai-hub-models Python package to
    export the model and provide the associated tokenizer.json. Next, this
    will produce all necessary configuration files, set up QNN/Genie, and
    run genie-t2t-run.exe.

.EXAMPLE
    # 1. Export the model
    python -m qai_hub_models.models.llama_v3_2_3b_instruct.export --device "Snapdragon X Elite CRD" --skip-inferencing --skip-profiling --output-dir genie_bundle

    # 2. Download tokenizer to genie_bundle\tokenizer.json (see ..\README.md for download links).

    # 3. Run the model on this device.
    & .\RunLlm.ps1 -ModelName llama_v3_2_3b_instruct -BundleRoot genie_bundle -RawPrompt "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHow many dogs are there in space?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

.LINK
    https://github.com/qualcomm/ai-hub-apps/tree/main/tutorials/llm_on_genie/

.LINK
    https://github.com/qualcomm/ai-hub-models/
#>
param (
    [Parameter(Mandatory = $true,
               HelpMessage = "The name of the AI Hub model to run,`n  e.g., llama_v3_2_3b_instruct.")]
    [string]$ModelName,

    [Parameter(Mandatory = $true,
               HelpMessage = "Path to the directory containing the model's context binaries.`n  e.g., genie_bundle")]
    [string]$BundleRoot,

    [Parameter(HelpMessage = "Path to the model's tokenizer.json. Default: `$BundleRoot\tokenizer.json")]
    [string]$TokenizerPath,

    [Parameter(Mandatory = $true,
               HelpMessage = "A raw prompt string, including system prompt header. Please note that this is model-specific; details in ..\README.md.`n  e.g., <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>")]
    [string]$RawPrompt,

    [Parameter(HelpMessage = "Path to the QNN SDK.")]
    [string]$QnnSdkRoot = $Env:QNN_SDK_ROOT,

    [Parameter(HelpMessage = "Override path to the ai-hub-models repository.")]
    [string]$QaihmRoot,

    [Parameter(HelpMessage = "This device's soc_model. 60 = Snapdragon X Elite / X Plus.")]
    [int]$SocModel = 60,

    [Parameter(HelpMessage = "This device's dsp_arch. v73 = Snapdragon X Elite / X Plus and many others.")]
    [string]$DspArch = "v73"
)

# Make utility functions available
$ScriptRoot = (Resolve-Path -Path "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)").Path
. $ScriptRoot\LlmUtils.ps1

if (-Not $TokenizerPath) {
    $TokenizerPath = Join-Path $BundleRoot "tokenizer.json"
}
$TokenizerPath = (Resolve-Path -Path $TokenizerPath)

if (-Not $QaihmRoot) {
    $QaihmRoot = (Resolve-Path -Path $ScriptRoot\..\..\..)
}

# Turn $BundleRoot into an absolute path
$BundleRoot = (Resolve-Path -Path $BundleRoot)

# Location of the sample Genie configs in the ai-hub-models repo
$GenieConfigsDir = "$QaihmRoot\tutorials\llm_on_genie\configs"
$HtpConfigTemplatePath = "$GenieConfigsDir\htp\htp_backend_ext_config.json.template"
$GenieConfigTemplatePath = "$GenieConfigsDir\genie\$ModelName.json"

# Ensure the SDKs and other files are available
if (-Not (Test-SdkDirectories -QnnSdkRoot $QnnSdkRoot -QaihmRoot $QaihmRoot)) {
    exit 1
}
if (-Not (Test-ModelFiles -GenieConfigTemplatePath $GenieConfigTemplatePath -BundleRoot $BundleRoot -TokenizerPath $TokenizerPath)) {
    exit 1
}

# Where the config files need to go
$HtpConfigPath = "$BundleRoot/htp_backend_ext_config.json"
$GenieConfigPath = "$BundleRoot/genie_config.json"

# Substitute two placeholder strings in the HTP config.
# Using -Encoding ascii in the Out-File invocation is required to avoid
# cryptic JSON parsing errors.
Write-Status "Creating $HtpConfigPath from template in $HtpConfigTemplatePath."
(Get-Content -Path $HtpConfigTemplatePath) `
    -replace "`"soc_model`": <TODO>", "`"soc_model`": $SocModel" `
    -replace "`"dsp_arch`": `"<TODO>`"", "`"dsp_arch`": `"$DspArch`"" |
    Out-File -Encoding ascii "$HtpConfigPath"
ThrowIfLastFailed

# Disable mmap on this Windows device and update some paths so we don't have to
# change directories to run Genie.
# As above, ascii encoding is required.
Write-Status "Creating $GenieConfigPath from template in $GenieConfigTemplatePath."
(Get-Content -Path "$GenieConfigTemplatePath") `
    -replace "`"use-mmap`": true", "`"use-mmap`": false" ` |
    Update-ModelPaths -GenieConfigTemplatePath $GenieConfigTemplatePath -BundleRoot $BundleRoot -TokenizerPath $TokenizerPath |
    Out-File -Encoding ascii "$GenieConfigPath"
ThrowIfLastFailed

Write-Status "Setting ADSP_LIBRARY_PATH = `"$BundleRoot`"."
$OldAdspLibraryPath = $Env:ADSP_LIBRARY_PATH
$Env:ADSP_LIBRARY_PATH = $BundleRoot

# I haven't been able to get Genie to run from $QnnSdkRoot so copy it into $BundleRoot and run from there.
# Note that setting ADSP_LIBRARY_PATH and PATH _is_ sufficient for QNN (including the validator),
# but _not_ for Genie on Windows.
Write-Status "Copying Genie and $DspArch binaries to $BundleRoot."
Copy-Genie -QnnSdkRoot $QnnSdkRoot -Destination $BundleRoot -DspArch $DspArch

# Check if QNN thinks our host is set up properly.
# Note that qnn-platform-validator always returns 0 so we can't easily bail on failure.
# Its output is a malformed CSV so even that is hard to use :-/
& $BundleRoot\qnn-platform-validator.exe --backend dsp --testBackend

# Run Genie
$GenieCmd = "$BundleRoot\genie-t2t-run.exe"
$GenieArgs = @("-c", "$BundleRoot\genie_config.json", "-p", $RawPrompt)
Write-Status "Running $GenieCmd $GenieArgs"
& $GenieCmd $GenieArgs

$Env:ADSP_LIBRARY_PATH = $OldAdspLibraryPath
