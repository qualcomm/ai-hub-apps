# Genie PowerShell script for Windows

This script simplifies the steps in the [LLM On-Device Deployment](..)
tutorial. It is assumed that you have followed the tutorial and prepared a
Genie bundle with the context binaries and tokenizer.

From the parent directory (`llm_on_genie`), please run the command in a
PowerShell terminal:

```powershell
.\powershell\RunLlm1.ps1
```

For more information, run:

```powershell
Get-Help .\powershell\RunLlm1.ps1
```

If it says that you do not have permission to run PowerShell scripts, you may
need to enable this first with this command:

```powershell
Set-ExecutionPolicy -Scope CurrentUser Unrestricted -Force
```

## UTF-8 support

If you want to use Unicode characters in your prompts, we recommend globally
enabling UTF-8 on Windows:

* Open Control Panel -> Region -> Administrative -> Change system locale...
* Tick the "Beta: Use Unicode UTF-8 for worldwide language support"
* Restart Windows

If you want to change this only for a single PowerShell session, execute:

```powershell
chcp 65001
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```
