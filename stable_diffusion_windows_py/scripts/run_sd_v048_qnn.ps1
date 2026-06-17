$ErrorActionPreference = "Stop"

$AppRoot = Split-Path -Parent $PSScriptRoot
$Python = "C:\Users\hirok\miniconda3\envs\AI_Hub_SD\python.exe"

Set-Location $AppRoot

& $Python -c "import onnxruntime as ort; print('ORT:', ort.__version__); print('Providers:', ort.get_available_providers())"

& $Python demo.py `
  --prompt "A girl taking a walk at sunset" `
  --num-steps 20 `
  --seed 47 `
  --output-dir generated_default_v048_20steps
