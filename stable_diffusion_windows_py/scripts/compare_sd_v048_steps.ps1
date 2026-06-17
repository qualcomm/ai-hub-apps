$ErrorActionPreference = "Stop"

$AppRoot = Split-Path -Parent $PSScriptRoot
$Python = "C:\Users\hirok\miniconda3\envs\AI_Hub_SD\python.exe"
$Prompt = "A girl taking a walk at sunset"
$Seed = 47

Set-Location $AppRoot

& $Python -c "import onnxruntime as ort; print('ORT:', ort.__version__); print('Providers:', ort.get_available_providers())"

foreach ($Steps in @(5, 10, 20)) {
  $Out = "generated_v048_${Steps}steps_seed${Seed}"
  Write-Host "Running steps=$Steps output=$Out"
  & $Python demo.py `
    --prompt $Prompt `
    --num-steps $Steps `
    --seed $Seed `
    --output-dir $Out
}
