@echo off
setlocal EnableDelayedExpansion

REM Locate the QNN EP nuget package directory
set "PACKAGE_DIR="
for /d %%f in ("%~dp0packages\Microsoft.ML.OnnxRuntime.QNN.*") do (
    set "PACKAGE_DIR=%%f\runtimes\win-arm64\native"
    goto :found_package
)

echo ERROR: QNN EP nuget package directory not found.
exit /b 1

:found_package
echo Resolved QNN EP directory: !PACKAGE_DIR!

REM Set the output directory (passed from Visual Studio)
set "OUTPUT_DIR=%~1"
if "%OUTPUT_DIR%"=="" (
    echo ERROR: Output directory not specified.
    exit /b 1
)

echo Copying missing QNN EP files from !PACKAGE_DIR! to %OUTPUT_DIR%

REM Required in older versions (<1.21)
copy /Y "!PACKAGE_DIR!\onnxruntime_providers_shared.dll" "%OUTPUT_DIR%"
REM Required in most recent (as of 1.22)
copy /Y "!PACKAGE_DIR!\onnxruntime_providers_qnn.dll" "%OUTPUT_DIR%"
copy /Y "!PACKAGE_DIR!\Qnn*.dll" "%OUTPUT_DIR%"
copy /Y "!PACKAGE_DIR!\lib*.so" "%OUTPUT_DIR%"

endlocal
