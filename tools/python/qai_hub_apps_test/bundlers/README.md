# App Bundler

The bundler packages an app's source code, shared library code, shared shell scripts, and version configuration into a standalone directory or zip that an end-user can run without any knowledge of the internal repository structure.

## Overview

Bundling is driven by `bundle_app()` in `__init__.py`. It:

1. Reads the app's `info.yaml` to verify it is a Python app.
2. Inside a temp directory, runs three steps in order:
   - **`bundle_source`** — copies app source and shared library code
     (per-language: e.g. Python apps get the imported `qai_hub_apps_utils`
     modules plus a merged `requirements.txt`).
   - **`bundle_scripts`** — processes `install_*.sh` / `install_*.ps1`,
     `test.sh` / `test.ps1`, and the generated `build.sh` / `build.ps1`
     (see [Building Apps](../../../../CONTRIBUTING.md#building-apps)), rewrites
     their source lines, transitively copies referenced shared scripts to
     `scripts/`, and copies `versions.env` (shell bundler's job).
   - **Finalize** — copies or zips the temp dir to the final output location.

```
bundle_app(app_id, output_dir)
  │
  ├─ bundle_source()    → source + shared library code (per-language)
  ├─ bundle_scripts()   → scripts/ dir + versions.env + rewritten source lines
  └─ finalize           → copy to <app_id>/ or zip to <app_id>.zip
```

### Output layout

```
<output_dir>/<app_id>/
  install_runtime.sh       # hand-written; source lines rewritten to scripts/
  install_build.sh         # optional; also rewritten
  install_runtime.ps1      # hand-written; source lines rewritten to scripts/
  build.sh / build.ps1     # generated (generate_build_scripts); also rewritten
  scripts/
    versions.env           # copied from apps/_shared/scripts/versions.env
    load_versions.sh       # copied (if transitively referenced)
    apt_utils.sh           # copied (if referenced)
    python_utils.sh        # copied (if referenced)
    pip_utils.sh           # copied (if referenced)
    load_versions.ps1      # copied (if transitively referenced)
    winget_utils.ps1       # copied (if referenced)
    python_utils.ps1       # copied (if referenced)
    pip_utils.ps1          # copied (if referenced)
  requirements.txt         # merged pip requirements
  <app source files>
  qai_hub_apps_utils/      # shared utils modules (if imported by app)
```

---

## Writing Install Scripts

App developers write their own `install_*.sh` (Ubuntu/Linux) and `install_*.ps1` (Windows) scripts at the app root. The bundler handles path rewriting automatically.

### Naming convention

- `install_runtime.sh` / `install_runtime.ps1` — runtime environment setup
- `install_build.sh` / `install_build.ps1` — build-time setup
- Any `install_*.sh` / `install_*.ps1` at the **app root** is processed
- `test.sh` / `test.ps1` (hand-written) and `build.sh` / `build.ps1` (generated) are processed the same way

> `build.sh` / `build.ps1` are **not** hand-written — they are produced by
> `generate_build_scripts` (see [Building Apps](../../../../CONTRIBUTING.md#building-apps)).
> The bundler treats them like any other script: it rewrites their `source` lines
> and pulls in the shared scripts they reference.

### Available shared utilities

All shared scripts live in `apps/_shared/scripts/`. Each utility script auto-loads version defaults from `versions.env` via `load_versions.sh` / `load_versions.ps1`.

> The entry script **must** export `QAIHA_APP_ROOT` (the app directory) before
> sourcing any shared utility — `load_versions` errors out if it is unset. If an
> `apps/<id>/versions.override.env` exists next to the entry script, its keys are
> layered on top of the global `versions.env` (override wins); the bundler merges
> it into the bundled `scripts/versions.env`.

#### Ubuntu/Linux (`.sh`)

| Script | Functions | Description |
|--------|-----------|-------------|
| `apt_utils.sh` | `install_apt_pkg <pkg> [args]` | Install an apt package (idempotent) |
| `python_utils.sh` | `install_python` | Install `python$PYTHON_VERSION` + venv, dev headers, uv |
| `pip_utils.sh` | `install_pip_deps <req_file> [venv_dir]` | Create `.venv` and install requirements via uv |

#### Windows (`.ps1`)

| Script | Functions | Description |
|--------|-----------|-------------|
| `winget_utils.ps1` | `Install-WingetPackage -Id <id> [-ExtraArgs]` | Install a winget package (idempotent) |
| `python_utils.ps1` | `Install-Python` | Install `Python.Python.$PYTHON_VERSION` + uv |
| `pip_utils.ps1` | `Install-PipDeps -RequirementsFile <path> [-VenvDir]` | Create `.venv` and install requirements via uv |

### Minimal example

```bash
# apps/my_ubuntu_app/install_runtime.sh
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QAIHA_APP_ROOT="$SCRIPT_DIR"   # required before sourcing shared scripts

source ../../_shared/scripts/python_utils.sh
source ../../_shared/scripts/apt_utils.sh
source ../../_shared/scripts/pip_utils.sh

install_python
install_apt_pkg libcairo2-dev
install_pip_deps "$SCRIPT_DIR/requirements.txt"
```

```powershell
# apps/my_windows_app/install_runtime.ps1
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:QAIHA_APP_ROOT = $ScriptDir   # required before sourcing shared scripts

. ..\..\..\_shared\scripts\python_utils.ps1
. ..\..\..\_shared\scripts\winget_utils.ps1
. ..\..\..\_shared\scripts\pip_utils.ps1

Install-Python
Install-WingetPackage -Id "Microsoft.Git"
Install-PipDeps -RequirementsFile (Join-Path $PSScriptRoot "requirements.txt")
```

### How bundling rewrites source paths

When the bundler processes an app it calls `collect_and_rewrite_scripts()` on each install script in a single pass:

1. Reads the script line by line.
2. For each `source` / `. ` / `& ` line, attempts to resolve the referenced path into `apps/_shared/scripts/` using the following strategies in order:
   - **Static path** (absolute or relative) — resolved directly against the script's directory.
   - **`_shared/scripts/` marker fallback** — for relative paths containing `_shared/scripts/`, the tail after the marker is resolved from `shared_scripts_root`. This handles install scripts copied to a temp directory where the original relative path (e.g. `../../_shared/scripts/foo.sh`) no longer resolves from the new location.
   - **`$VAR/path` fallback** — for paths starting with `$` (e.g. `"$_SOME_DIR/foo.sh"` used inside shared scripts), the variable prefix is stripped up to the first `/` or `\` and the tail is resolved from `shared_scripts_root`. Pure variable references with no separator (e.g. `$_VERSIONS_FILE`) are silently skipped since they cannot be statically resolved.
3. If a resolved path falls under `shared_scripts_root` and exists, the line is rewritten and the path recorded. If resolution fails for a statically-determinable reference, a warning is emitted.
4. Writes the rewritten content back.

Then `find_transitive_scripts()` performs a DFS from each directly-referenced shared script, following their own source lines (using the same resolution logic) to collect the full transitive closure.

Rewrite rules:
- Bash: `source ../../_shared/scripts/foo.sh` → `source "$(dirname "${BASH_SOURCE[0]}")/scripts/foo.sh"`
- PowerShell: `. ..\_shared\scripts\foo.ps1` → `. "$PSScriptRoot\scripts\foo.ps1"`

`$PSScriptRoot` is set automatically by PowerShell. Subdirectory structure within `_shared/scripts/` is preserved in the bundle.

### Convention for shared scripts sourcing siblings

Shared scripts that source other shared scripts (e.g. `apt_utils.sh` sourcing `load_versions.sh`) must use the `$VAR/filename` pattern so the bundler can extract the tail:

```bash
# Correct — bundler extracts "load_versions.sh" from the $VAR/ prefix
_APT_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_APT_UTILS_DIR/load_versions.sh"
```

```powershell
# Correct — bundler extracts "load_versions.ps1" from the $VAR\ prefix
$_WingetUtilsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. "$_WingetUtilsDir\load_versions.ps1"
```

Pure variable references like `source "$_VERSIONS_FILE"` are silently skipped (the bundler cannot resolve them statically). Non-script files such as `versions.env` are always copied unconditionally and do not need to be referenced in source lines.

---

## Architecture Reference

```
bundlers/
  __init__.py       bundle_app() — orchestrates temp dir, bundle_source, bundle_scripts, finalize
  python/
    bundle.py       bundle_source() — copies source + qai_hub_apps_utils modules + requirements
    utils_collector.py  AST-based qai_hub_apps_utils import scanner
    utils_resolver.py   Locates the qai_hub_apps_utils package root
    requirements.py   requirements.txt parsing and merging
  shell/
    bundle.py       collect_and_rewrite_scripts(), find_transitive_scripts(), bundle_scripts()
    script_resolver.py  Locates apps/_shared/scripts/

apps/
  _shared/
    python/         qai_hub_apps_utils shared Python package
    scripts/
      versions.env          Hand-maintained KEY=VALUE version defaults
      load_versions.sh      Sources versions.env into bash environment
      load_versions.ps1     Parses versions.env into PowerShell variables
      apt_utils.sh          install_apt_pkg
      python_utils.sh       install_python
      pip_utils.sh          install_pip_deps
      winget_utils.ps1      Install-WingetPackage
      python_utils.ps1      Install-Python
      pip_utils.ps1         Install-PipDeps
```
