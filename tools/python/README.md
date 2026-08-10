# qai_hub_apps_test

Internal build, test, and CI tooling for Qualcomm® AI Hub Apps. This package provides utilities
for bundling apps, parsing app metadata, running on-device tests, and generating the CLI registry.

## Installation

Must be installed in editable mode from the repository root — the package resolves paths relative
to the repo tree at runtime:

```bash
pip install -e tools/python/
```

## Package structure

```
qai_hub_apps_test/
├── configs/          # YAML config loaders (info.yaml, versions.yaml, etc.)
├── bundlers/         # App packaging — Python source + shell scripts —> zip
├── utils/
│   ├── paths.py      # REPOSITORY_ROOT, APPS_ROOT, get_all_apps(), find_app_dir()
│   ├── aws/          # S3 credential validation and download helpers
│   ├── models/       # Model download and compatibility checking
│   ├── android/      # Gradle build and version verification
│   └── windows/      # Visual Studio / MSBuild helpers
├── qdc/              # Qualcomm Device Cloud job submission and polling
├── scripts/          # CLI entry points (generate_registry, generate_build_scripts, publish_wheel_s3) + build-script templates/
└── test/             # pytest fixtures and integration tests
```

## Key APIs

| Function / Class | Module | Purpose |
|---|---|---|
| `QAIHAAppInfo.from_app(path)` | `configs/info_yaml.py` | Load and validate `info.yaml` for an app |
| `bundle_app(app_dir, out_dir)` | `bundlers/__init__.py` | Bundle an app into a zip or directory |
| `get_all_apps()` | `utils/paths.py` | Yield paths to every app directory in the repo |
| `find_app_dir(app_id)` | `utils/paths.py` | Locate an app directory by its ID |
| `VersionsRegistry.load()` | `configs/versions_yaml.py` | Load SDK / tool version pins |
| `AssetBases.load()` | `configs/asset_bases_yaml.py` | Load S3 and GitHub base URLs |

## CLI scripts

```bash
# Generate registry.yaml from all published apps (optionally build + upload to S3)
python -m qai_hub_apps_test.scripts.generate_registry --output_dir cli/qai_hub_apps/

# Generate each in-scope app's build.sh / build.ps1 from the Jinja templates
# (see CONTRIBUTING.md "Building Apps"). CI regenerates with --scope test.
python -m qai_hub_apps_test.scripts.generate_build_scripts --scope test
```

## Configuration files

| File | Class | Description |
|---|---|---|
| `apps/<id>/info.yaml` | `QAIHAAppInfo` | App metadata — name, languages, models, status, etc. |
| `apps/asset_bases.yaml` | `AssetBases` | S3 bucket base and GitHub repo base URLs |

## `info.yaml` schema constraints

### `model_file_paths` — same parent directory required

All entries in `model_file_paths` must share the same parent directory (we want to remove this restriction; [tetracode#19248](https://github.com/qcom-ai-hub/tetracode/issues/19248)). For example, this is valid:

```yaml
model_file_paths:
  - models/detector.tflite
  - models/classifier.tflite
```

But this is **not** valid and will raise a `ValidationError` at load time:

```yaml
model_file_paths:
  - models/detector.tflite
  - assets/classifier.tflite  # different parent — rejected
```
The CLI fetch logic copies the entire model asset directory (including `metadata.json`,
`LICENSE`, and any other bundled files) into a single destination directory derived from the common
parent. Allowing paths with different parents would require splitting non-model files across multiple
directories with no clear rule for where they land, leading to inconsistent app layouts.

If an app genuinely needs model files in different directories, the recommended approach is to place
them all in a shared subdirectory and have the app reference them from there.
