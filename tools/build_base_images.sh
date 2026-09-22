# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Build the android/ubuntu docker base images and point the app build at it:
#
#   bash tools/build_base_images.sh android
#   export QAIHA_BASE_IMAGE=qai-hub-apps-android-base:local
#   qai-hub-apps run <app_id>
#
# Windows containers cannot be built on linux; use build_base_images.ps1 there.
set -euo pipefail

# shellcheck source=tools/ci/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ci/common.sh"

REPO_ROOT="$(repo_root)"
TAG="local"
NO_CACHE=false
PUSH=false
REGISTRY="ghcr.io/qcom-ai-hub"
PLATFORMS=()

usage() {
    cat <<'EOF'
Usage: build_base_images.sh [options] [android|ubuntu|all]

Builds a docker base image locally and prints the QAIHA_BASE_IMAGE to export.
Defaults to 'all'.

Options:
  --tag TAG         Image tag (default: local).
  --no-cache        Build without the docker layer cache.
  --push            Build and push to the registry instead of loading locally.
  --registry HOST   Registry namespace for --push (default: ghcr.io/qcom-ai-hub).
  -h, --help        Show this help.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --no-cache) NO_CACHE=true ;;
        --push) PUSH=true ;;
        --tag)
            [ $# -ge 2 ] || { echo "error: --tag needs a value" >&2; exit 2; }
            TAG="$2"
            shift
            ;;
        --registry)
            [ $# -ge 2 ] || { echo "error: --registry needs a value" >&2; exit 2; }
            REGISTRY="$2"
            shift
            ;;
        -h|--help) usage; exit 0 ;;
        all) PLATFORMS=(android ubuntu) ;;
        android|ubuntu) PLATFORMS+=("$1") ;;
        windows)
            echo "error: windows containers cannot be built on linux; use build_base_images.ps1." >&2
            exit 2
            ;;
        *) echo "error: unexpected argument '$1'" >&2; usage >&2; exit 2 ;;
    esac
    shift
done
[ ${#PLATFORMS[@]} -gt 0 ] || PLATFORMS=(android ubuntu)

# The android dockerfile COPYs from scripts/, so the context must be a directory
# with the shared scripts at that path -- not the repo root.
build_base_image() {
    local platform="$1"
    local image="qai-hub-apps-${platform}-base"
    local ctx
    local build_args=()

    [ "$NO_CACHE" = false ] || build_args+=(--no-cache)
    [ "$PUSH" = false ] || image="${REGISTRY}/${image}"
    image="${image}:${TAG}"

    ctx="$(mktemp -d)"
    # shellcheck disable=SC2064  # expand $ctx now, not at trap time
    trap "rm -rf '$ctx'" RETURN
    mkdir -p "$ctx/scripts"
    cp -r "$REPO_ROOT/apps/_shared/scripts/." "$ctx/scripts/"
    cp "$REPO_ROOT/tools/docker/${platform}.dockerfile" "$ctx/Dockerfile"

    echo "::step::Building $image for $(uname -m)"
    if [ "$PUSH" = true ]; then
        docker buildx build "${build_args[@]}" --push --provenance=false \
            --label "org.opencontainers.image.revision=$(git -C "$REPO_ROOT" rev-parse HEAD)" \
            -t "$image" "$ctx"
    else
        docker build "${build_args[@]}" \
            --label "org.opencontainers.image.revision=$(git -C "$REPO_ROOT" rev-parse HEAD)" \
            -t "$image" "$ctx"
    fi
    echo "::done::$image"
    BUILT_IMAGES+=("$platform=$image")
}

BUILT_IMAGES=()
for platform in "${PLATFORMS[@]}"; do
    build_base_image "$platform"
done

echo
if [ "$PUSH" = true ]; then
    echo "Pushed ${#BUILT_IMAGES[@]} base image(s) for arch $(uname -m):"
    for entry in "${BUILT_IMAGES[@]}"; do
        printf '  %-8s %s\n' "${entry%%=*}" "${entry#*=}"
    done
else
    # The app build reads a single QAIHA_BASE_IMAGE, so export the one matching the
    # app being run.
    echo "Built ${#BUILT_IMAGES[@]} base image(s) for arch $(uname -m). Export the one for the app you run:"
    for entry in "${BUILT_IMAGES[@]}"; do
        printf '  %-8s export QAIHA_BASE_IMAGE=%s\n' "${entry%%=*}" "${entry#*=}"
    done
fi
