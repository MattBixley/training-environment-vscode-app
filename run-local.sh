#!/usr/bin/env bash
#
# Run the training session on your own machine, without deploying anything.
#
# This starts the same container image the cluster runs, with the GPU emulator
# and JupyterLab, and prints a URL to open. What it does not reproduce is the
# Open OnDemand wrapper around it: no login page, no k8s, no NFS home
# directories, no LDAP. Everything a learner actually does inside the session -
# nvidia-smi, nvtop, sbatch, seff, the exercises, PyTorch - behave identically.
#
# Usage:
#   ./run-local.sh                          # the whole fleet, 1 GiB per card
#   ./run-local.sh --fleet l4:4:2GiB        # four L4s with 2 GiB each
#   ./run-local.sh --fleet 'l4,a100:full'   # an L4 and a full-size A100
#   ./run-local.sh --build                  # build from this checkout first
#   ./run-local.sh --shell                  # drop to a terminal instead
#
# --fleet takes name[:count][:vram] entries, comma separated. A bare number is
# a count, so l4:2 is two cards and l4:2GiB is one card with 2 GiB. Card names
# are the ones --gpus-per-node wants: l4, a100_40, a100, h100, pro_6000.
#
set -euo pipefail

REGISTRY_IMAGE="ghcr.io/nesi/training-environment-jupyter-gpu-app"

# Derive the tag from this checkout rather than hardcoding one. A pinned
# default goes stale the moment a release is cut, and then this script quietly
# runs an old image while claiming to be the current app - which is exactly
# what happened with v0.1.1. Falls back to :latest outside a git checkout.
default_image() {
    local tag
    tag=$(git -C "$(dirname "${BASH_SOURCE[0]}")" describe --tags --abbrev=0 2>/dev/null || true)
    echo "${REGISTRY_IMAGE}:${tag:-latest}"
}

IMAGE="$(default_image)"
PORT=8888
# Every card Mahuika has, one of each, deliberately shrunk to 1 GiB so that
# running out of GPU memory is cheap to demonstrate.
FLEET="l4,a100_40,a100,h100,pro_6000"
CPUS="4"
MEMORY="8g"
BUILD=0
SHELL_ONLY=0
ONE_DEVICE=""
ONE_VRAM=""
ONE_COUNT=""

# \s is a GNU extension that BSD sed (macOS) does not understand, so spell the
# optional space out longhand.
usage() { sed -n '2,21p' "$0" | sed 's/^#\{1,\} \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)   IMAGE="$2"; shift 2 ;;
        --port)    PORT="$2"; shift 2 ;;
        --fleet)   FLEET="$2"; shift 2 ;;
        # The old single-card spelling, kept so existing notes keep working.
        # Any of the three switches the node back to one kind of card.
        --device)  ONE_DEVICE="$2"; shift 2 ;;
        --vram)    ONE_VRAM="$2"; shift 2 ;;
        --gpus)    ONE_COUNT="$2"; shift 2 ;;
        --cpus)    CPUS="$2"; shift 2 ;;
        --memory)  MEMORY="$2"; shift 2 ;;
        --build)   BUILD=1; shift ;;
        --shell)   SHELL_ONLY=1; shift ;;
        -h|--help) usage ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# --device/--vram/--gpus together describe one kind of card, which is what this
# script used to offer. Any of them replaces the mixed fleet entirely, rather
# than editing it, so the result does not depend on the order of the arguments.
if [[ -n "$ONE_DEVICE" || -n "$ONE_VRAM" || -n "$ONE_COUNT" ]]; then
    FLEET="${ONE_DEVICE:-l4}:${ONE_COUNT:-1}:${ONE_VRAM:-100MiB}"
fi

if [[ "$BUILD" == "1" ]]; then
    echo "Building from this checkout..."
    IMAGE="gpu-app:local"
    docker build -t "$IMAGE" "$(dirname "$0")/docker"
fi

# The cluster runs amd64, so an image we have to pull should be the amd64 one
# even on an Apple Silicon Mac - it runs under emulation, slowly but correctly.
# An image already present locally is used as-is, whatever it was built for;
# forcing a platform on it would trigger a pointless pull.
PLATFORM=()
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    if [[ "$(uname -m)" == "arm64" || "$(uname -m)" == "aarch64" ]]; then
        PLATFORM=(--platform linux/amd64)
        echo "Note: pulling the amd64 image; it runs under emulation and is slow"
        echo "      to start. Use --build for a native image if you are iterating."
    fi
fi

# -i needs a terminal on stdin. Without this the script cannot be run from a
# pipe, a CI job or the background, which is exactly how you would smoke-test it.
TTY=(-t)
[[ -t 0 ]] && TTY=(-it)

# macOS still ships bash 3.2, where expanding an empty array under `set -u` is
# an "unbound variable" error. The ${a[@]+"${a[@]}"} form expands to nothing
# when the array is empty instead of failing.
COMMON=(
    --rm
    "${TTY[@]}"
    ${PLATFORM[@]+"${PLATFORM[@]}"}
    --cpus "$CPUS"
    --memory "$MEMORY"
    -e GPUEMU_FLEET="$FLEET"
    -e TERM=xterm-256color
)

if [[ "$SHELL_ONLY" == "1" ]]; then
    echo "Starting a shell with the emulator running. Try: nvidia-smi, nvtop, sbatch"
    exec docker run "${COMMON[@]}" "$IMAGE" bash -lc 'gpuemu-ctl start >/dev/null && exec bash -l'
fi

cat <<BANNER

  Starting the GPU training session locally.

    Emulated GPUs: $FLEET
    Resources    : $CPUS CPUs, $MEMORY RAM
    Image        : $IMAGE

  JupyterLab will be at:  http://localhost:${PORT}/lab
  Exercises are under     /root/gpu-training/
  Press Ctrl-C to stop.

BANNER

exec docker run "${COMMON[@]}" -p "${PORT}:8888" "$IMAGE" bash -lc '
    set -e
    gpuemu-ctl start >/dev/null

    # Mirror what template/script.sh.erb does on the cluster, so the session
    # you see locally has the same contents as the deployed one.
    mkdir -p "${HOME}/gpu-training"
    rsync --ignore-existing -a /opt/gpu-training/workshop/ "${HOME}/gpu-training/"
    gpuemu-seed-workshop --quiet || true

    nvidia-smi -L
    echo

    cd "${HOME}"
    exec jupyter lab \
        --ip 0.0.0.0 --port 8888 --no-browser --allow-root \
        --ServerApp.token="" --ServerApp.password="" \
        --ServerApp.root_dir="${HOME}"
'
