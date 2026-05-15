#!/usr/bin/env bash
# Launch the GR00T N1.7 + MuJoCo container on Jetson Thor / JetPack 7.2
#
# Usage:
#   ./run.sh                      # interactive shell, headless EGL rendering
#   ./run.sh --display            # interactive shell with X11 display (viewer)
#   ./run.sh --headless <cmd>     # run a command non-interactively
#
# Examples:
#   ./run.sh --display
#   ./run.sh --headless "python gr00t/eval/open_loop_eval.py \
#       --model-path nvidia/GR00T-N1.7-3B \
#       --dataset-path demo_data/droid_sample \
#       --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
#       --traj-ids 0 --action-horizon 16"

set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-gr00t-thor}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-gr00t-mujoco}"

DISPLAY_MODE="headless"    # headless | display
EXTRA_CMD=()
EXTRA_DOCKER_ARGS=()
MUJOCO_GL="egl"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --display)
            DISPLAY_MODE="display"
            shift
            ;;
        --headless)
            DISPLAY_MODE="headless"
            shift
            ;;
        --)
            shift
            EXTRA_CMD=("$@")
            break
            ;;
        *)
            EXTRA_CMD=("$@")
            break
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Display / X11 setup
# ---------------------------------------------------------------------------
if [[ "$DISPLAY_MODE" == "display" ]]; then
    MUJOCO_GL="glfw"
    DISPLAY_VAL="${DISPLAY:-:0}"

    # Allow the container to connect to the host X server
    xhost +local:docker 2>/dev/null || true

    EXTRA_DOCKER_ARGS+=(
        "-e" "DISPLAY=${DISPLAY_VAL}"
        "-e" "MUJOCO_GL=glfw"
        "-e" "PYOPENGL_PLATFORM="
        "-v" "/tmp/.X11-unix:/tmp/.X11-unix:rw"
    )
    echo "==> Display mode: X11 (DISPLAY=${DISPLAY_VAL})"
else
    EXTRA_DOCKER_ARGS+=(
        "-e" "MUJOCO_GL=egl"
        "-e" "PYOPENGL_PLATFORM=egl"
    )
    echo "==> Display mode: headless EGL (no DISPLAY needed)"
fi

# ---------------------------------------------------------------------------
# Build docker run command
# ---------------------------------------------------------------------------
DOCKER_ARGS=(
    "--rm"
    "--name" "${CONTAINER_NAME}"
    "--runtime" "nvidia"         # Jetson Thor NVIDIA container runtime
    "--ipc=host"
    "--ulimit" "memlock=-1"
    "--ulimit" "stack=67108864"
    "-e" "NVIDIA_VISIBLE_DEVICES=all"
    "-e" "NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute"
    # Mount a local checkpoints directory for model weights persistence
    "-v" "${PWD}/checkpoints:/checkpoints"
    # Mount output directory for evaluation plots
    "-v" "${PWD}/output:/tmp/open_loop_eval"
    "${EXTRA_DOCKER_ARGS[@]}"
    "${IMAGE_NAME}:${IMAGE_TAG}"
)

mkdir -p "${PWD}/checkpoints" "${PWD}/output"

if [[ ${#EXTRA_CMD[@]} -gt 0 ]]; then
    echo "==> Running: ${EXTRA_CMD[*]}"
    docker run "${DOCKER_ARGS[@]}" bash -c "${EXTRA_CMD[*]}"
else
    echo "==> Starting interactive shell in ${IMAGE_NAME}:${IMAGE_TAG}"
    echo "    Working directory: /workspace/gr00t"
    echo ""
    docker run -it "${DOCKER_ARGS[@]}" bash
fi
