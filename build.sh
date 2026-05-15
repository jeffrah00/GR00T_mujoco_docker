#!/usr/bin/env bash
# Build the GR00T N1.7 + MuJoCo Docker image for Jetson Thor / JetPack 7.2
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-gr00t-thor}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "==> Building ${IMAGE_NAME}:${IMAGE_TAG}"
echo "    This clones Isaac-GR00T and installs all dependencies."
echo "    Expect 20-40 min on first build (torchcodec source compile)."
echo ""

docker build \
    --network=host \
    --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
    --file Dockerfile \
    .

echo ""
echo "==> Build complete: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "    Run with:  ./run.sh"
