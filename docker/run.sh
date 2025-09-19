#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-heir:dev}"

# Build the development image using the Dockerfile in docker/
docker buildx build --platform linux/amd64 -t "${IMAGE_TAG}" -f "${SCRIPT_DIR}/Dockerfile" "${PROJECT_ROOT}"

# Run the container with the repo root bind mounted into the container workspace
docker run --user heiruser --platform linux/amd64 -v "${PROJECT_ROOT}:/home/heiruser/heir" -it "${IMAGE_TAG}"
