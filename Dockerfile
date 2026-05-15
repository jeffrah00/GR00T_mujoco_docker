# GR00T N1.7 + MuJoCo — Jetson Thor / JetPack 7.2
# Base: CUDA 13.0 + Ubuntu 24.04 (aarch64 / H100 on Jetson Thor)
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

# ---------------------------------------------------------------------------
# NVIDIA capabilities: "graphics" enables OpenGL/EGL for MuJoCo rendering
# ---------------------------------------------------------------------------
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
ENV DEBIAN_FRONTEND=noninteractive
ENV DOCKER_CONTAINER=1
ENV UV_PROJECT_ENVIRONMENT=/opt/gr00t-venv

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core tools
    git git-lfs curl ca-certificates tmux \
    # Build toolchain
    build-essential yasm cmake libtool pkg-config \
    autoconf automake texinfo \
    python3 python3-pip python3-venv \
    # FFmpeg (video processing required by GR00T)
    ffmpeg libavdevice-dev libavfilter-dev libavformat-dev \
    libavcodec-dev libavutil-dev libswresample-dev libswscale-dev \
    # FFmpeg codec support
    libass-dev libfreetype6-dev libvorbis-dev \
    # MuJoCo display — EGL (headless GPU) + GLX (physical display)
    # Note: libgl1-mesa-glx was removed in Ubuntu 24.04; use libglx-mesa0 instead
    libgl1 libglx-mesa0 libgl1-mesa-dri \
    libglu1-mesa libglu1-mesa-dev \
    libglew-dev libglfw3-dev \
    libegl1 libegl-mesa0 \
    # X11 (for interactive MuJoCo viewer via display/SSH -X)
    libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    libx11-xcb1 libxcb-dri2-0 libxcb-dri3-0 libxcb-present0 \
    xvfb x11-utils \
    # Python bindings helper
    pybind11-dev python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Clone Isaac-GR00T (full repo + submodules, LFS objects)
# ---------------------------------------------------------------------------
RUN git lfs install && \
    git clone --depth 1 --recurse-submodules \
        https://github.com/NVIDIA/Isaac-GR00T /workspace/gr00t

WORKDIR /workspace/gr00t

# ---------------------------------------------------------------------------
# Install uv (fast Python package manager used by GR00T)
# ---------------------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv

# ---------------------------------------------------------------------------
# Install all Thor-platform Python deps via the official installer
# This installs PyTorch (cu130), torchvision, flash-attn, etc. into
# /opt/gr00t-venv using the Thor-specific pyproject.toml + uv.lock
# ---------------------------------------------------------------------------
RUN bash scripts/deployment/thor/install_deps.sh

# ---------------------------------------------------------------------------
# Activate venv and expose NVIDIA package libraries (cuDNN, cuBLAS, cuDSS)
# ---------------------------------------------------------------------------
ENV VIRTUAL_ENV=/opt/gr00t-venv
ENV PATH="$VIRTUAL_ENV/bin:/usr/local/cuda/bin:$PATH"
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV CUDA_HOME=/usr/local/cuda-13.0
ENV CUDA_PATH=/usr/local/cuda-13.0
ENV CPATH="/usr/local/cuda-13.0/include:${CPATH:-}"
ENV C_INCLUDE_PATH="/usr/local/cuda-13.0/include:${C_INCLUDE_PATH:-}"
ENV CPLUS_INCLUDE_PATH="/usr/local/cuda-13.0/include:${CPLUS_INCLUDE_PATH:-}"
ENV LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib:\
$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13/lib:\
$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudss/lib:\
$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:\
${LD_LIBRARY_PATH:-}"

# ---------------------------------------------------------------------------
# MuJoCo rendering backend
#   EGL  = headless GPU rendering (default, no display needed)
#   glfw = interactive viewer (requires DISPLAY or Xvfb)
# Override at runtime: docker run -e MUJOCO_GL=glfw ...
# ---------------------------------------------------------------------------
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# ---------------------------------------------------------------------------
# Copy scripts from this repo into the image
# Files must live in the same directory as the Dockerfile (the build context)
# ---------------------------------------------------------------------------
COPY mujoco_viewer_demo.py /workspace/gr00t/mujoco_viewer_demo.py

# ---------------------------------------------------------------------------
# Apply Triton patch required for CUDA 13+ on Thor/Spark
# ---------------------------------------------------------------------------
RUN bash scripts/patch_triton_cuda13.sh || true

# ---------------------------------------------------------------------------
# Pre-download GR00T N1.7-3B model weights (optional — comment out to skip
# and instead pass --model-path nvidia/GR00T-N1.7-3B at runtime for
# on-demand HuggingFace download)
# ---------------------------------------------------------------------------
# RUN python -c "from huggingface_hub import snapshot_download; \
#     snapshot_download('nvidia/GR00T-N1.7-3B', local_dir='/checkpoints/GR00T-N1.7-3B')"

WORKDIR /workspace/gr00t
