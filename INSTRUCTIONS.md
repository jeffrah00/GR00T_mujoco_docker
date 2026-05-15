# GR00T N1.7 + MuJoCo on Jetson Thor (JetPack 7.2) — Docker Setup

Step-by-step guide to build and run NVIDIA Isaac GR00T N1.7 with MuJoCo visualization inside Docker on a Jetson Thor board running JetPack 7.2.

---

## Overview

| Item | Value |
|------|-------|
| GR00T model | `nvidia/GR00T-N1.7-3B` (HuggingFace) |
| Base image | `nvidia/cuda:13.0.0-devel-ubuntu24.04` (aarch64) |
| CUDA | 13.0 (provided by JetPack 7.2) |
| Python | 3.12 |
| MuJoCo rendering | EGL (headless) or GLFW (interactive with display) |
| Target hardware | Jetson AGX Thor (H100 GPU module) |

---

## Prerequisites

### 1. Flash JetPack 7.2

Flash your Jetson Thor with JetPack 7.2 using SDK Manager on a host Ubuntu PC:

```
https://developer.nvidia.com/sdk-manager
```

Select: Jetson AGX Thor → JetPack 7.2 → Full install (includes CUDA 13, cuDNN, TensorRT).

Verify after boot:

```bash
# On the Jetson
jetson_release        # should show JetPack 7.2
nvcc --version        # should show CUDA 13.x
```

### 2. Install Docker + NVIDIA Container Runtime

JetPack 7.2 ships Docker and the NVIDIA container toolkit. Verify:

```bash
docker --version
sudo docker run --rm --runtime nvidia --gpus all \
    nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

If `nvidia-smi` runs successfully inside the container, you're ready.

If Docker is not installed:

```bash
sudo apt-get update
sudo apt-get install -y docker.io nvidia-container-toolkit
sudo systemctl restart docker
# Add your user to the docker group to avoid sudo
sudo usermod -aG docker $USER && newgrp docker
```

### 3. Configure NVIDIA container runtime as default (optional but convenient)

```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
```

### 4. HuggingFace account (for model weights)

Create a free account at `https://huggingface.co` if you don't have one.
The GR00T N1.7-3B model is public; no gating required.

To pre-authenticate inside the container:

```bash
# On the Jetson host
pip install huggingface_hub
huggingface-cli login    # paste your HF token
```

---

## Step 1 — Clone this repository

```bash
git clone https://github.com/jeffrah00/gr00t_mujoco_docker.git
cd gr00t_mujoco_docker
chmod +x build.sh run.sh
```

---

## Step 2 — Build the Docker image

```bash
./build.sh
```

What this does:
- Pulls `nvidia/cuda:13.0.0-devel-ubuntu24.04` (aarch64)
- Clones `NVIDIA/Isaac-GR00T` with all submodules
- Installs `uv`, NVPL LAPACK/BLAS, CUDA dev packages
- Runs the official `scripts/deployment/thor/install_deps.sh`:
  - Syncs Thor-specific `pyproject.toml` (PyTorch cu130, flash-attn, etc.)
  - Builds or installs `torchcodec` for video processing
- Installs MuJoCo display libraries (EGL + GLX + X11)
- Patches Triton for CUDA 13+ compatibility

**Expected build time:** 20–40 minutes (torchcodec may compile from source).

**Disk space required:** ~25 GB (image) + ~10 GB (model weights at runtime).

---

## Step 3 — Verify the image

```bash
docker run --rm --runtime nvidia --gpus all gr00t-thor:latest \
    python -c "
import torch, mujoco
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('MuJoCo:', mujoco.__version__)
"
```

Expected output:
```
PyTorch: 2.10.0+cu130
CUDA available: True
GPU: NVIDIA GH200 (or similar Jetson Thor GPU)
MuJoCo: 3.x.x
```

---

## Step 4 — Run GR00T demos

### Option A: Open-loop evaluation (headless, generates comparison plots)

This runs GR00T N1.7 inference on sample DROID data and saves visual output to `./output/`.

```bash
./run.sh -- python gr00t/eval/open_loop_eval.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --traj-ids 0 \
    --action-horizon 16
```

Plots are saved to `./output/traj_*.jpeg` on the host (mounted from `/tmp/open_loop_eval` inside the container).

The first run downloads the model weights (~6 GB) from HuggingFace automatically.
To persist weights across runs they are mounted to `./checkpoints/` on the host.

---

### Option B: Interactive MuJoCo viewer (requires physical display or SSH -X)

#### If the Jetson has a monitor connected (HDMI/DisplayPort)

```bash
./run.sh --display
```

Then inside the container:

```bash
# Headless EGL rendering — 100 frames, saves PNGs to /tmp/mujoco_frames
python /workspace/gr00t/mujoco_viewer_demo.py --headless --frames 200

# Interactive viewer — opens a window
python /workspace/gr00t/mujoco_viewer_demo.py --interactive
```

#### If working remotely over SSH

On your local machine (Linux with X11):

```bash
ssh -X user@<jetson-ip>
```

Then run with the display flag:

```bash
./run.sh --display
# Inside container:
python /workspace/gr00t/mujoco_viewer_demo.py --interactive
```

#### Headless with Xvfb (virtual framebuffer, no physical display needed)

```bash
./run.sh -- bash -c "
    Xvfb :99 -screen 0 1280x720x24 &
    DISPLAY=:99 MUJOCO_GL=glfw python mujoco_viewer_demo.py --interactive
"
```

---

### Option C: Server-client inference (real-time, production-style)

**Terminal 1 — Start the GR00T policy server:**

```bash
./run.sh -- python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --device cuda:0 \
    --port 5555
```

**Terminal 2 — Run an evaluation client against the server:**

```bash
# Get the container's IP
CONTAINER_IP=$(docker inspect gr00t-mujoco \
    --format '{{.NetworkSettings.IPAddress}}' 2>/dev/null || echo "127.0.0.1")

./run.sh -- python gr00t/eval/open_loop_eval.py \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --host 127.0.0.1 \
    --port 5555 \
    --traj-ids 1 2
```

---

### Option D: Standalone inference script

```bash
./run.sh -- python scripts/deployment/standalone_inference_script.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --traj-ids 1 2 \
    --inference-mode pytorch \
    --action-horizon 8
```

---

## Step 5 — Download model weights manually (optional)

If you want to pre-download weights before running (avoids download delay at inference time):

```bash
# On the host
mkdir -p checkpoints

docker run --rm --runtime nvidia \
    -v "${PWD}/checkpoints:/checkpoints" \
    gr00t-thor:latest \
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/GR00T-N1.7-3B', local_dir='/checkpoints/GR00T-N1.7-3B')
print('Done.')
"
```

Then reference the local path in demos:

```bash
./run.sh -- python gr00t/eval/open_loop_eval.py \
    --model-path /checkpoints/GR00T-N1.7-3B \
    ...
```

---

## Troubleshooting

### `CUDA error: no kernel image is available for execution`

The GR00T image needs to be compiled for the Thor GPU's compute capability.
Make sure you are using the `--profile=thor` Dockerfile (which this repo uses).

### `libGL error: No matching fbConfigs or visuals found`

MuJoCo can't find an EGL device. Check that the NVIDIA runtime is active:

```bash
docker run --rm --runtime nvidia gr00t-thor:latest \
    python -c "import OpenGL; print(OpenGL.__version__)"
```

If it fails, restart Docker and verify:

```bash
sudo systemctl restart docker
```

### Container exits immediately / `OOM killed`

GR00T N1.7-3B requires ~10 GB GPU memory. Ensure no other large processes are using the GPU:

```bash
nvidia-smi
```

### Flash-attn glibc error on older JetPack

```bash
# Inside the container
pip install flash-attn==2.7.4.post1 --no-binary flash-attn --no-cache
```

### MuJoCo viewer: `Cannot connect to X server`

You need `--display` flag and a running X server. See Option B above.

### `git-lfs: command not found` during build

```bash
sudo apt-get install -y git-lfs
git lfs install
```

---

## Directory structure

```
gr00t_mujoco_docker/
├── Dockerfile             # Jetson Thor (CUDA 13, Ubuntu 24.04, Python 3.12)
├── build.sh               # Build the Docker image
├── run.sh                 # Run with GPU + optional X11 display
├── mujoco_viewer_demo.py  # Standalone MuJoCo viewer / headless renderer
├── checkpoints/           # Model weights (created at runtime, gitignored)
├── output/                # Evaluation plots (created at runtime, gitignored)
└── INSTRUCTIONS.md        # This file
```

---

## Key environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MUJOCO_GL` | `egl` | MuJoCo renderer: `egl` (headless) or `glfw` (display) |
| `DISPLAY` | unset | X11 display (set when using `--display`) |
| `IMAGE_NAME` | `gr00t-thor` | Docker image name override |
| `IMAGE_TAG` | `latest` | Docker image tag override |
| `CONTAINER_NAME` | `gr00t-mujoco` | Container name override |

---

## References

- Isaac-GR00T GitHub: `https://github.com/NVIDIA/Isaac-GR00T`
- GR00T-WholeBodyControl: `https://github.com/NVlabs/GR00T-WholeBodyControl`
- Model on HuggingFace: `https://huggingface.co/nvidia/GR00T-N1.7-3B`
- JetPack SDK Manager: `https://developer.nvidia.com/sdk-manager`
- MuJoCo documentation: `https://mujoco.readthedocs.io`
