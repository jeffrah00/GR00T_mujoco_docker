# GR00T N1.7 + MuJoCo on Jetson Thor (JetPack 7.2)

Run NVIDIA Isaac GR00T N1.7 with MuJoCo visualization in Docker on a Jetson AGX Thor board running JetPack 7.2.

---

## Hardware & Software Requirements

| Item | Requirement |
|------|-------------|
| Board | Jetson AGX Thor |
| JetPack | 7.2 (Ubuntu 24.04, CUDA 13.x) |
| GPU Memory | 10 GB+ (H100 module on Thor) |
| Disk | ~35 GB free (25 GB image + 10 GB weights) |
| GR00T model | `nvidia/GR00T-N1.7-3B` |
| Base image | `nvidia/cuda:13.0.0-devel-ubuntu24.04` (aarch64) |
| Python | 3.12 |
| MuJoCo rendering | EGL (headless) or GLFW (interactive with display) |

---

## Step 1 — Flash JetPack 7.2

On a host Ubuntu PC, download and run NVIDIA SDK Manager:

```
https://developer.nvidia.com/sdk-manager
```

Select: **Jetson AGX Thor → JetPack 7.2 → Full install**
(includes CUDA 13, cuDNN, TensorRT, Docker, NVIDIA container toolkit).

After the Jetson boots, verify the installation:

```bash
jetson_release        # JetPack 7.2
nvcc --version        # CUDA 13.x
docker --version
```

---

## Step 2 — Verify Docker + NVIDIA runtime on the Jetson

```bash
sudo docker run --rm --runtime nvidia --gpus all \
    nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

`nvidia-smi` should print the GPU info. If it does, skip to Step 3.

If Docker or the NVIDIA runtime is missing:

```bash
sudo apt-get update
sudo apt-get install -y docker.io nvidia-container-toolkit
sudo systemctl restart docker
sudo usermod -aG docker $USER && newgrp docker
```

Optionally set the NVIDIA runtime as the Docker default:

```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
```

---

## Step 3 — Clone this repository

```bash
git clone https://github.com/jeffrah00/gr00t_mujoco_docker.git
cd gr00t_mujoco_docker
chmod +x build.sh run.sh
```

Repository layout:

```
gr00t_mujoco_docker/
├── Dockerfile              # Jetson Thor image (CUDA 13, Ubuntu 24.04, Python 3.12)
├── build.sh                # Builds the Docker image
├── run.sh                  # Runs the container (headless or with display)
├── mujoco_viewer_demo.py   # Standalone MuJoCo headless renderer / interactive viewer
├── checkpoints/            # Model weights land here (gitignored, host-mounted)
└── output/                 # Evaluation plots land here (gitignored, host-mounted)
```

---

## Step 4 — Build the Docker image

```bash
./build.sh
```

**Expected build time:** 20–40 minutes on first build.

What happens inside:

1. Pulls `nvidia/cuda:13.0.0-devel-ubuntu24.04` (aarch64)
2. Clones `NVIDIA/Isaac-GR00T` with all submodules
3. Installs `uv`, NVPL LAPACK/BLAS, CUDA 13 dev packages
4. Runs the official `scripts/deployment/thor/install_deps.sh`:
   - Syncs the Thor-specific `pyproject.toml` (PyTorch cu130, flash-attn, transformers, etc.)
   - Installs or builds `torchcodec` for video processing
5. Installs MuJoCo display libraries (EGL + GLX + X11)
6. Applies the Triton CUDA 13+ compatibility patch

---

## Step 5 — Verify the image

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
GPU: NVIDIA GH200 120GB
MuJoCo: 3.x.x
```

---

## Step 6 — (Optional) Pre-download model weights

The first inference run auto-downloads `nvidia/GR00T-N1.7-3B` (~6 GB) from HuggingFace.
To download in advance and avoid the wait:

```bash
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

Pass the local path to any demo with `--model-path /checkpoints/GR00T-N1.7-3B`.

---

## Step 7 — Run the demos

### Demo A — Open-loop evaluation (headless, generates comparison plots)

Runs GR00T N1.7 inference on sample DROID robot data. No display needed.

```bash
./run.sh -- python gr00t/eval/open_loop_eval.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --traj-ids 0 \
    --action-horizon 16
```

Output plots (`traj_*.jpeg`) are written to `./output/` on the host.

---

### Demo B — Interactive MuJoCo viewer

Requires a display. Pick the method that matches your setup:

**Jetson has a monitor connected (HDMI/DisplayPort):**

```bash
./run.sh --display
# Inside the container:
python mujoco_viewer_demo.py --interactive
```

**Remote access over SSH with X11 forwarding:**

```bash
# On your local machine (Linux with X11)
ssh -X user@<jetson-ip>

# Back on the Jetson
./run.sh --display
# Inside the container:
python mujoco_viewer_demo.py --interactive
```

**Headless virtual framebuffer (no physical display, saves PNGs):**

```bash
./run.sh -- python mujoco_viewer_demo.py --headless --frames 200 --out /tmp/mujoco_frames
```

---

### Demo C — Standalone inference script

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

### Demo D — Server-client inference (real-time / production)

Open two terminals on the Jetson.

**Terminal 1 — start the GR00T policy server:**

```bash
./run.sh -- python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --device cuda:0 \
    --port 5555
```

**Terminal 2 — run the evaluation client:**

```bash
./run.sh -- python gr00t/eval/open_loop_eval.py \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --host 127.0.0.1 \
    --port 5555 \
    --traj-ids 1 2
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MUJOCO_GL` | `egl` | `egl` = headless GPU, `glfw` = window with display |
| `DISPLAY` | unset | X11 display socket (set automatically by `--display`) |
| `IMAGE_NAME` | `gr00t-thor` | Override the Docker image name |
| `IMAGE_TAG` | `latest` | Override the Docker image tag |
| `CONTAINER_NAME` | `gr00t-mujoco` | Override the container name |

---

## Troubleshooting

**`CUDA error: no kernel image is available for execution`**
Confirm you are using this repo's Dockerfile (targets Jetson Thor / CUDA 13).

**`libGL error: No matching fbConfigs or visuals found`**
The NVIDIA runtime is not active. Run:
```bash
sudo systemctl restart docker
docker run --rm --runtime nvidia gr00t-thor:latest nvidia-smi
```

**Container OOM-killed**
GR00T N1.7-3B needs ~10 GB GPU memory. Free up GPU memory with `nvidia-smi`, then retry.

**`Cannot connect to X server`**
The interactive viewer needs a display. Use `./run.sh --display` and ensure `DISPLAY` is set on the host (e.g., `:0` for a local monitor or forwarded by `ssh -X`).

**flash-attn glibc error**
```bash
# Inside the container
pip install flash-attn==2.7.4.post1 --no-binary flash-attn --no-cache
```

**`git-lfs: command not found` during build**
```bash
sudo apt-get install -y git-lfs && git lfs install
```

---

## References

- Isaac-GR00T GitHub: https://github.com/NVIDIA/Isaac-GR00T
- GR00T-WholeBodyControl: https://github.com/NVlabs/GR00T-WholeBodyControl
- Model weights: https://huggingface.co/nvidia/GR00T-N1.7-3B
- JetPack SDK Manager: https://developer.nvidia.com/sdk-manager
- MuJoCo docs: https://mujoco.readthedocs.io
