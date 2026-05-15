"""
Interactive MuJoCo viewer demo for GR00T N1.7 on Jetson Thor.

Loads the Unitree G1 humanoid model (bundled with GR00T's submodules),
runs a random-policy rollout, and renders each frame in the MuJoCo viewer.

Requirements (already satisfied inside the gr00t-thor Docker image):
    mujoco>=3.0, numpy

Usage inside the container:
    # Headless rendering to PNG (no DISPLAY needed):
    python mujoco_viewer_demo.py --headless --frames 200 --out /tmp/frames

    # Interactive viewer (requires DISPLAY — use ./run.sh --display):
    python mujoco_viewer_demo.py --interactive
"""

import argparse
import os
import pathlib
import sys

import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# Locate a humanoid XML bundled with GR00T or fall back to the built-in
# MuJoCo humanoid
# ---------------------------------------------------------------------------
def find_model_xml() -> str:
    candidates = [
        # GR00T submodule: unitree G1 model
        "/workspace/gr00t/third_party/unitree_mujoco/unitree_robots/g1/scene.xml",
        "/workspace/gr00t/third_party/unitree_mujoco/unitree_robots/h1/scene.xml",
    ]
    for p in candidates:
        if pathlib.Path(p).exists():
            return p
    # Fall back to MuJoCo built-in humanoid
    return str(pathlib.Path(mujoco.__file__).parent / "model" / "humanoid" / "humanoid.xml")


# ---------------------------------------------------------------------------
# Headless: render frames to PNG files using EGL
# ---------------------------------------------------------------------------
def run_headless(model_xml: str, n_frames: int, out_dir: str) -> None:
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(model_xml)
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=480, width=640)

    print(f"Rendering {n_frames} frames to {out_dir} ...")
    for i in range(n_frames):
        # Random torques as placeholder policy
        data.ctrl[:] = np.random.uniform(-0.1, 0.1, model.nu)
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera="track")
        pixels = renderer.render()
        frame_path = out_path / f"frame_{i:04d}.png"
        import PIL.Image
        PIL.Image.fromarray(pixels).save(frame_path)
        if i % 50 == 0:
            print(f"  frame {i}/{n_frames}")

    print(f"Done. Frames saved to {out_dir}")


# ---------------------------------------------------------------------------
# Interactive: launch the MuJoCo passive viewer (requires DISPLAY)
# ---------------------------------------------------------------------------
def run_interactive(model_xml: str) -> None:
    if not os.environ.get("DISPLAY") and os.environ.get("MUJOCO_GL", "egl") != "glfw":
        print("ERROR: Interactive viewer requires a display.")
        print("  Run the container with:  ./run.sh --display")
        print("  Or set:  export DISPLAY=:0  and  MUJOCO_GL=glfw")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(model_xml)
    data = mujoco.MjData(model)

    print(f"Launching interactive MuJoCo viewer for: {model_xml}")
    print("Press Ctrl+C or close the window to exit.")

    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            data.ctrl[:] = np.random.uniform(-0.05, 0.05, model.nu)
            mujoco.mj_step(model, data)
            v.sync()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo viewer demo for GR00T")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--interactive", action="store_true",
                      help="Launch interactive MuJoCo viewer (needs DISPLAY)")
    mode.add_argument("--headless", action="store_true",
                      help="Render frames headlessly via EGL to PNG files")
    parser.add_argument("--model", default=None,
                        help="Path to a MuJoCo XML model file (auto-detected if omitted)")
    parser.add_argument("--frames", type=int, default=100,
                        help="Number of frames to render in headless mode (default: 100)")
    parser.add_argument("--out", default="/tmp/mujoco_frames",
                        help="Output directory for headless PNG frames")
    args = parser.parse_args()

    model_xml = args.model or find_model_xml()
    print(f"Using model: {model_xml}")

    if args.interactive:
        run_interactive(model_xml)
    else:
        # Default to headless
        run_headless(model_xml, args.frames, args.out)


if __name__ == "__main__":
    main()
