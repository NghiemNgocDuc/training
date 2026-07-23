#!/usr/bin/env python3
import os
import sys
import subprocess
import torch

GPU_COUNT = torch.cuda.device_count()
SCRIPT = sys.argv[1] if len(sys.argv) > 1 else None
ARGS = sys.argv[2:]

if SCRIPT is None:
    print("Usage: python train.py <script.py> [args...]")
    print("  Automatically uses all available GPUs")
    sys.exit(1)

script_path = os.path.join("solvation-gnn", SCRIPT)
if not os.path.exists(script_path):
    script_path = SCRIPT

if GPU_COUNT <= 1:
    cmd = [sys.executable, script_path] + ARGS
    os.execvp(cmd[0], cmd)
else:
    print(f"Detected {GPU_COUNT} GPUs - launching with torchrun")
    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={GPU_COUNT}",
        script_path
    ] + ARGS
    subprocess.run(cmd, check=True)
