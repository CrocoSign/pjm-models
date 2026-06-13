"""GPU memory probe + device selection — torch-free, via nvidia-smi.

Lets a backend decide, before loading, whether a model fits in free GPU memory;
if not, callers free resident models (see ``signsdk.load``) and/or fall back to CPU
rather than OOMing.
"""
from __future__ import annotations

import shutil
import subprocess


def free_mib(device_index: int = 0):
    """Free GPU memory in MiB for ``device_index``, or ``None`` if it can't be probed."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.check_output(
            [exe, "--query-gpu=memory.free", "--format=csv,noheader,nounits",
             "-i", str(device_index)], timeout=5).decode().strip().splitlines()
        return int(out[0])
    except Exception:
        return None


def pick_device(requested: str, need_mib: int, device_index: int = 0) -> str:
    """Return ``"cuda"`` if ``requested == "cuda"`` and at least ``need_mib`` is free,
    else ``"cpu"``. If the probe fails (no nvidia-smi), trust the request — the ONNX
    runtime still falls back to CPU when no CUDA provider is available."""
    if requested != "cuda":
        return requested
    free = free_mib(device_index)
    if free is None:
        return "cuda"
    return "cuda" if free >= need_mib else "cpu"
