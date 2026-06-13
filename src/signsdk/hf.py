"""Resolve model artifacts from HF croco-corp/pjm-models, downloading on demand."""
import os
from huggingface_hub import snapshot_download
from .base import SignTranslatorError

REPO = os.environ.get("SIGNSDK_HF_REPO", "croco-corp/pjm-models")

def _home():
    return os.environ.get("SIGNSDK_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "signsdk")

def _snap(patterns):
    return snapshot_download(REPO, allow_patterns=patterns,
                             local_dir=os.path.join(_home(), "pjm-models"))

def resolve_unisign(name):
    root = _snap([f"unisign/{name}/*", "assets/mt5_tokenizer/*"])
    md = os.path.join(root, "unisign", name)
    if not os.path.exists(os.path.join(md, "decoder.onnx")):
        raise SignTranslatorError(f"{name!r} is not yet published on {REPO}")
    return {"model_dir": md, "assets": os.path.join(root, "assets")}

def resolve_spamo(name, kind):
    pats = [f"spamo/{name}/*", "spamo/shared/clip_vit_s2.onnx*",
            "spamo/shared/videomae.onnx*", "assets/flan_t5_tokenizer/*", "assets/yolo/*"]
    if kind == "quad":
        pats += ["spamo/shared/hand_clip.onnx*", "assets/mediapipe/*"]
    root = _snap(pats)
    md = os.path.join(root, "spamo", name)
    if not os.path.exists(os.path.join(md, "decoder.onnx")):
        raise SignTranslatorError(f"{name!r} is not yet published on {REPO}")
    return {"model_dir": md, "shared": os.path.join(root, "spamo", "shared"),
            "assets": os.path.join(root, "assets")}
