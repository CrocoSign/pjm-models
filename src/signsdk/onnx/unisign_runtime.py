"""Repo-free Uni-Sign ONNX runtime: raw mp4 -> text, WITHOUT the Uni-Sign repo.

Depends only on pip packages (onnxruntime, rtmlib, transformers tokenizer, cv2,
numpy) + pre-exported ONNX graphs + a bundled mt5 tokenizer asset. No torch model,
no sys.path into /home/croco/Uni-Sign, no chdir. (Graph export is a separate
build step that does need the repo.)
"""
from __future__ import annotations

import json
import os

import numpy as np

from .glue import pose as _pose
from .glue import decode as _decode

ASSETS = os.environ.get("SIGNSDK_ASSETS", "/home/croco/signsdk/dev/assets")


class UniSignRuntime:
    def __init__(self, cache_dir, device="cuda", assets=None, max_length=256):
        import onnxruntime as ort
        from rtmlib import Wholebody
        from transformers import T5Tokenizer
        assets = assets or ASSETS

        self.meta = json.load(open(f"{cache_dir}/meta.json"))
        self.max_length = int(max_length)
        rt = "cuda" if device == "cuda" else "cpu"
        prov = (["CUDAExecutionProvider"] if device == "cuda" and
                "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"])
        self.wb = Wholebody(to_openpose=True, mode="lightweight", backend="onnxruntime", device=rt)
        self.proj = ort.InferenceSession(f"{cache_dir}/projector.onnx", providers=prov)
        self.enc = ort.InferenceSession(f"{cache_dir}/encoder.onnx", providers=prov)
        self.dec = ort.InferenceSession(f"{cache_dir}/decoder.onnx", providers=prov)
        self.prefix = np.load(f"{cache_dir}/prefix.npy").astype(np.float32)
        self.tok = T5Tokenizer.from_pretrained(os.path.join(assets, "mt5_tokenizer"), legacy=False)

    def _extract_pose(self, video_path):
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(np.uint8(fr))
        cap.release()
        kp, sc = [], []
        for fr in frames:
            k, s = self.wb(fr)
            h, w = fr.shape[:2]
            kp.append(k / np.array([w, h])[None, None])
            sc.append(s)
        return np.asarray(kp), np.asarray(sc)

    def translate(self, video_path, num_beams=4, max_new_tokens=100, seed=42):
        kp, sc = self._extract_pose(video_path)
        parts = _pose.pose_to_parts(kp, sc, self.max_length, seed)
        feeds = {k: parts[k].astype(np.float32) for k in ["body", "left", "right", "face_all"]}
        pe = self.proj.run(None, feeds)[0]
        ie = np.concatenate([self.prefix, pe], axis=1).astype(np.float32)
        am = np.ones((1, ie.shape[1]), dtype=np.int64)
        eh = self.enc.run(None, {"inputs_embeds": ie, "attention_mask": am})[0]
        ids = _decode.beam_search(self.dec, eh, am, self.meta["decoder_start_token_id"],
                                  self.meta["eos_token_id"], num_beams, max_new_tokens)
        return self.tok.decode(ids, skip_special_tokens=True)

    def close(self):
        """Drop the ONNX sessions (rtmlib pose + projector/encoder/decoder) to free GPU."""
        for a in ("wb", "proj", "enc", "dec"):
            if hasattr(self, a):
                setattr(self, a, None)
