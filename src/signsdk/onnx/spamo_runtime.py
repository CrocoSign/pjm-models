"""Repo-free SpaMo ONNX runtime: raw mp4 -> text, WITHOUT the SpaMo-PJM/crop-params
repos. YOLO11n crop + CLIP/VideoMAE/(hand-CLIP) + projector + T5, all onnxruntime;
HF processors/tokenizer by id/asset; MediaPipe for quad. No torch model.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .glue import video as _video
from .glue import decode as _decode
from .glue.yolo_crop import YoloCropper

ASSETS = os.environ.get("SIGNSDK_ASSETS", "/home/croco/signsdk/dev/assets")
SHARED = os.environ.get("SIGNSDK_SPAMO_SHARED", "/home/croco/signsdk/dev/onnx/spamo")


def _flan_tokenizer(assets):
    from transformers import T5Tokenizer
    local = os.path.join(assets, "flan_t5_tokenizer")
    src = local if os.path.exists(os.path.join(local, "spiece.model")) else "google/flan-t5-xl"
    return T5Tokenizer.from_pretrained(src, legacy=False)


class SpaMoRuntime:
    def __init__(self, model_dir, shared=None, assets=None, device="cuda"):
        import onnxruntime as ort
        # explicit slow (PIL/numpy) classes -- AutoImageProcessor pulls a torch backend in transformers 5.x
        from transformers import CLIPImageProcessor, VideoMAEImageProcessor
        shared = shared or SHARED
        assets = assets or ASSETS

        self.meta = json.load(open(f"{model_dir}/meta.json"))
        self.quad = self.meta["kind"] == "quad"
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            prov = [("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]
        else:
            prov = ["CPUExecutionProvider"]
        self.clip = ort.InferenceSession(f"{shared}/clip_vit_s2.onnx", providers=prov)
        self.mae = ort.InferenceSession(f"{shared}/videomae.onnx", providers=prov)
        self.proj = ort.InferenceSession(f"{model_dir}/projector.onnx", providers=prov)
        self.enc = ort.InferenceSession(f"{model_dir}/encoder.onnx", providers=prov)
        self.dec = ort.InferenceSession(f"{model_dir}/decoder.onnx", providers=prov)
        self.prompt = np.load(f"{model_dir}/prompt_embeds.npy").astype(np.float32)
        self.clip_proc = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14", do_center_crop=False, do_resize=False)
        self.mae_proc = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-large")
        self.tok = _flan_tokenizer(assets)
        self.yolo = YoloCropper(os.path.join(assets, "yolo", "yolo11n.onnx"), device=device)
        if self.quad:
            from .glue import mediapipe_kp as mp_glue
            self._mp_glue = mp_glue
            self.hand_clip = ort.InferenceSession(f"{shared}/hand_clip.onnx", providers=prov)
            self.hand_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.pose_det, self.hand_det = mp_glue.create_detectors(
                os.path.join(assets, "mediapipe", "pose_landmarker_full.task"),
                os.path.join(assets, "mediapipe", "hand_landmarker.task"))

    def _spatial(self, processed):
        px = self.clip_proc(list(processed), return_tensors="np").pixel_values.astype(np.float32)
        T, B, out = px.shape[0], 8, []
        for i in range(0, T, B):
            ch = px[i:i + B]; real = len(ch)
            if real < B:
                ch = np.concatenate([ch, np.repeat(ch[-1:], B - real, 0)], 0)
            out.append(self.clip.run(None, {"pixel_values": ch})[0][:real])
        return np.concatenate(out, 0)

    def _motion(self, processed):
        pf = list(processed)
        if len(pf) < 16:
            pf += [pf[-1]] * (16 - len(pf))
        chunks = _video.sliding_window(pf, 16, 8)
        px = self.mae_proc(images=chunks, return_tensors="np").pixel_values.astype(np.float32)
        # VideoMAE attention softmax buffer is ~O(batch); 32 needs ~5GB and OOMs the GPU on
        # long clips next to the 11.5GB SpaMo weights. Default 8 (~1.3GB); tune up on a big GPU.
        bs = int(os.environ.get("SIGNSDK_MAE_BATCH", "8"))
        out = [self.mae.run(None, {"pixel_values": px[i:i + bs]})[0] for i in range(0, len(px), bs)]
        return np.concatenate(out, 0)

    def _quad_streams(self, processed):
        import mediapipe as mp
        from PIL import Image
        kp_feats, crops, sizes = [], [], []
        for im in processed:
            rgb = np.array(im.convert("RGB"))
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            pr, hr = self.pose_det.detect(mp_img), self.hand_det.detect(mp_img)
            kp_feats.append(self._mp_glue.extract_landmarks(pr, hr))
            h, w = rgb.shape[:2]; n = 0
            if hr.hand_landmarks:
                for hl in hr.hand_landmarks:
                    x1, y1, x2, y2 = self._mp_glue.hand_bbox(hl, w, h)
                    if x2 > x1 and y2 > y1:
                        crops.append(rgb[y1:y2, x1:x2].copy()); n += 1
            sizes.append(n)
        kp = np.stack(kp_feats, 0).astype(np.float32)
        if crops:
            px = self.hand_proc(images=[Image.fromarray(c) for c in crops],
                                return_tensors="np").pixel_values.astype(np.float32)
            feats = [self.hand_clip.run(None, {"pixel_values": px[i:i + 32]})[0] for i in range(0, len(px), 32)]
            allf = np.concatenate(feats, 0)
        else:
            allf = np.empty((0, 2048), np.float32)
        zero, aux, idx = np.zeros(2048, np.float32), [], 0
        for n in sizes:
            aux.append(allf[idx:idx + n].mean(0) if n else zero); idx += n
        return kp, np.stack(aux, 0).astype(np.float32)

    def translate(self, video_path, num_beams=5, max_new_tokens=64):
        with open(video_path, "rb") as f:
            frames = _video.get_video_frames(f.read())
        # Native SpaMo caps the sequence at max_frame_length=512 (dataset/T5.py). Longer
        # clips both OOM the T5 encoder and are out-of-distribution; cap deterministically.
        max_frames = int(os.environ.get("SIGNSDK_MAX_FRAMES", "512"))
        if len(frames) > max_frames:
            frames = frames[:max_frames]
        crop = self.yolo.crop_params([np.asarray(f)[:, :, ::-1] for f in frames])
        processed = _video.process_frames(frames, crop)
        feeds = {"pixel": self._spatial(processed)[None], "glor": self._motion(processed)[None]}
        if self.quad:
            kp, aux = self._quad_streams(processed)
            feeds["aux"] = aux[None]; feeds["kp"] = kp[None]
        visual = self.proj.run(None, feeds)[0]
        ie = np.concatenate([visual, self.prompt], axis=1).astype(np.float32)
        mask = np.ones((1, ie.shape[1]), dtype=bool)
        eh = self.enc.run(None, {"inputs_embeds": ie, "attention_mask": mask})[0]
        ids = _decode.beam_search(self.dec, eh, mask, self.meta["decoder_start_token_id"],
                                  self.meta["eos_token_id"], num_beams, max_new_tokens)
        return self.tok.decode(ids, skip_special_tokens=True).lower()

    def close(self):
        """Drop the ONNX sessions so their GPU memory is freed (each session owns a CUDA
        arena; deleting it returns the memory)."""
        for a in ("clip", "mae", "proj", "enc", "dec", "hand_clip", "yolo", "hand_proc",
                  "pose_det", "hand_det"):
            if hasattr(self, a):
                setattr(self, a, None)
