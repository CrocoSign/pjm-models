"""Uni-Sign pose preprocessing, extracted from the repo (datasets.py) as repo-free
numpy. Turns rtmlib 134-keypoint wholebody output into the 4 part-streams
(body/left/right/face_all) the pose projector consumes.

Faithful reimplementation of Uni-Sign's crop_scale + load_part_kp + frame
subsampling (so the ONNX runtime needs neither the repo nor torch).
"""
from __future__ import annotations

import random
import numpy as np

THR = 0.3
_FACE_IDX = [i for i in list(range(23, 23 + 17))[::2]] + list(range(83, 83 + 8)) + [53]
_BODY_IDX = [0] + list(range(3, 11))


def crop_scale(motion, thr=THR):
    """Normalize joints to [-1, 1] using the body bbox. motion: (T, V, 3)."""
    result = motion.copy()
    valid = motion[motion[..., 2] > thr][:, :2]
    if len(valid) < 4:
        return np.zeros(motion.shape), 0, None
    xmin, xmax = valid[:, 0].min(), valid[:, 0].max()
    ymin, ymax = valid[:, 1].min(), valid[:, 1].max()
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros(motion.shape), 0, None
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    result[result[..., 2] <= thr] = 0
    return result, scale, [xs, ys]


def load_part_kp(skeletons, confs):
    """skeletons: list of (1,134,2), confs: list of (1,134). -> dict of (T,V,3)."""
    out, scale = {}, None
    for part in ["body", "left", "right", "face_all"]:
        kps, confidences = [], []
        for skeleton, conf in zip(skeletons, confs):
            skeleton, conf = skeleton[0], conf[0]
            if part == "body":
                kp2d, c = skeleton[_BODY_IDX, :], conf[_BODY_IDX]
            elif part == "left":
                kp2d = skeleton[91:112, :]; kp2d = kp2d - kp2d[0, :]; c = conf[91:112]
            elif part == "right":
                kp2d = skeleton[112:133, :]; kp2d = kp2d - kp2d[0, :]; c = conf[112:133]
            else:  # face_all
                kp2d = skeleton[_FACE_IDX, :]; kp2d = kp2d - kp2d[-1, :]; c = conf[_FACE_IDX]
            kps.append(kp2d); confidences.append(c)
        kps = np.stack(kps, 0)
        confidences = np.stack(confidences, 0)
        feat = np.concatenate([kps, confidences[..., None]], axis=-1)
        if part == "body":
            result, scale, _ = crop_scale(feat, THR)
        else:
            result = feat
            if scale == 0:
                result = np.zeros(result.shape)
            else:
                result[..., :2] = result[..., :2] / scale
                result = np.clip(result, -1, 1)
                result[result[..., 2] <= THR] = 0
        out[part] = result.astype(np.float32)
    return out


def pose_to_parts(keypoints, scores, max_length=256, seed=42):
    """rtmlib output -> {body,left,right,face_all} each (1, T, V, 3) float32.

    keypoints: (T,1,134,2) already /[W,H]; scores: (T,1,134). Random frame
    subsampling (seeded) when T > max_length, matching the repo.
    """
    duration = len(scores)
    if duration > max_length:
        rng = random.Random(seed)
        idx = sorted(rng.sample(range(duration), max_length))
    else:
        idx = list(range(duration))
    skeletons = [keypoints[i] for i in idx]
    confs = [scores[i] for i in idx]
    parts = load_part_kp(skeletons, confs)
    return {k: v[None] for k, v in parts.items()}  # add batch dim
