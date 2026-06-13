"""Repo-free MediaPipe glue for Mod-SpaMo (quad): pose+hand landmarks (258-d) and
hand-crop bboxes. Extracted from SpaMo-PJM scripts/. Uses mediapipe (pip) + the
.task models bundled in SIGNSDK_ASSETS/mediapipe/."""
from __future__ import annotations

import numpy as np

POSE_DIM = 33 * 4
HAND_DIM = 21 * 3
FEAT_DIM = POSE_DIM + HAND_DIM * 2   # 258
HAND_CROP_PAD = 0.25


def create_detectors(pose_task_path, hand_task_path):
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python.vision import (
        HandLandmarker, HandLandmarkerOptions, PoseLandmarker, PoseLandmarkerOptions, RunningMode)
    pose_det = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=pose_task_path),
        running_mode=RunningMode.IMAGE))
    hand_det = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=hand_task_path),
        num_hands=2, running_mode=RunningMode.IMAGE))
    return pose_det, hand_det


def extract_landmarks(pose_result, hand_result):
    vec = np.zeros(FEAT_DIM, dtype=np.float32)
    if pose_result.pose_landmarks:
        for i, lm in enumerate(pose_result.pose_landmarks[0]):
            vec[i * 4:i * 4 + 4] = [lm.x, lm.y, lm.z, lm.visibility]
    if hand_result.hand_landmarks:
        for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            offset = POSE_DIM if handedness[0].category_name == "Left" else POSE_DIM + HAND_DIM
            for i, lm in enumerate(landmarks):
                vec[offset + i * 3:offset + i * 3 + 3] = [lm.x, lm.y, lm.z]
    return vec


def hand_bbox(landmarks, img_w, img_h, pad=HAND_CROP_PAD):
    xs = [lm.x * img_w for lm in landmarks]
    ys = [lm.y * img_h for lm in landmarks]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    return (int(max(0, x1 - pw)), int(max(0, y1 - ph)),
            int(min(img_w, x2 + pw)), int(min(img_h, y2 + ph)))
