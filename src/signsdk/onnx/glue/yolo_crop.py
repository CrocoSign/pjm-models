"""Repo-free signer crop via YOLO11n ONNX + numpy (replaces the crop-params repo).

Per frame: YOLO11n person detection (onnxruntime) -> boxes; then the crop-params
logic (1st/99th-percentile x across all frames -> square crop centered on the
signer, full height, clamped + padded). No ultralytics/torch, no crop-params repo,
no LMDB.
"""
from __future__ import annotations

from math import floor, ceil
import numpy as np


def _letterbox(img, new=640, color=114):
    import cv2
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (new - nw) / 2, (new - nh) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(color, color, color))
    return out, r, left, top


def _nms(boxes, scores, iou_thr):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1); h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


class YoloCropper:
    PERSON = 0

    def __init__(self, onnx_path, device="cuda", conf=0.25, iou=0.7, imgsz=640):
        import onnxruntime as ort
        prov = (["CUDAExecutionProvider"] if device == "cuda" and
                "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"])
        self.sess = ort.InferenceSession(onnx_path, providers=prov)
        self.conf, self.iou, self.imgsz = conf, iou, imgsz

    def _detect(self, frame_bgr):
        img, r, padx, pady = _letterbox(frame_bgr, self.imgsz)
        x = (img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)) / 255.0
        out = self.sess.run(None, {"images": x})[0][0].T   # (8400, 84)
        person = out[:, 4 + self.PERSON]
        m = person > self.conf
        if not m.any():
            return np.zeros((0, 4), np.float32)
        b, s = out[m, :4], person[m]
        xyxy = np.empty_like(b)
        xyxy[:, 0] = b[:, 0] - b[:, 2] / 2; xyxy[:, 1] = b[:, 1] - b[:, 3] / 2
        xyxy[:, 2] = b[:, 0] + b[:, 2] / 2; xyxy[:, 3] = b[:, 1] + b[:, 3] / 2
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - padx) / r
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pady) / r
        keep = _nms(xyxy, s, self.iou)
        return xyxy[keep]

    def crop_params(self, frames):
        """frames: list of BGR np arrays -> crop_params dict (matches crop-params repo)."""
        H, W = frames[0].shape[:2]
        boxes = [self._detect(f) for f in frames]
        xyxy = np.concatenate([b for b in boxes if len(b)], axis=0)
        if len(xyxy) == 0:
            raise RuntimeError("no person detected by YOLO")
        box_start = floor(np.percentile(xyxy[:, 0], 1))
        box_end = ceil(np.percentile(xyxy[:, 2], 99))
        box_width = box_end - box_start
        x_middle = (box_end + box_start) / 2
        size = max(box_width, H)

        def _pad_split(p):
            h = int(p / 2)
            return (h, h) if p % 2 == 0 else (h, h + 1)

        hpad = vpad = 0
        x_start = floor(x_middle - size / 2); x_end = x_start + size
        if x_start < 0: hpad += -x_start; x_start = 0
        if x_end > W: hpad += x_end - W; x_end = W
        y_start = floor(H / 2 - size / 2); y_end = y_start + size
        if y_start < 0: vpad += -y_start; y_start = 0
        if y_end > H: vpad += y_end - H; y_end = H
        padding = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        if hpad or vpad:
            le, ri = _pad_split(hpad); to, bo = _pad_split(vpad)
            padding = {"top": to, "bottom": bo, "left": le, "right": ri}
        return {"x_start": x_start, "y_start": y_start, "x_end": x_end, "y_end": y_end,
                "padding": padding}
