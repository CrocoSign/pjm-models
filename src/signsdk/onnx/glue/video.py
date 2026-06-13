"""Repo-free video glue for SpaMo: decode (PyAV) + signer crop/resize + chunking.
Extracted from SpaMo-PJM/utils/pjm/preprocessing.py + utils/helpers.py."""
from __future__ import annotations


def get_video_frames(video_bytes):
    """mp4 bytes -> list of PIL RGB Images (PyAV)."""
    from io import BytesIO
    from av import open as av_open
    container = av_open(BytesIO(video_bytes))
    stream = container.streams.video[0]
    return [frame.to_image() for frame in container.decode(stream)]


def process_frames(frames, crop_params, output_size=(224, 224)):
    """Crop each PIL frame to the signer box, then PIL.ImageOps.pad to output_size."""
    from PIL.ImageOps import pad as resize_and_pad
    box = (crop_params["x_start"], crop_params["y_start"],
           crop_params["x_end"], crop_params["y_end"])
    return [resize_and_pad(f.crop(box), output_size) for f in frames]


def sliding_window(data, window=16, overlap=8):
    step = window - overlap
    return [data[i:i + window] for i in range(0, len(data), step) if i + window <= len(data)]
