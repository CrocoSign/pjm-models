"""Export SpaMo's motion extractor (VideoMAE-large) to ONNX and validate parity.

Input  : (B, 16, 3, 224, 224) preprocessed 16-frame clips (HF VideoMAE processor
         + chunking stay in the light host glue).
Output : (B, 1024) motion features (CLS, final layer).
Run with the SpaMo-PJM environment, from the root of the SpaMo-PJM repository.
"""
import os, sys
import numpy as np, torch, torch.nn as nn

_REPO = os.getcwd(); sys.path.insert(0, _REPO)
from scripts.mae_features_for_pjm import VideoMAEFeatureReader
from utils.pjm.preprocessing import get_video_frames, ImageConverter
from utils.helpers import sliding_window_for_list

OUT = os.path.join(os.environ["SIGNSDK_SPAMO_SHARED"], "videomae.onnx")
SAMPLE = os.environ["SIGNSDK_SAMPLE"]
KEY = "004_000_M"
CACHE = os.environ.get("SIGNSDK_HF_CACHE") or None


class MAEWrap(nn.Module):
    def __init__(self, reader):
        super().__init__()
        self.model = reader.model
        self.nth = reader.nth_layer

    def forward(self, pixel_values):
        h = self.model(pixel_values=pixel_values, output_hidden_states=True).hidden_states[self.nth]
        return h[:, 0]


def main():
    reader = VideoMAEFeatureReader(model_name="MCG-NJU/videomae-large",
                                   cache_dir=CACHE, device="cuda:0", overlap_size=8)
    wrap = MAEWrap(reader).eval().cuda()

    import lmdb
    env = lmdb.open("crop_params/crop_params.lmdb", readonly=True, lock=False, subdir=True)
    conv = ImageConverter(env, (224, 224))
    frames = get_video_frames(open(SAMPLE, "rb").read())
    pf = list(conv.process_frames(frames, KEY))
    if len(pf) < 16:
        pf += [pf[-1]] * (16 - len(pf))
    chunks = sliding_window_for_list(pf, 16, overlap_size=8)[:4]
    pixel = reader.image_processor(images=chunks, return_tensors="pt").pixel_values.cuda()
    print("pixel:", tuple(pixel.shape))

    with torch.no_grad():
        ref_wrap = wrap(pixel)
        ref_reader = reader.get_feats(chunks)
    print("wrap vs reader.get_feats:", float((ref_wrap - ref_reader).abs().max()))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    try:
        torch.onnx.export(wrap, (pixel,), OUT,
                          input_names=["pixel_values"], output_names=["features"],
                          dynamic_axes={"pixel_values": {0: "B"}, "features": {0: "B"}},
                          opset_version=17, do_constant_folding=True, dynamo=False)
        print("exported (legacy) ->", OUT)
    except Exception as e:
        print("legacy failed, dynamo:", type(e).__name__)
        from torch.export import Dim
        torch.onnx.export(wrap, (pixel,), OUT, dynamo=True,
                          input_names=["pixel_values"], output_names=["features"],
                          dynamic_shapes={"pixel_values": {0: Dim("B", min=1, max=4096)}})
        print("exported (dynamo) ->", OUT)

    import onnxruntime as ort
    prov = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(OUT, providers=prov)
    onnx_out = sess.run(None, {"pixel_values": pixel.cpu().numpy().astype(np.float32)})[0]
    d = np.abs(onnx_out - ref_wrap.cpu().numpy())
    print(f"ONNX parity: max={d.max():.3e} mean={d.mean():.3e}")
    print("PASS" if d.max() < 5e-2 else "CHECK")


if __name__ == "__main__":
    main()
