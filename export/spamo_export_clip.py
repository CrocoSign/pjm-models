"""Export SpaMo's spatial extractor (CLIP ViT-L/14 + s2-wrapping) to ONNX and
validate parity vs the native ViTFeatureReader.

Input  : preprocessed pixel batch (N, 3, 224, 224)  -- HF image_processor stays
         in the (light, CPU) host glue.
Output : (N, 2048) spatial features (s2 scales [1,2], CLS token).
Run with the SpaMo-PJM environment, from the root of the SpaMo-PJM repository.
"""
import os, sys
import numpy as np, torch, torch.nn as nn

_REPO = os.getcwd(); sys.path.insert(0, _REPO)
from scripts.pjm_vit_extract import ViTFeatureReader
from utils.s2wrapper import forward as multiscale_forward
from utils.pjm.preprocessing import get_video_frames, ImageConverter

OUT = os.path.join(os.environ["SIGNSDK_SPAMO_SHARED"], "clip_vit_s2.onnx")
SAMPLE = os.environ["SIGNSDK_SAMPLE"]
KEY = "004_000_M"
CACHE = os.environ.get("SIGNSDK_HF_CACHE") or None


class CLIPs2(nn.Module):
    """pixel_values (N,3,224,224) -> (N, 2048) via s2 multiscale + CLS."""
    def __init__(self, reader):
        super().__init__()
        self.model = reader.model
        self.scales = reader.scales
        self.nth = reader.nth_layer

    def forward(self, pixel_values):
        def ff(x):
            return self.model(x).hidden_states[self.nth]
        out = multiscale_forward(ff, pixel_values, scales=self.scales, num_prefix_token=1)
        return out[:, 0]


def main():
    reader = ViTFeatureReader(model_name="openai/clip-vit-large-patch14",
                              cache_dir=CACHE, device="cuda:0",
                              s2_mode="s2wrapping", scales=[1, 2])
    wrap = CLIPs2(reader).eval().cuda()

    # real pixel input: decode -> crop -> image_processor (host glue)
    import lmdb
    env = lmdb.open("crop_params/crop_params.lmdb", readonly=True, lock=False, subdir=True)
    conv = ImageConverter(env, (224, 224))
    frames = get_video_frames(open(SAMPLE, "rb").read())
    processed = conv.process_frames(frames, KEY)[:8]   # 8 frames is enough to validate
    pixel = reader.image_processor(list(processed), return_tensors="pt").pixel_values.cuda()
    print("pixel:", tuple(pixel.shape))

    with torch.no_grad():
        ref_wrap = wrap(pixel)                          # graph reference
        ref_reader = reader.get_feats(processed)        # full native path
    print("wrap vs reader.get_feats:", float((ref_wrap - ref_reader).abs().max()))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # legacy exporter chokes on s2's adaptive pooling (non-constant output_size);
    # the dynamo exporter handles it. Dynamic batch via torch.export.Dim.
    from torch.export import Dim
    torch.onnx.export(wrap, (pixel,), OUT, dynamo=True,
                      input_names=["pixel_values"], output_names=["features"],
                      dynamic_shapes={"pixel_values": {0: Dim("N", min=1, max=4096)}})
    print("exported ->", OUT)

    import onnxruntime as ort
    prov = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(OUT, providers=prov)
    onnx_out = sess.run(None, {"pixel_values": pixel.cpu().numpy().astype(np.float32)})[0]
    d = np.abs(onnx_out - ref_wrap.cpu().numpy())
    print(f"ONNX parity: max={d.max():.3e} mean={d.mean():.3e}")
    print("PASS" if d.max() < 5e-2 else "CHECK")


if __name__ == "__main__":
    main()
