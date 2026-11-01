"""Export the hand-crop CLIP ViT-L/14 (Mod-SpaMo aux stream) to ONNX.

Plain CLIP (no s2): per crop -> concat[CLS, patch_mean] = 2048-d. Input is the
HF CLIP processor output (N,3,224,224); host does decode/MediaPipe-hand-crop +
the processor (light glue). Checkpoint-independent (shared across quad variants).
Run with the SpaMo-PJM environment, from the root of the SpaMo-PJM repository.
"""
import os, sys
import numpy as np, torch, torch.nn as nn
from PIL import Image

_REPO = os.getcwd(); sys.path.insert(0, _REPO)
from transformers import AutoImageProcessor, CLIPVisionModel
from scripts.hand_crop_vit_extract import encode_crops

OUT = os.path.join(os.environ["SIGNSDK_SPAMO_SHARED"], "hand_clip.onnx")
CACHE = os.environ.get("SIGNSDK_HF_CACHE") or None


class HandCLIP(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model
    def forward(self, pixel_values):
        h = self.model(pixel_values, output_hidden_states=True).hidden_states[-1]
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=-1)   # (N, 2048)


def main():
    proc = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=CACHE)
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14",
                                            output_hidden_states=True, cache_dir=CACHE).cuda().eval()
    wrap = HandCLIP(model).eval().cuda()

    rng = np.random.RandomState(0)
    crops = [rng.randint(0, 255, (80, 60, 3), dtype=np.uint8) for _ in range(5)]
    pixel = proc(images=[Image.fromarray(c) for c in crops], return_tensors="pt").pixel_values.cuda()
    print("pixel:", tuple(pixel.shape))

    with torch.no_grad():
        ref_wrap = wrap(pixel)
        ref_enc = torch.from_numpy(encode_crops(crops, proc, model, "cuda")).cuda()
    print("wrap vs encode_crops:", float((ref_wrap - ref_enc).abs().max()))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    try:
        torch.onnx.export(wrap, (pixel,), OUT,
                          input_names=["pixel_values"], output_names=["features"],
                          dynamic_axes={"pixel_values": {0: "N"}, "features": {0: "N"}},
                          opset_version=17, do_constant_folding=True, dynamo=False)
        print("exported (legacy)")
    except Exception:
        from torch.export import Dim
        torch.onnx.export(wrap, (pixel,), OUT, dynamo=True,
                          input_names=["pixel_values"], output_names=["features"],
                          dynamic_shapes={"pixel_values": {0: Dim("N", min=1, max=4096)}})
        print("exported (dynamo)")

    import onnxruntime as ort
    prov = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(OUT, providers=prov)
    out = sess.run(None, {"pixel_values": pixel.cpu().numpy().astype(np.float32)})[0]
    d = np.abs(out - ref_wrap.cpu().numpy())
    print(f"ONNX parity: max={d.max():.3e} mean={d.mean():.3e}")
    print("PASS" if d.max() < 5e-2 else "CHECK")


if __name__ == "__main__":
    main()
