"""Generalized per-checkpoint SpaMo ONNX exporter (any of the 8 variants).

Exports, for a given (config, ckpt) into --out:
  * projector.onnx   -- projectors + (quad: aux/kp + cross-attn) + TemporalConv + fusion_proj
  * encoder.onnx, decoder.onnx -- LoRA-merged flan-t5-xl seq2seq
  * prompt_embeds.npy, meta.json  -- fixed prompt + decoder ids + kind
CLIP/VideoMAE/hand-CLIP are checkpoint-independent (see export_spamo_shared.py).

  python tools/onnx/export_spamo_model.py --config CFG.yaml --ckpt CKPT.ckpt --out DIR
Run with the SpaMo-PJM environment, from the root of the SpaMo-PJM repository.
"""
import argparse, gc, json, os, sys
import numpy as np, torch, torch.nn as nn, h5py

_REPO = os.getcwd(); sys.path.insert(0, _REPO)
torch.serialization.add_safe_globals([
    np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType, np.dtypes.Int64DType])
from omegaconf import OmegaConf
from utils.helpers import instantiate_from_config

VAL_KEY = "001_003_M"


class Projector(nn.Module):
    """Dual or quad projector -> visual_embeds (1, L, 2048)."""
    def __init__(self, m, quad):
        super().__init__()
        self.quad = quad
        self.spatio_proj = m.spatio_proj
        self.spatiotemp_proj = m.spatiotemp_proj
        self.temporal_conv = m.temporal_encoder.temporal_conv
        self.fusion_proj = m.fusion_proj
        if quad:
            self.aux_proj = m.aux_proj
            self.kp_proj = m.kp_proj
            self.aux_xattn = m.aux_xattn

    def forward(self, pixel, glor, aux=None, kp=None):
        s = self.spatio_proj(pixel)
        mo = self.spatiotemp_proj(glor)
        if self.quad:
            context = torch.cat([self.aux_proj(aux), self.kp_proj(kp)], dim=1)
            s = s + self.aux_xattn(s, context)
        joint = torch.cat([s, mo], dim=1)
        v = self.temporal_conv(joint.permute(0, 2, 1))
        return self.fusion_proj(v.permute(0, 2, 1))


class EncWrap(nn.Module):
    def __init__(s, t5): super().__init__(); s.enc = t5.encoder
    def forward(s, ie, am): return s.enc(inputs_embeds=ie, attention_mask=am).last_hidden_state


class DecWrap(nn.Module):
    def __init__(s, t5):
        super().__init__(); s.t5 = t5; s.tie = t5.config.tie_word_embeddings; s.dm = t5.config.d_model
    def forward(s, di, eh, eam):
        h = s.t5.decoder(input_ids=di, encoder_hidden_states=eh, encoder_attention_mask=eam).last_hidden_state
        if s.tie: h = h * (s.dm ** -0.5)
        return s.t5.lm_head(h)


def _to_cuda(x):
    if torch.is_tensor(x): return x.cuda()
    if isinstance(x, list): return [_to_cuda(i) for i in x]
    if isinstance(x, dict): return {k: _to_cuda(v) for k, v in x.items()}
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from torch.export import Dim

    cfg = OmegaConf.load(a.config)
    quad = int(cfg.model.params.get("aux_input_size", 0)) > 0
    print(f"[export] kind={'quad' if quad else 'dual'}")
    model = instantiate_from_config(cfg.model)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["state_dict"], strict=False)
    model = model.eval().cuda()
    model.use_in_context = False

    # ---- projector ----
    d = cfg.data.params
    with h5py.File(d.visual_features_path, "r") as f:
        pixel = torch.from_numpy(f[VAL_KEY][()]).float()
    with h5py.File(d.motion_features_path, "r") as f:
        glor = torch.from_numpy(f[VAL_KEY][()]).float()
    pix1, glor1 = pixel[None].cuda(), glor[None].cuda()
    fe = Projector(model, quad).eval().cuda().float()
    ins = (pix1, glor1)
    names = ["pixel", "glor"]
    dyn = {"pixel": {1: Dim("T", min=8, max=4096)}, "glor": {1: Dim("n", min=1, max=4096)}}
    if quad:
        with h5py.File(d.aux_features_path, "r") as f:
            aux = torch.from_numpy(f[VAL_KEY][()]).float()[None].cuda()
        with h5py.File(d.keypoint_features_path, "r") as f:
            kp = torch.from_numpy(f[VAL_KEY][()]).float()[None].cuda()
        ins = (pix1, glor1, aux, kp)
        names += ["aux", "kp"]
        dyn["aux"] = {1: Dim("Ta", min=1, max=4096)}
        dyn["kp"] = {1: Dim("Tk", min=1, max=4096)}
    torch.onnx.export(fe, ins, f"{a.out}/projector.onnx", dynamo=True,
                      input_names=names, output_names=["visual_embeds"], dynamic_shapes=dyn)
    print("[export] projector.onnx")

    # ---- prompt + meta ----
    prompt = model.prompt.format("English")
    toks = model.t5_tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")
    with torch.no_grad():
        pe = model.t5_model.encoder.embed_tokens(toks.input_ids).float().cpu().numpy()
    np.save(f"{a.out}/prompt_embeds.npy", pe)
    json.dump({"kind": "quad" if quad else "dual", "prompt": prompt,
               "decoder_start_token_id": int(model.t5_model.config.decoder_start_token_id),
               "eos_token_id": int(model.t5_tokenizer.eos_token_id)}, open(f"{a.out}/meta.json", "w"))

    # ---- T5 (LoRA-merged), built inputs_embeds via native path ----
    model.t5_model = model.t5_model.merge_and_unload()
    t5 = model.t5_model
    sample = {"id": VAL_KEY, "pixel_value": pixel, "num_frames": int(pixel.shape[0]),
              "glor_value": glor, "keypoint_value": None, "aux_value": None,
              "text": "", "lang": "English", "speaker_id": "unknown"}
    if quad:
        sample["keypoint_value"] = kp[0].cpu()
        sample["aux_value"] = aux[0].cpu()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        inp = _to_cuda(model.get_inputs([sample]))
        vo, vm = model.prepare_visual_inputs(inp)
        vo = model.fusion_proj(vo)
        ie, im, _, _ = model.prepare_inputs(vo, vm, inp, "test", 0)
    ie = ie.float().cpu(); im = im.cpu()
    del model, fe, pix1, glor1
    gc.collect(); torch.cuda.empty_cache()

    t5_cpu = t5.float().cpu()
    enc_w, dec_w = EncWrap(t5_cpu).eval(), DecWrap(t5_cpu).eval()
    torch.onnx.export(enc_w, (ie, im), f"{a.out}/encoder.onnx",
                      input_names=["inputs_embeds", "attention_mask"], output_names=["encoder_hidden_states"],
                      dynamic_axes={"inputs_embeds": {0: "B", 1: "S"}, "attention_mask": {0: "B", 1: "S"},
                                    "encoder_hidden_states": {0: "B", 1: "S"}},
                      opset_version=17, do_constant_folding=True, dynamo=False)
    with torch.no_grad():
        eh = enc_w(ie, im)
    di = torch.tensor([[t5.config.decoder_start_token_id, 100, 200]])
    torch.onnx.export(dec_w, (di, eh, im), f"{a.out}/decoder.onnx",
                      input_names=["decoder_input_ids", "encoder_hidden_states", "encoder_attention_mask"],
                      output_names=["logits"],
                      dynamic_axes={"decoder_input_ids": {0: "B", 1: "L"}, "encoder_hidden_states": {0: "B", 1: "S"},
                                    "encoder_attention_mask": {0: "B", 1: "S"}, "logits": {0: "B", 1: "L"}},
                      opset_version=17, do_constant_folding=True, dynamo=False)
    print("[export] encoder.onnx + decoder.onnx  DONE ->", a.out)


if __name__ == "__main__":
    main()
