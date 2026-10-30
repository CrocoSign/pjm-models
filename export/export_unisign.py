"""Uni-Sign ONNX graph EXPORT (build step; repo-coupled).

Exports the 3 graphs (pose projector, mT5 encoder, mT5 decoder) + prompt prefix +
meta for a checkpoint into a cache dir. This needs the Uni-Sign repo (models.py /
datasets.py), so it does sys.path/chdir into the repo INSIDE the function. The
inference RUNTIME (unisign_runtime.UniSignRuntime) is repo-free and consumes these
graphs.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile

def export_graphs(checkpoint, out_dir, repo):
    """Export the 3 graphs; needs the Uni-Sign ``repo`` (models.py/datasets.py)."""
    cwd = os.getcwd()
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.chdir(repo)
    try:
        _export(checkpoint, out_dir)
    finally:
        os.chdir(cwd)


def _export(checkpoint, out_dir):
    import numpy as np, torch, torch.nn as nn, h5py
    from types import SimpleNamespace
    from torch.utils.data import DataLoader
    from datasets import S2T_Dataset_PJM
    from models import Uni_Sign

    margs = SimpleNamespace(dataset="PJM", task="SLT", pjm_split="si", max_length=256,
                            hidden_dim=256, rgb_support=False, label_smoothing=0.2)
    model = Uni_Sign(args=margs).cuda().eval()
    ret = model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"], strict=False)
    assert not ret.missing_keys and not ret.unexpected_keys, (ret.missing_keys, ret.unexpected_keys)
    mt5 = model.mt5_model.float()

    class PoseProjector(nn.Module):
        def __init__(s, m):
            super().__init__()
            s.modes = list(m.modes); s.proj_linear = m.proj_linear
            s.gcn_modules = m.gcn_modules; s.fusion_gcn_modules = m.fusion_gcn_modules
            s.part_para = m.part_para; s.pose_proj = m.pose_proj
        def forward(s, body, left, right, face_all):
            src = {"body": body, "left": left, "right": right, "face_all": face_all}
            feats, bf = [], None
            for p in s.modes:
                g = s.gcn_modules[p](s.proj_linear[p](src[p]).permute(0, 3, 1, 2))
                if p == "body": bf = g
                elif p == "left": g = g + bf[..., -2][..., None]
                elif p == "right": g = g + bf[..., -1][..., None]
                elif p == "face_all": g = g + bf[..., 0][..., None]
                feats.append(s.fusion_gcn_modules[p](g).mean(-1).transpose(1, 2))
            return s.pose_proj(torch.cat(feats, dim=-1) + s.part_para)

    class ENC(nn.Module):
        def __init__(s, m): super().__init__(); s.enc = m.encoder
        def forward(s, ie, am): return s.enc(inputs_embeds=ie, attention_mask=am).last_hidden_state

    class DEC(nn.Module):
        def __init__(s, m):
            super().__init__(); s.m = m
            s.tie = m.config.tie_word_embeddings; s.dm = m.config.d_model
        def forward(s, di, eh, eam):
            h = s.m.decoder(input_ids=di, encoder_hidden_states=eh, encoder_attention_mask=eam).last_hidden_state
            if s.tie: h = h * (s.dm ** -0.5)
            return s.m.lm_head(h)

    tmp = tempfile.mkdtemp(); key = "004_000_M"
    with open(f"{tmp}/k.csv", "w", newline="") as f:
        csv.writer(f).writerows([["key"], [key]])
    with h5py.File(f"{tmp}/t.h5", "w") as h:
        h.create_dataset(key, data=np.bytes_(""))
    ds = S2T_Dataset_PJM(path=f"{tmp}/k.csv", texts_path=f"{tmp}/t.h5", args=margs, phase="test")
    src, tgt = next(iter(DataLoader(ds, batch_size=1, collate_fn=ds.collate_fn)))
    for k in src:
        if hasattr(src[k], "cuda"):
            src[k] = src[k].cuda().float() if src[k].is_floating_point() else src[k].cuda()
    body, left, right, face = (src[k] for k in ["body", "left", "right", "face_all"])

    os.makedirs(out_dir, exist_ok=True)
    proj, enc, dec = PoseProjector(model).eval(), ENC(mt5).eval(), DEC(mt5).eval()
    dyn = lambda d: {0: "B", 1: d}
    torch.onnx.export(proj, (body, left, right, face), f"{out_dir}/projector.onnx",
                      input_names=["body", "left", "right", "face_all"], output_names=["pose_embeds"],
                      dynamic_axes={"body": dyn("T"), "left": dyn("T"), "right": dyn("T"),
                                    "face_all": dyn("T"), "pose_embeds": dyn("T")},
                      opset_version=17, do_constant_folding=True)
    with torch.no_grad():
        stack = model(src, tgt)
    ie, am = stack["inputs_embeds"].float(), stack["attention_mask"]
    torch.onnx.export(enc, (ie, am), f"{out_dir}/encoder.onnx",
                      input_names=["inputs_embeds", "attention_mask"], output_names=["encoder_hidden_states"],
                      dynamic_axes={"inputs_embeds": dyn("S"), "attention_mask": dyn("S"),
                                    "encoder_hidden_states": dyn("S")},
                      opset_version=17, do_constant_folding=True)
    with torch.no_grad():
        eh = enc(ie, am)
    di = torch.tensor([[mt5.config.decoder_start_token_id, 100, 200]], device="cuda")
    torch.onnx.export(dec, (di, eh, am), f"{out_dir}/decoder.onnx",
                      input_names=["decoder_input_ids", "encoder_hidden_states", "encoder_attention_mask"],
                      output_names=["logits"],
                      dynamic_axes={"decoder_input_ids": dyn("L"), "encoder_hidden_states": dyn("S"),
                                    "encoder_attention_mask": dyn("S"), "logits": dyn("L")},
                      opset_version=17, do_constant_folding=True)
    with torch.no_grad():
        pt = model.mt5_tokenizer([f"Translate sign language video to {model.lang}: "],
                                 padding="longest", return_tensors="pt").to("cuda")
        np.save(f"{out_dir}/prefix.npy", mt5.encoder.embed_tokens(pt["input_ids"]).float().cpu().numpy())
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump({"decoder_start_token_id": int(mt5.config.decoder_start_token_id),
                   "eos_token_id": int(model.mt5_tokenizer.eos_token_id)}, f)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Export a Uni-Sign checkpoint to projector/encoder/decoder ONNX graphs.")
    ap.add_argument("checkpoint", help="best_checkpoint.pth, absolute path")
    ap.add_argument("out_dir", help="output directory, absolute path")
    ap.add_argument("--repo", required=True, help="Uni-Sign-PJM checkout (needs dataset/PJM/pose_format/004_000_M.pkl)")
    a = ap.parse_args()
    export_graphs(a.checkpoint, a.out_dir, a.repo)
