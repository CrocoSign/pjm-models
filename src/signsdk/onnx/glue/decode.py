"""Repo-free decoding helpers (numpy + an ONNX decoder session): greedy and
HF-faithful beam search over a no-cache decoder graph."""
from __future__ import annotations

import numpy as np


def _log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def sample_top_p(dec, eh, attn, start, eos, max_new_tokens, top_p=0.9, seed=0):
    """Stochastic top-p (nucleus) sampling, matching the SpaMo eval's do_sample=True."""
    rng = np.random.default_rng(seed)
    ids = [start]
    for _ in range(max_new_tokens):
        lg = dec.run(None, {"decoder_input_ids": np.array([ids], np.int64),
                            "encoder_hidden_states": eh, "encoder_attention_mask": attn})[0][0, -1]
        lg = lg.astype(np.float64) - lg.max()
        p = np.exp(lg); p /= p.sum()
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        keep = order[:max(1, int((cum <= top_p).sum()) + 1)]
        kp = p[keep]; kp /= kp.sum()
        nxt = int(rng.choice(keep, p=kp))
        if nxt == eos:
            break
        ids.append(nxt)
    return ids[1:]


def _top_k_top_p_warp(logprobs, top_k=50, top_p=0.9):
    """HF logits-warper chain for do_sample: TopK(50) then TopP(0.9), per beam row.
    Operates on log-probs (rows = beams); removed entries -> -inf. min_tokens_to_keep=1."""
    out = logprobs.copy()
    V = out.shape[-1]
    k = min(top_k, V)
    if k < V:
        kth = np.partition(out, V - k, axis=-1)[:, V - k][:, None]  # k-th largest per row
        out[out < kth] = -np.inf
    order = np.argsort(out, axis=-1)                                # ascending
    sl = np.take_along_axis(out, order, axis=-1)
    ex = np.exp(sl - sl.max(-1, keepdims=True)); ex[~np.isfinite(sl)] = 0.0
    cum = np.cumsum(ex / ex.sum(-1, keepdims=True), axis=-1)
    remove = cum <= (1.0 - top_p)
    remove[:, -1] = False                                           # always keep top-1
    remove_un = np.zeros_like(remove)
    np.put_along_axis(remove_un, order, remove, axis=-1)
    out[remove_un] = -np.inf
    return out


def beam_sample(dec, eh, attn, start, eos, num_beams, max_new_tokens,
                top_p=0.9, top_k=50, lp=1.0, seed=0):
    """HF-faithful beam-sample (num_beams>1 + do_sample=True): the exact SpaMo eval
    config (num_beams=5, top_p=0.9, default top_k=50, temperature=1.0). Stochastic;
    fix ``seed`` for reproducibility. NOTE: assumes temperature==1.0 (no temperature
    warper applied); the warp-before-beam-score reordering is only exact at T=1.0."""
    rng = np.random.default_rng(seed)
    beam_scores = np.full(num_beams, -1e9); beam_scores[0] = 0.0
    seqs, finished = [[start] for _ in range(num_beams)], []
    for _ in range(max_new_tokens):
        m = len(seqs)                                               # active beams (may shrink on eos)
        logits = dec.run(None, {"decoder_input_ids": np.array(seqs, np.int64),
                                "encoder_hidden_states": np.repeat(eh, m, 0),
                                "encoder_attention_mask": np.repeat(attn, m, 0)})[0]
        V = logits.shape[-1]
        warped = _top_k_top_p_warp(_log_softmax(logits[:, -1, :].astype(np.float64)), top_k, top_p)
        flat = (warped + beam_scores[:, None]).reshape(-1)
        ex = np.exp(flat - flat.max()); ex[~np.isfinite(flat)] = 0.0
        probs = ex / ex.sum()
        n = min(2 * num_beams, int((probs > 0).sum()))
        cand = rng.choice(flat.size, size=n, replace=False, p=probs)
        cand = cand[np.argsort(-flat[cand])]                        # sort by score desc
        ns, nsc = [], []
        for idx in cand:
            bi, tok = idx // V, idx % V
            if tok == eos:
                seq = seqs[bi][1:]
                finished.append((flat[idx] / (max(len(seq), 1) ** lp), seq))
            else:
                ns.append(seqs[bi] + [int(tok)]); nsc.append(float(flat[idx]))
            if len(ns) == num_beams:
                break
        if not ns:
            break
        seqs, beam_scores = ns, np.array(nsc)
        if len(finished) >= num_beams:                             # HF is_done: worst of top-K finished
            kth = sorted((s for s, _ in finished), reverse=True)[num_beams - 1]
            if kth >= beam_scores.max() / (len(seqs[0]) ** lp):
                break
    for s, seq in zip(beam_scores, seqs):
        finished.append((s / (max(len(seq) - 1, 1) ** lp), seq[1:]))
    return max(finished, key=lambda x: x[0])[1]


def greedy(dec, eh, attn, start, eos, max_new_tokens):
    ids = [start]
    for _ in range(max_new_tokens):
        lg = dec.run(None, {"decoder_input_ids": np.array([ids], np.int64),
                            "encoder_hidden_states": eh, "encoder_attention_mask": attn})[0]
        nxt = int(lg[0, -1].argmax())
        if nxt == eos:
            break
        ids.append(nxt)
    return ids[1:]


def beam_search(dec, eh, attn, start, eos, num_beams, max_new_tokens, lp=1.0):
    if num_beams <= 1:
        return greedy(dec, eh, attn, start, eos, max_new_tokens)
    eh_r, attn_r = np.repeat(eh, num_beams, 0), np.repeat(attn, num_beams, 0)
    beam_scores = np.full(num_beams, -1e9); beam_scores[0] = 0.0
    seqs, finished = [[start] for _ in range(num_beams)], []
    for _ in range(max_new_tokens):
        logits = dec.run(None, {"decoder_input_ids": np.array(seqs, np.int64),
                                "encoder_hidden_states": eh_r, "encoder_attention_mask": attn_r})[0]
        V = logits.shape[-1]
        scores = _log_softmax(logits[:, -1, :].astype(np.float64)) + beam_scores[:, None]
        flat = scores.reshape(-1)
        top = np.argpartition(-flat, 2 * num_beams)[:2 * num_beams]
        top = top[np.argsort(-flat[top])]
        ns, nsc = [], []
        for idx in top:
            bi, tok = idx // V, idx % V
            if tok == eos:
                seq = seqs[bi][1:]
                finished.append((flat[idx] / (max(len(seq), 1) ** lp), seq))
            else:
                ns.append(seqs[bi] + [int(tok)]); nsc.append(float(flat[idx]))
            if len(ns) == num_beams:
                break
        seqs, beam_scores = ns, np.array(nsc)
        if len(finished) >= num_beams and max(s for s, _ in finished) >= beam_scores.max() / (len(seqs[0]) ** lp):
            break
    for s, seq in zip(beam_scores, seqs):
        finished.append((s / (max(len(seq) - 1, 1) ** lp), seq[1:]))
    return max(finished, key=lambda x: x[0])[1]
