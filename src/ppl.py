"""Perplexity-style features from an open-weight causal LM (default Qwen2.5-1.5B, multilingual incl. German).

For every token t of the first `max_tokens` tokens we take log p(x_t), the entropy H_t of the predicted
distribution and its log-prob variance. Global features follow the "Fast-DetectGPT" idea
(Bao et al., 2024): analytic curvature  (sum log p - sum E[log p]) / sqrt(sum Var[log p]).
Because fusion texts are only partly rewritten by an LLM, we also compute the same statistics on
consecutive 64-token segments and keep their spread (std / min / max / range) - a human segment next to an
AI segment should give a large spread.

    python src/ppl.py --model Qwen/Qwen2.5-1.5B --name qwen15        ->  preds/ppl_<name>_{sub,val,test}.parquet
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from common import PRED, fixed_sets
from transformers import AutoModelForCausalLM, AutoTokenizer

SEG = 64


def seg_stats(x, prefix):
    n = len(x) // SEG
    if n < 2:
        return {f"{prefix}_std": 0.0, f"{prefix}_min": float(x.mean()), f"{prefix}_max": float(x.mean()), f"{prefix}_rng": 0.0}
    s = x[:n * SEG].reshape(n, SEG).mean(1)
    return {f"{prefix}_std": float(s.std()), f"{prefix}_min": float(s.min()), f"{prefix}_max": float(s.max()),
            f"{prefix}_rng": float(s.max() - s.min())}


def text_features(lp, ent, var, rank):
    f = {"n_tok": len(lp), "lp_mean": lp.mean(), "lp_std": lp.std(), "lp_p10": np.percentile(lp, 10),
         "lp_p50": np.median(lp), "lp_p90": np.percentile(lp, 90), "ent_mean": ent.mean(), "ent_std": ent.std(),
         "top1": (rank == 0).mean(), "top10": (rank < 10).mean(), "lrr": -lp.mean() / max(np.log1p(rank).mean(), 1e-6),
         "fdg": (lp + ent).sum() / np.sqrt(var.sum() + 1e-6),          # Fast-DetectGPT analytic score
         "fdg_tok": ((lp + ent) / np.sqrt(var + 1e-6)).mean()}
    f.update(seg_stats(lp, "seg_lp"))
    f.update(seg_stats((lp + ent) / np.sqrt(var + 1e-6), "seg_fdg"))
    f.update(seg_stats(ent, "seg_ent"))
    return f


@torch.no_grad()
def run(model, tok, texts, max_tokens, bs, dev):
    enc = tok(list(texts), add_special_tokens=False, truncation=True, max_length=max_tokens)["input_ids"]
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    order = np.argsort([len(e) for e in enc])[::-1]
    feats = [None] * len(enc)
    t0 = time.time()
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        seqs = [[bos] + enc[i] for i in idx]
        L = max(len(s) for s in seqs)
        x = torch.full((len(seqs), L), tok.pad_token_id or 0, dtype=torch.long)
        m = torch.zeros((len(seqs), L), dtype=torch.long)
        for k, s in enumerate(seqs):
            x[k, :len(s)] = torch.tensor(s)
            m[k, :len(s)] = 1
        x, m = x.to(dev), m.to(dev)
        # full-vocabulary (150k) float32 logits for a whole batch do not fit in 8 GB: keep the batch in bf16
        # and do the float32 vocabulary statistics one sequence at a time
        logits = model(input_ids=x, attention_mask=m).logits
        for k, i in enumerate(idx):
            n = len(seqs[k]) - 1
            logp = torch.log_softmax(logits[k, :n].float(), -1)
            tgt = x[k, 1:n + 1]
            lp = logp.gather(-1, tgt[:, None])[:, 0]
            p = logp.exp()
            ent = -(p * logp).sum(-1)
            var = (p * logp.pow(2)).sum(-1) - ent.pow(2)
            rank = (logp > lp[:, None]).sum(-1)
            feats[i] = text_features(lp.cpu().numpy(), ent.cpu().numpy(), var.clamp_min(0).cpu().numpy(), rank.cpu().numpy())
            del logp, p
        del logits
        if (b // bs) % 200 == 0:
            print(f"  {b}/{len(order)} {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--name", default="qwen15")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--sets", nargs="+", default=["val", "test", "sub"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    sub, val, test = fixed_sets()
    for name, df in zip(["sub", "val", "test"], [sub, val, test]):
        if name not in a.sets:
            continue
        if a.limit:
            df = df.head(a.limit)
        t0 = time.time()
        f = run(model, tok, df.text.values, a.max_tokens, a.bs, dev)
        f.insert(0, "id", df.id.values)
        f.to_parquet(os.path.join(PRED, f"ppl_{a.name}_{name}{'_lim' if a.limit else ''}.parquet"))
        print(name, len(df), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
