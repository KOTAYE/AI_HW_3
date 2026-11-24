"""Explainability (bonus).

1) `lr`      : global explanation of the TF-IDF + logistic regression baseline - n-grams with the largest
               per-class coefficients.
2) `ig`      : Integrated Gradients (Sundararajan et al., 2017) on the input embeddings of the fine-tuned encoder,
               for val texts of every class; subword attributions are merged into words and aggregated
               into the most class-indicative words. Implemented directly (no captum): 16-step Riemann sum
               from an all-[PAD]-embedding baseline.
3) `fusion`  : does the encoder look at the LLM-written part of a fusion text? For val fusion texts whose human
               source is also in val (same reconstructed group) we know exactly which words the LLM wrote
               (difflib alignment). We slide a short window (128 tokens, stride 64) over the text and correlate the
               window's share of LLM-written words with the encoder's P(human) for that window.

    python src/xai.py lr | ig --model models/mdeberta | fusion --model models/mdeberta
Outputs go to results/xai_*.csv / .json
"""
import argparse
import difflib
import json
import os
import re

import numpy as np
import pandas as pd
import torch

from common import LABELS, RES, fixed_sets, y_of


def run_lr():
    from baseline import make
    sub, val, _ = fixed_sets()
    pipe = make({"feat": "word", "model": "lr", "C": 16, "wng": (1, 2)}).fit(sub.text, y_of(sub))
    vocab = np.array(pipe.named_steps["vec"].get_feature_names_out())
    coef = pipe.named_steps["clf"].coef_
    rows = []
    for k, lab in enumerate(LABELS):
        for i in np.argsort(-coef[k])[:30]:
            rows.append({"class": lab, "ngram": vocab[i], "coef": round(float(coef[k, i]), 3)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "xai_lr_top.csv"), index=False)
    for lab in LABELS:
        print(lab, df[df["class"] == lab].ngram.head(25).tolist())


def load_model(path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).cuda().eval()
    return tok, model


def integrated_gradients(model, emb, ids, target, steps=16, bs=8):
    """IG of the target logit w.r.t. input embeddings for one sequence; baseline = PAD embedding."""
    x = emb(ids)                                   # [1, L, H]
    base = emb(torch.full_like(ids, model.config.pad_token_id or 0))
    total = torch.zeros_like(x)
    alphas = torch.linspace(1 / steps, 1, steps, device=x.device)
    for s in range(0, steps, bs):
        a = alphas[s:s + bs][:, None, None]
        inp = (base + a * (x - base)).detach().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(inputs_embeds=inp, attention_mask=torch.ones(inp.shape[:2], device=x.device, dtype=torch.long)).logits
        g, = torch.autograd.grad(out[:, target].sum(), inp)
        total += g.float().sum(0, keepdim=True)
    return ((x - base) * total / steps).sum(-1)[0].detach().float().cpu().numpy()


def words_from_tokens(tok, ids, attr):
    """Merge sentencepiece / BPE subwords into words, summing attributions."""
    pieces = tok.convert_ids_to_tokens(ids)
    words, scores = [], []
    for p, a in zip(pieces, attr):
        new = p.startswith("▁") or p.startswith("Ġ") or not words
        w = p.lstrip("▁Ġ")
        if new:
            words.append(w)
            scores.append(a)
        else:
            words[-1] += w
            scores[-1] += a
    return words, scores


def run_ig(path, per_class=150, max_len=256):
    tok, model = load_model(path)
    emb = model.get_input_embeddings()
    _, val, _ = fixed_sets()
    rng = np.random.default_rng(0)
    rows, examples = [], []
    for k, lab in enumerate(LABELS):
        idx = rng.choice(np.where(y_of(val) == k)[0], per_class, replace=False)
        for j, i in enumerate(idx):
            ids = tok(val.text.iloc[i], truncation=True, max_length=max_len, return_tensors="pt")["input_ids"].cuda()
            with torch.no_grad():
                pred = int(model(input_ids=ids).logits.argmax(-1))
            attr = integrated_gradients(model, emb, ids, target=k)
            words, sc = words_from_tokens(tok, ids[0].tolist(), attr)
            for w, s in zip(words, sc):
                if re.search(r"\w", w) or w.strip():
                    rows.append({"class": lab, "word": w.lower(), "attr": float(s), "correct": pred == k})
            if j < 2:
                examples.append({"class": lab, "pred": LABELS[pred], "id": val.id.iloc[i],
                                 "tokens": [[w, round(float(s), 4)] for w, s in zip(words, sc)]})
    df = pd.DataFrame(rows)
    agg = (df.groupby(["class", "word"]).attr.agg(["mean", "count"]).reset_index())
    agg = agg[agg["count"] >= 15]
    top = agg.sort_values("mean", ascending=False).groupby("class").head(25)
    top.to_csv(os.path.join(RES, "xai_ig_top_words.csv"), index=False)
    json.dump(examples, open(os.path.join(RES, "xai_ig_examples.json"), "w", encoding="utf-8"), ensure_ascii=False)
    for lab in LABELS:
        print(lab, top[top["class"] == lab].word.tolist())


def llm_mask(human, fusion):
    """1 for every word of the fusion text that is not copied from the human source."""
    h, f = human.split(), fusion.split()
    m = np.ones(len(f))
    for a, b, n in difflib.SequenceMatcher(None, h, f, autojunk=False).get_matching_blocks():
        if n >= 3:                      # ignore accidental 1-2 word matches
            m[b:b + n] = 0
    return f, m


@torch.no_grad()
def run_fusion(path, n_texts=300, win=128, stride=64):
    tok, model = load_model(path)
    _, val, _ = fixed_sets()
    pairs = []
    for g, d in val.groupby("group"):
        if set(d.label) >= {"human", "fusion"}:
            pairs.append((d[d.label == "human"].text.iloc[0], d[d.label == "fusion"].text.iloc[0]))
        if len(pairs) >= n_texts:
            break
    rows = []
    for t, (h, f) in enumerate(pairs):
        words, m = llm_mask(h, f)
        enc = tok(words, is_split_into_words=True, add_special_tokens=False)
        wid = np.array([w for w in enc.word_ids()])
        ids = enc["input_ids"]
        for s in range(0, max(len(ids) - win, 0) + 1, stride):
            chunk = ids[s:s + win]
            x = torch.tensor([[tok.cls_token_id] + chunk + [tok.sep_token_id]]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = torch.softmax(model(input_ids=x).logits.float(), -1)[0].cpu().numpy()
            rows.append({"text": t, "start": s, "llm_share": float(m[wid[s:s + win]].mean()),
                         **{f"p_{l}": float(v) for l, v in zip(LABELS, p)}})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "xai_fusion_windows.csv"), index=False)
    df["bin"] = pd.cut(df.llm_share, [-.01, .05, .25, .5, .75, .95, 1.0])
    summ = df.groupby("bin", observed=True)[["p_human", "p_ai", "p_fusion"]].agg(["mean", "count"]).round(3)
    print(summ)
    print("spearman(llm_share, p_human) =", round(df[["llm_share", "p_human"]].corr("spearman").iloc[0, 1], 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["lr", "ig", "fusion"])
    ap.add_argument("--model", default="models/mdeberta")
    a = ap.parse_args()
    path = a.model if os.path.isabs(a.model) else os.path.join(os.path.dirname(RES), a.model)
    {"lr": run_lr, "ig": lambda: run_ig(path), "fusion": lambda: run_fusion(path)}[a.mode]()


if __name__ == "__main__":
    main()
