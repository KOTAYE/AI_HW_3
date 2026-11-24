"""Fine-tune a multilingual encoder (mDeBERTa-v3 / XLM-R) for human / ai / fusion classification.

Long texts (median ~700 subword tokens, max > 4k): during training every epoch takes a random window of
`max_len` tokens from each text (so the model sees all parts of long texts); at inference the text is
split into overlapping windows and the per-window logits are saved, so different aggregations
(first window only, head+tail, mean over windows) can be compared offline without re-running the model.

    python src/train_encoder.py --model microsoft/mdeberta-v3-base --name mdeberta --epochs 2
    python src/train_encoder.py ... --train-set sub      (default: fixed 50k subsample)
Outputs: preds/<name>_chunks_{val,test}.npz  (logits per window + doc index), preds/<name>_{val,test}.npy (mean agg.)
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import PRED, ROOT, fixed_sets, load, report, save_probs, y_of


def tokenize(tok, texts, bs=2000):
    ids = []
    for s in range(0, len(texts), bs):
        ids += tok(list(texts[s:s + bs]), add_special_tokens=False, truncation=False)["input_ids"]
    return ids


def windows(ids, body, stride, max_windows):
    """Overlapping windows over a token list; always includes the last window (the tail)."""
    if len(ids) <= body:
        return [ids]
    starts = list(range(0, len(ids) - body + 1, stride))
    if starts[-1] != len(ids) - body:
        starts.append(len(ids) - body)
    if len(starts) > max_windows:  # keep head, tail and evenly spaced middle windows
        keep = np.unique(np.linspace(0, len(starts) - 1, max_windows).round().astype(int))
        starts = [starts[k] for k in keep]
    return [ids[s:s + body] for s in starts]


def collate(seqs, cls, sep, pad):
    L = max(len(s) for s in seqs) + 2
    x = torch.full((len(seqs), L), pad, dtype=torch.long)
    m = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        s = [cls] + s + [sep]
        x[i, :len(s)] = torch.tensor(s)
        m[i, :len(s)] = 1
    return x, m


@torch.inference_mode()
def predict(model, tok, ids, body, stride, max_windows, bs, dev):
    model.eval()
    chunks, doc = [], []
    for d, s in enumerate(ids):
        for w in windows(s, body, stride, max_windows):
            chunks.append(w)
            doc.append(d)
    order = np.argsort([len(c) for c in chunks])[::-1]  # length-sorted batches: much faster
    out = np.zeros((len(chunks), model.config.num_labels), np.float32)
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        x, m = collate([chunks[i] for i in idx], tok.cls_token_id, tok.sep_token_id, tok.pad_token_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[idx] = model(input_ids=x.to(dev), attention_mask=m.to(dev)).logits.float().cpu().numpy()
    return out, np.array(doc)


def mean_agg(logits, doc, n):
    p = torch.softmax(torch.tensor(logits), 1).numpy()
    s = np.zeros((n, p.shape[1]))
    np.add.at(s, doc, p)
    return s / np.bincount(doc, minlength=n)[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/mdeberta-v3-base")
    ap.add_argument("--name", default="mdeberta")
    ap.add_argument("--train-set", default="sub", choices=["sub", "full"])
    ap.add_argument("--max-train", type=int, default=0, help="cap number of training texts (0 = all of train-set)")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--stride", type=int, default=384)
    ap.add_argument("--max-windows", type=int, default=8)
    ap.add_argument("--head-only", action="store_true", help="train on the first window only (reference)")
    ap.add_argument("--class-weights", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--train-emb", action="store_true", help="also fine-tune the word embedding matrix")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--mem-frac", type=float, default=0.85)
    ap.add_argument("--eval-bs", type=int, default=16)
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline end to end")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    # on Windows (WDDM) CUDA silently spills to system RAM when VRAM is full, which makes training ~10x slower;
    # capping the caching allocator turns that into an explicit OOM instead
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    rng = np.random.default_rng(a.seed)
    dev = "cuda"

    sub, val, test = fixed_sets()
    if a.train_set == "full":
        tr = load("train")
        sub = tr[tr.fold != 0].reset_index(drop=True)
    if a.max_train:
        sub = sub.sample(a.max_train, random_state=a.seed).reset_index(drop=True)
    if a.smoke:
        sub, val, test = sub.head(320), val.head(300), test.head(100)
    tok = AutoTokenizer.from_pretrained(a.model)
    t0 = time.time()
    tr_ids = tokenize(tok, sub.text.values)
    va_ids = tokenize(tok, val.text.values)
    te_ids = [] if a.no_test else tokenize(tok, test.text.values)
    print(f"tokenized in {time.time() - t0:.0f}s; train tokens median {np.median([len(x) for x in tr_ids]):.0f}", flush=True)
    ytr = torch.tensor(y_of(sub))
    body = a.max_len - 2

    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=3, dtype=torch.float32).to(dev)  # v5 defaults to the checkpoint dtype (fp16) -> NaN with AdamW
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
    if not a.train_emb:
        # the 250k-token multilingual vocabulary is ~2/3 of all weights; freezing it saves ~3 GB of
        # gradients + Adam state, which is what lets the model train on an 8 GB GPU
        model.get_input_embeddings().weight.requires_grad_(False)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for _, p in params) / 1e6:.0f}M", flush=True)
    decay = [p for n, p in params if p.ndim > 1]
    nodecay = [p for n, p in params if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": nodecay, "weight_decay": 0}], lr=a.lr)
    steps_per_epoch = math.ceil(len(tr_ids) / (a.bs * a.accum))
    total = int(steps_per_epoch * a.epochs)
    sch = get_linear_schedule_with_warmup(opt, int(a.warmup * total), total)
    cw = None
    if a.class_weights:
        cnt = np.bincount(ytr.numpy(), minlength=3)
        cw = torch.tensor(cnt.sum() / (3 * cnt), dtype=torch.float32, device=dev)
        print("class weights", cw.tolist())

    step, t0 = 0, time.time()
    model.train()
    ep = 0
    while step < total:
        perm = rng.permutation(len(tr_ids))
        # bucket by length inside blocks of 50 batches to cut padding but keep randomness
        blk = a.bs * 50
        perm = np.concatenate([p[np.argsort([min(len(tr_ids[i]), body) for i in p])] for p in np.array_split(perm, max(1, len(perm) // blk))])
        batches = [perm[i:i + a.bs] for i in range(0, len(perm), a.bs)]
        rng.shuffle(batches)
        run_loss = 0.0
        for bi, idx in enumerate(batches):
            seqs = []
            for i in idx:
                s = tr_ids[i]
                st = 0 if (a.head_only or len(s) <= body) else int(rng.integers(0, len(s) - body + 1))
                seqs.append(s[st:st + body])
            x, m = collate(seqs, tok.cls_token_id, tok.sep_token_id, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=x.to(dev), attention_mask=m.to(dev)).logits.float()
            loss = F.cross_entropy(logits, ytr[idx].to(dev), weight=cw) / a.accum
            loss.backward()
            run_loss += loss.item()
            if (bi + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % a.log_every == 0:
                    el = time.time() - t0
                    print(f"ep {ep} step {step}/{total} loss {run_loss / a.log_every:.4f} {el / 60:.1f}min eta {(total - step) * el / step / 60:.0f}min", flush=True)
                    run_loss = 0.0
                if step >= total:
                    break
        ep += 1
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    model.save_pretrained(os.path.join(ROOT, "models", a.name))
    tok.save_pretrained(os.path.join(ROOT, "models", a.name))

    del opt, sch
    torch.cuda.empty_cache()
    t0 = time.time()
    lv, dv = predict(model, tok, va_ids, body, a.stride, a.max_windows, a.eval_bs, dev)
    np.savez(os.path.join(PRED, f"{a.name}_chunks_val.npz"), logits=lv, doc=dv)
    pv = mean_agg(lv, dv, len(val))
    first = np.array([np.where(dv == d)[0][0] for d in range(len(val))])
    report(f"{a.name}_first", y_of(val), lv[first], {"train": f"{a.train_set}{len(sub)}"})
    report(f"{a.name}_meanchunks", y_of(val), pv, {"train": f"{a.train_set}{len(sub)}", "infer_sec": int(time.time() - t0)})
    pt = None
    if not a.no_test:
        lt, dt = predict(model, tok, te_ids, body, a.stride, a.max_windows, a.eval_bs, dev)
        np.savez(os.path.join(PRED, f"{a.name}_chunks_test.npz"), logits=lt, doc=dt)
        pt = mean_agg(lt, dt, len(test))
    save_probs(a.name, pv, pt)


if __name__ == "__main__":
    main()
