"""Hand-crafted stylometric features (no model, CPU only).

Global style (lengths, punctuation, typography, vocabulary richness, typical LLM connectives) plus
"inconsistency" features: the same statistics on consecutive 60-word blocks and their spread.
A fusion text is a human text where only some parts were rewritten, so its style should vary more
inside the text than a fully human or fully AI one.

    python src/stylometry.py   ->  preds/stylo_{sub,val,test}.parquet
"""
import os
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import PRED, fixed_sets

SENT = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"\w+", re.UNICODE)
PUNCT = {"comma": ",", "semi": ";", "colon": ":", "excl": "!", "quest": "?", "dash_em": "—", "dash_en": "–",
         "hyph": "-", "lpar": "(", "quote_ascii": '"', "quote_curly": "“", "quote_low": "„", "apos_ascii": "'",
         "apos_curly": "’", "ellipsis": "…", "star": "*", "hash": "#"}
CONNECT = ["furthermore", "moreover", "additionally", "in conclusion", "overall", "however", "notably", "ultimately",
           "crucial", "delve", "significant", "landscape", "foster", "ensure", "underscore", "highlight", "insbesondere",
           "darüber hinaus", "zudem", "letztlich", "jedoch", "zusammenfassend", "entscheidend"]
BLOCK = 60


def block_stats(words):
    n = len(words) // BLOCK
    if n < 2:
        return {"blk_wlen_std": 0.0, "blk_ttr_std": 0.0, "blk_long_std": 0.0}
    wl, ttr, lg = [], [], []
    for b in range(n):
        w = words[b * BLOCK:(b + 1) * BLOCK]
        L = np.array([len(x) for x in w])
        wl.append(L.mean())
        ttr.append(len(set(x.lower() for x in w)) / BLOCK)
        lg.append((L > 8).mean())
    return {"blk_wlen_std": float(np.std(wl)), "blk_ttr_std": float(np.std(ttr)), "blk_long_std": float(np.std(lg))}


def feats(t):
    words = WORD.findall(t)
    nw = max(len(words), 1)
    nc = max(len(t), 1)
    sents = [s for s in SENT.split(t) if s.strip()]
    sl = np.array([len(WORD.findall(s)) for s in sents]) if sents else np.array([0])
    low = t.lower()
    wl = np.array([len(w) for w in words]) if words else np.array([0])
    f = {"n_words": len(words), "n_chars": len(t), "log_words": np.log1p(len(words)), "n_sent": len(sents),
         "sent_mean": sl.mean(), "sent_std": sl.std(), "sent_cv": sl.std() / (sl.mean() + 1e-6),
         "sent_max": sl.max(), "wlen_mean": wl.mean(), "wlen_std": wl.std(),
         "ttr": len(set(w.lower() for w in words[:300])) / max(min(nw, 300), 1),
         "upper_ratio": sum(c.isupper() for c in t) / nc, "digit_ratio": sum(c.isdigit() for c in t) / nc,
         "newline_rate": t.count("\n") / nw * 100, "para_rate": t.count("\n\n") / nw * 100,
         "space_before_punct": len(re.findall(r" [.,;:!?]", t)) / nw * 100,
         "double_space": t.count("  ") / nw * 100, "nonascii": sum(ord(c) > 127 for c in t) / nc,
         "caps_words": sum(w.isupper() and len(w) > 1 for w in words) / nw}
    for k, ch in PUNCT.items():
        f[f"p_{k}"] = t.count(ch) / nw * 100
    for c in CONNECT:
        f[f"w_{c.replace(' ', '_')}"] = low.count(c) / nw * 100
    f.update(block_stats(words))
    return f


def build(texts):
    with Pool(max(os.cpu_count() - 2, 1)) as p:
        return pd.DataFrame(p.map(feats, texts, chunksize=500))


if __name__ == "__main__":
    sub, val, test = fixed_sets()
    for name, df in [("sub", sub), ("val", val), ("test", test)]:
        f = build(df.text.tolist())
        f.insert(0, "id", df.id.values)
        f.to_parquet(os.path.join(PRED, f"stylo_{name}.parquet"))
        print(name, f.shape, flush=True)
