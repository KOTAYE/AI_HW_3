"""Shared paths, label mapping and helpers."""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, classification_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("HW3_DATA", os.path.join(ROOT, "data"))
PRED = os.path.join(ROOT, "preds")        # cached class probabilities: <name>_val.npy / <name>_test.npy
RES = os.path.join(ROOT, "results")
LABELS = ["human", "ai", "fusion"]
L2I = {l: i for i, l in enumerate(LABELS)}
os.makedirs(PRED, exist_ok=True)
os.makedirs(RES, exist_ok=True)


def load(split="train", cols=None):
    """train rows carry `fold` (0 = fixed validation fold) and `sub` (fixed 50k training subsample) from data/split.parquet."""
    df = pd.read_parquet(os.path.join(DATA, f"{split}.parquet"), columns=cols)
    if split == "train":
        sp = pd.read_parquet(os.path.join(DATA, "split.parquet"))
        df = df.merge(sp, on="id", how="left")
    return df


def fixed_sets():
    """The fixed sets every experiment uses: (train_sub, val, test)."""
    tr = load("train")
    val = tr[tr.fold == 0].reset_index(drop=True)
    sub = tr[tr.in_sub].reset_index(drop=True)
    return sub, val, load("test")


def y_of(df):
    return df.label.map(L2I).values


def macro_f1(y, p):
    return f1_score(y, p, average="macro")


def report(name, y, prob, extra=None):
    """Print a compact metric line and append it to results/val_metrics.jsonl."""
    pred = prob.argmax(1) if prob.ndim == 2 else prob
    f1s = f1_score(y, pred, average=None, labels=range(3))
    rec = {"name": name, "macro_f1": round(float(f1s.mean()), 4),
           **{f"f1_{l}": round(float(v), 4) for l, v in zip(LABELS, f1s)},
           "acc": round(float((pred == y).mean()), 4), "time": time.strftime("%Y-%m-%d %H:%M")}
    if extra:
        rec.update(extra)
    print(json.dumps(rec), flush=True)
    with open(os.path.join(RES, "val_metrics.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def save_probs(name, val_prob, test_prob=None):
    np.save(os.path.join(PRED, f"{name}_val.npy"), val_prob.astype(np.float32))
    if test_prob is not None:
        np.save(os.path.join(PRED, f"{name}_test.npy"), test_prob.astype(np.float32))


def write_submission(test_df, prob_or_pred, path):
    pred = prob_or_pred.argmax(1) if np.ndim(prob_or_pred) == 2 else prob_or_pred
    sub = pd.DataFrame({"id": test_df.id.values, "label": np.array(LABELS)[pred]})
    order = pd.read_csv(os.path.join(DATA, "sample_submission.csv")).id
    sub = sub.set_index("id").loc[order].reset_index()
    assert len(sub) == 25000 and sub.label.isin(LABELS).all()
    sub.to_csv(path, index=False)
    print("saved", path, sub.label.value_counts(normalize=True).round(3).to_dict())
