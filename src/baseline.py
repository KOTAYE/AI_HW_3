"""Classical baseline: TF-IDF word / char n-grams + linear models (sklearn Pipelines, vectorizers fit on train only).

    python src/baseline.py search            # small grid on the fixed 50k subsample, scored on the fixed val fold
    python src/baseline.py final [--full]    # best config -> val/test probabilities in preds/ (--full: all non-val folds)
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from common import LABELS, RES, fixed_sets, load, report, save_probs, y_of


def word_vec(ngram=(1, 2), max_features=300_000):
    # keep case: capitalisation and function-word use are style signals
    return TfidfVectorizer(analyzer="word", ngram_range=ngram, lowercase=False, sublinear_tf=True, min_df=3,
                           max_df=0.9, max_features=max_features, token_pattern=r"(?u)\b\w+\b|[^\w\s]", dtype=np.float32)


def char_vec(ngram=(2, 5), max_features=500_000):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, lowercase=False, sublinear_tf=True, min_df=5,
                           max_features=max_features, dtype=np.float32)


def make(cfg):
    feats = []
    if cfg["feat"] in ("word", "word+char"):
        feats.append(("word", word_vec(tuple(cfg.get("wng", (1, 2))))))
    if cfg["feat"] in ("char", "word+char"):
        feats.append(("char", char_vec(tuple(cfg.get("cng", (2, 5))))))
    vec = FeatureUnion(feats) if len(feats) > 1 else feats[0][1]
    m = cfg["model"]
    if m == "lr":
        clf = LogisticRegression(C=cfg["C"], max_iter=1000, class_weight=cfg.get("cw"))
    elif m == "svm":
        clf = CalibratedClassifierCV(LinearSVC(C=cfg["C"], class_weight=cfg.get("cw")), cv=3)
    elif m == "nb":
        clf = ComplementNB(alpha=cfg["C"])
    return Pipeline([("vec", vec), ("clf", clf)])


GRID = [
    {"feat": "word", "model": "nb", "C": 0.1},
    {"feat": "word", "model": "lr", "C": 4},
    {"feat": "word", "model": "lr", "C": 16},
    {"feat": "word", "model": "lr", "C": 16, "wng": (1, 3)},
    {"feat": "word", "model": "svm", "C": 0.5},
    {"feat": "char", "model": "lr", "C": 16},
    {"feat": "char", "model": "svm", "C": 0.5},
    {"feat": "word+char", "model": "svm", "C": 0.25},
    {"feat": "word+char", "model": "svm", "C": 1},
    {"feat": "word+char", "model": "svm", "C": 0.5, "cw": "balanced"},
]


def tag(cfg):
    return "tfidf_" + "_".join(f"{k}{'-'.join(map(str, v)) if isinstance(v, (tuple, list)) else v}" for k, v in cfg.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["search", "final"])
    ap.add_argument("--cfg", default=None, help="json config for final")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    sub, val, test = fixed_sets()
    yv = y_of(val)
    if a.mode == "search":
        for cfg in GRID:
            t = time.time()
            pipe = make(cfg).fit(sub.text, y_of(sub))
            report(tag(cfg), yv, pipe.predict_proba(val.text), {"train": "sub50k", "sec": int(time.time() - t)})
        return
    cfg = json.loads(a.cfg)
    tr = load("train")
    tr = tr[tr.fold != 0] if a.full else sub
    t = time.time()
    pipe = make(cfg).fit(tr.text, y_of(tr))
    pv, pt = pipe.predict_proba(val.text), pipe.predict_proba(test.text)
    name = tag(cfg) + ("_full" if a.full else "")
    report(name, yv, pv, {"train": "full" if a.full else "sub50k", "sec": int(time.time() - t)})
    save_probs("baseline_full" if a.full else "baseline", pv, pt)
    print(classification_report(yv, pv.argmax(1), target_names=LABELS, digits=4))
    cm = confusion_matrix(yv, pv.argmax(1))
    np.save(os.path.join(RES, f"cm_{'baseline_full' if a.full else 'baseline'}.npy"), cm)
    print(cm)


if __name__ == "__main__":
    main()
