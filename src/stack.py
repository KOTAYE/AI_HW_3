"""Experiments on top of cached model outputs.

1) `feat`  : LightGBM on hand-crafted features only (perplexity and/or stylometry), trained on the fixed 50k
             subsample and scored on the fixed val fold -> same protocol as baseline / encoder.
2) `stack` : level-2 model on the val fold. Inputs are val/test outputs of level-1 models that never saw val
             (TF-IDF, encoders, perplexity + stylometry features). Scored with 5-fold StratifiedGroupKFold
             *inside* val (groups from src/groups.py), so the reported macro-F1 is out-of-fold. The final model
             is refit on the whole val fold and applied to test.
3) Class-bias tuning for macro-F1: additive biases on log-probabilities, chosen on the OOF predictions.

    python src/stack.py feat --sets ppl_qwen15 stylo
    python src/stack.py stack --enc mdeberta --tfidf baseline_full --feats ppl_qwen15 stylo --name stack_all --submit
"""
import argparse
import itertools
import os
import re

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import LABELS, PRED, ROOT, fixed_sets, macro_f1, report, save_probs, write_submission, y_of

LGB = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63, min_child_samples=40,
           feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, n_jobs=10)


def clean(X):
    """LightGBM rejects JSON-special / non-ASCII characters in feature names."""
    X.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", c) for c in X.columns]
    return X


def load_feats(names, split, ids):
    out = []
    for n in names:
        f = pd.read_parquet(os.path.join(PRED, f"{n}_{split}.parquet")).set_index("id").loc[ids]
        out.append(f.add_prefix(n + ":"))
    return pd.concat(out, axis=1).reset_index(drop=True)


def meta(df):
    m = pd.DataFrame({"is_de": (df.lang == "de-DE").astype(int).values})
    for d in sorted(df.domain.unique()):
        m[f"dom_{d}"] = (df.domain == d).astype(int).values
    return m


def enc_feats(name, split, n):
    """Aggregations of the per-window logits of an encoder: mean prob, first window, max/min/std over windows."""
    z = np.load(os.path.join(PRED, f"{name}_chunks_{split}.npz"))
    lg, doc = z["logits"], z["doc"]
    p = np.exp(lg - lg.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    df = pd.DataFrame(p, columns=LABELS)
    df["doc"] = doc
    g = df.groupby("doc")[LABELS]
    parts = {"mean": g.mean(), "max": g.max(), "min": g.min(), "std": g.std().fillna(0), "first": g.first(),
             "last": g.last()}
    out = pd.concat([v.add_prefix(f"{name}:{k}_") for k, v in parts.items()], axis=1)
    out[f"{name}:n_win"] = g.size()
    return out.reindex(range(n)).reset_index(drop=True)


def tune_bias(y, logp, grid=np.linspace(-1.5, 1.5, 31)):
    """Additive log-prob biases for ai and fusion (human fixed at 0) that maximise macro-F1."""
    best = (macro_f1(y, logp.argmax(1)), 0.0, 0.0)
    for b1, b2 in itertools.product(grid, grid):
        s = macro_f1(y, (logp + np.array([0, b1, b2])).argmax(1))
        if s > best[0]:
            best = (s, b1, b2)
    return np.array([0, best[1], best[2]]), best[0]


def cv_lgb(X, y, groups, strat, Xt, rounds=2000, seed=0):
    oof = np.zeros((len(X), 3))
    pt = np.zeros((len(Xt), 3))
    its = []
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, strat, groups):
        m = lgb.train({**LGB, "seed": seed}, lgb.Dataset(X.iloc[tr], y[tr]), rounds, valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        its.append(m.best_iteration)
    full = lgb.train({**LGB, "seed": seed}, lgb.Dataset(X, y), int(np.mean(its) * 1.1))
    pt = full.predict(Xt)
    return oof, pt, full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["feat", "stack"])
    ap.add_argument("--sets", nargs="+", default=[], help="feature tables for `feat`")
    ap.add_argument("--enc", nargs="*", default=[])
    ap.add_argument("--tfidf", nargs="*", default=[])
    ap.add_argument("--feats", nargs="*", default=[])
    ap.add_argument("--no-meta", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    sub, val, test = fixed_sets()
    yv = y_of(val)

    if a.mode == "feat":
        Xs = pd.concat([load_feats(a.sets, "sub", sub.id), meta(sub)], axis=1)
        Xv = pd.concat([load_feats(a.sets, "val", val.id), meta(val)], axis=1)
        Xt = pd.concat([load_feats(a.sets, "test", test.id), meta(test)], axis=1)
        Xs, Xv, Xt = clean(Xs), clean(Xv), clean(Xt)
        ys = y_of(sub)
        # early stopping on a group-held-out part of the subsample, never on val
        gk = np.isin(sub.group, np.unique(sub.group)[::10])
        m = lgb.train(LGB, lgb.Dataset(Xs[~gk], ys[~gk]), 3000, valid_sets=[lgb.Dataset(Xs[gk], ys[gk])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        pv, pt = m.predict(Xv, num_iteration=m.best_iteration), m.predict(Xt, num_iteration=m.best_iteration)
        name = a.name or "lgb_" + "+".join(a.sets)
        report(name, yv, pv, {"train": "sub50k", "n_feat": Xs.shape[1]})
        save_probs(name, pv, pt)
        imp = pd.Series(m.feature_importance("gain"), Xs.columns).sort_values(ascending=False)
        imp.to_csv(os.path.join(ROOT, "results", f"imp_{name}.csv"))
        print(imp.head(15).round(0).to_dict())
        return

    blocks_v, blocks_t = [], []
    for e in a.enc:
        blocks_v.append(enc_feats(e, "val", len(val)))
        blocks_t.append(enc_feats(e, "test", len(test)))
    for t in a.tfidf:
        blocks_v.append(pd.DataFrame(np.load(os.path.join(PRED, f"{t}_val.npy")), columns=[f"{t}:{l}" for l in LABELS]))
        blocks_t.append(pd.DataFrame(np.load(os.path.join(PRED, f"{t}_test.npy")), columns=[f"{t}:{l}" for l in LABELS]))
    if a.feats:
        blocks_v.append(load_feats(a.feats, "val", val.id))
        blocks_t.append(load_feats(a.feats, "test", test.id))
    if not a.no_meta:
        blocks_v.append(meta(val))
        blocks_t.append(meta(test))
    Xv, Xt = clean(pd.concat(blocks_v, axis=1)), clean(pd.concat(blocks_t, axis=1))
    strat = (val.domain + "_" + val.label).values
    oof, pt, full = cv_lgb(Xv, yv, val.group.values, strat, Xt)
    name = a.name or "stack_" + "+".join(a.enc + a.tfidf + a.feats)
    report(name, yv, oof, {"train": "val-5fold-oof", "n_feat": Xv.shape[1]})
    # bias tuning is itself fitted on OOF -> estimate its gain honestly with a 2-fold split by group
    logp = np.log(oof + 1e-9)
    gsel = np.isin(val.group.values, np.unique(val.group.values)[::2])
    b1, _ = tune_bias(yv[gsel], logp[gsel])
    b2, _ = tune_bias(yv[~gsel], logp[~gsel])
    honest = np.empty(len(yv), int)
    honest[~gsel] = (logp[~gsel] + b1).argmax(1)
    honest[gsel] = (logp[gsel] + b2).argmax(1)
    report(name + "_bias_crossfit", yv, honest, {"train": "val-5fold-oof", "bias_half1": b1.round(2).tolist(),
                                                 "bias_half2": b2.round(2).tolist()})
    bias, _ = tune_bias(yv, logp)
    save_probs(name, oof, pt)
    imp = pd.Series(full.feature_importance("gain"), Xv.columns).sort_values(ascending=False)
    imp.to_csv(os.path.join(ROOT, "results", f"imp_{name}.csv"))
    print(imp.head(20).round(0).to_dict())
    if a.submit:
        os.makedirs(os.path.join(ROOT, "submissions"), exist_ok=True)
        write_submission(test, pt, os.path.join(ROOT, "submissions", f"{name}.csv"))
        write_submission(test, np.log(pt + 1e-9) + bias, os.path.join(ROOT, "submissions", f"{name}_bias.csv"))
        print("bias", bias)


if __name__ == "__main__":
    main()
