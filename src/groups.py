"""Reconstruct source groups (a human text + its AI / fusion versions) in train.csv.

The data card says a human text and its rewrites are always on the same side of the train/test split,
but no group id is given. Inside each domain we embed texts with word TF-IDF (fit on train only), take
the top-K cosine neighbours of every text and join pairs above a threshold with union-find.
The connected components are the groups used for GroupKFold-style validation.

    python src/groups.py [--thr 0.4] [--k 10]   ->  data/groups.parquet (id, group)
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def find(p, x):
    while p[x] != x:
        p[x] = p[p[x]]
        x = p[x]
    return x


def domain_edges(texts, k, chunk=2000):
    X = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.5, dtype=np.float32).fit_transform(texts)
    n = X.shape[0]
    nbr = np.zeros((n, k), np.int64)
    sim = np.zeros((n, k), np.float32)
    for s in range(0, n, chunk):
        S = (X[s:s + chunk] @ X.T).toarray()
        S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -1
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        nbr[s:s + chunk] = idx
        sim[s:s + chunk] = np.take_along_axis(S, idx, 1)
    return nbr, sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, nargs="+", default=[0.4])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="data/groups.parquet")
    a = ap.parse_args()
    tr = pd.read_parquet("data/train.parquet", columns=["id", "text", "domain", "label"])
    res = {t: np.zeros(len(tr), np.int64) for t in a.thr}
    for dom, d in tr.groupby("domain"):
        rows = d.index.values
        nbr, sim = domain_edges(d.text.values, a.k)
        for t in a.thr:
            p = np.arange(len(d))
            for i, j in zip(*np.where(sim > t)):
                ri, rj = find(p, i), find(p, nbr[i, j])
                if ri != rj:
                    p[ri] = rj
            roots = np.array([find(p, i) for i in range(len(d))])
            res[t][rows] = roots + rows.min() * 0 + hash(dom) % 10**6 * 10**6
        print(dom, "done", flush=True)
    for t in a.thr:
        g = pd.Series(res[t]).factorize()[0]
        sz = pd.Series(g).value_counts()
        tr["g"] = g
        comp = tr.groupby("g").label.agg(lambda s: "".join(sorted(set(x[0] for x in s))))
        print(f"thr={t}: groups={len(sz)}  max={sz.max()}  >50={int((sz > 50).sum())}  rows_in_>50={int(sz[sz > 50].sum())}")
        print("  size dist:", sz.value_counts().sort_index().head(8).to_dict())
        print("  label sets:", comp.value_counts().to_dict())
    t = a.thr[0]
    pd.DataFrame({"id": tr.id, "group": pd.Series(res[t]).factorize()[0]}).to_parquet(a.out)


if __name__ == "__main__":
    main()
