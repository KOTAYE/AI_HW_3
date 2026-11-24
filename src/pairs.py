"""(Optional, transductive) near-duplicate features *within* an evaluation set.

Because a human text and its fusion version are always on the same side of the split, a text that has a
near-duplicate in the same set is very likely human or fusion. For every text in val (resp. test) we compute
its maximum TF-IDF cosine similarity to another text of the same set and domain. The vectorizer is fit on the
training subsample only (no fitting on val/test rows, no labels are used), but the feature does use the
other unlabeled rows of the evaluation set, so it is reported separately and is NOT part of the main model
unless we decide so explicitly.

    python src/pairs.py   ->  preds/pairs_{val,test}.parquet
"""
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from common import PRED, fixed_sets


def max_sim(X, dom, chunk=2000):
    out = np.zeros(X.shape[0], np.float32)
    second = np.zeros(X.shape[0], np.float32)
    for d in np.unique(dom):
        idx = np.where(dom == d)[0]
        Xd = X[idx]
        for s in range(0, len(idx), chunk):
            S = (Xd[s:s + chunk] @ Xd.T).toarray()
            S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -1
            S.sort(1)
            out[idx[s:s + chunk]] = S[:, -1]
            second[idx[s:s + chunk]] = S[:, -2] if S.shape[1] > 1 else 0
    return out, second


if __name__ == "__main__":
    sub, val, test = fixed_sets()
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.5, dtype=np.float32).fit(sub.text)
    for name, df in [("val", val), ("test", test)]:
        m1, m2 = max_sim(vec.transform(df.text), df.domain.values)
        pd.DataFrame({"id": df.id.values, "pair_sim1": m1, "pair_sim2": m2}).to_parquet(os.path.join(PRED, f"pairs_{name}.parquet"))
        if "label" in df:
            print(pd.DataFrame({"l": df.label, "has_pair": m1 > 0.4}).groupby("l").has_pair.mean().round(3).to_dict())
        else:
            print(name, "share with pair:", round(float((m1 > 0.4).mean()), 3))
