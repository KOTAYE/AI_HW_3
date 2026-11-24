"""Fixed validation split: StratifiedGroupKFold over reconstructed source groups (src/groups.py),
stratified by domain x label. Fold 0 (~1/10 of train) is the validation set for ALL comparisons.
`sub` marks a fixed stratified 50k subsample of the remaining folds used to train transformers
and the models compared with them.

    python src/split.py  ->  data/split.parquet (id, group, fold, in_sub)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import DATA

tr = pd.read_parquet(f"{DATA}/train.parquet", columns=["id", "domain", "label"])
g = pd.read_parquet(f"{DATA}/groups.parquet")
tr = tr.merge(g, on="id")
strat = (tr.domain + "_" + tr.label).values
fold = np.full(len(tr), -1)
for k, (_, va) in enumerate(StratifiedGroupKFold(10, shuffle=True, random_state=42).split(tr, strat, tr.group)):
    fold[va] = k
tr["fold"] = fold
rest = tr[tr.fold != 0]
# 50k subsample: sample whole groups so the subsample stays leak-free w.r.t. itself, stratify roughly by domain x label
rng = np.random.default_rng(0)
gids = rest.group.unique()
rng.shuffle(gids)
cum = rest.groupby("group").size().loc[gids].cumsum()
keep = set(cum.index[cum <= 50000])
tr["in_sub"] = tr.group.isin(keep) & (tr.fold != 0)
print(tr.groupby("fold").size().to_dict())
print("val label share:", tr[tr.fold == 0].label.value_counts(normalize=True).round(3).to_dict())
print("sub:", int(tr["in_sub"].sum()), tr[tr["in_sub"]].label.value_counts(normalize=True).round(3).to_dict())
tr[["id", "group", "fold", "in_sub"]].to_parquet(f"{DATA}/split.parquet")
