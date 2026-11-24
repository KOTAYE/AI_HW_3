"""Builds notebooks/01_eda.ipynb (kept as code so the notebook stays clean and reproducible).
Run:  python notebooks/build_eda.py && jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
"""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("# 01 · Exploratory data analysis\n"
   "Human / AI / fusion texts (GhostWriter subset). Figures are saved to `images/` for the report.")
code("""import sys, re, difflib
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, str(Path.cwd().parent / 'src'))
from common import DATA, ROOT, LABELS
IMG = Path(ROOT) / 'images'; IMG.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': .3})
COL = {'human': '#2a9d8f', 'ai': '#e76f51', 'fusion': '#8a5cf6'}
tr = pd.read_parquet(f'{DATA}/train.parquet').merge(pd.read_parquet(f'{DATA}/split.parquet'), on='id')
te = pd.read_parquet(f'{DATA}/test.parquet')
for d in (tr, te):
    d['n_words'] = d.text.str.split().str.len()
print(f'train {len(tr):,} rows, test {len(te):,} rows')""")

md("## 1. Class balance by language and domain")
code("""print(tr.label.value_counts(normalize=True).round(3).to_dict())
tab = pd.crosstab([tr.domain, tr.lang], tr.label)[LABELS]
tab['n'] = tab.sum(axis=1)
share = tab[LABELS].div(tab.n, axis=0).round(2)
display(pd.concat([tab, share.add_suffix('_%')], axis=1))
dom = pd.concat([tr.domain.value_counts(normalize=True).rename('train'),
                 te.domain.value_counts(normalize=True).rename('test')], axis=1).round(3)
display(dom)""")
code("""fig, ax = plt.subplots(1, 2, figsize=(12, 4))
s = pd.crosstab(tr.domain, tr.label)[LABELS]
s.div(s.sum(axis=1), axis=0).plot.barh(stacked=True, ax=ax[0], color=[COL[l] for l in LABELS], width=.8)
ax[0].set_title('Label share per domain (train)'); ax[0].set_xlabel('share'); ax[0].set_ylabel('')
dom.plot.barh(ax=ax[1], width=.8); ax[1].set_title('Domain share: train vs test'); ax[1].set_ylabel('')
plt.tight_layout(); plt.savefig(IMG / 'eda_balance.png', bbox_inches='tight'); plt.show()""")
md("**Observations.** Overall ai 45% / fusion 31% / human 25%, so macro-F1 punishes ignoring human/fusion. "
   "The balance differs a lot by domain: arXiv and Gutenberg are ~85-90% ai with very few human/fusion texts, "
   "while blogs and Bundestag are almost balanced. German exists only in Bundestag (and a few Gutenberg books), "
   "so language is confounded with domain. Test has more Gutenberg (7.4% vs 2.7%) and arXiv - the domains with "
   "the rarest human/fusion examples - so per-domain behaviour matters. Domain and language are given in test "
   "and are used as features in the level-2 model.")

md("## 2. Text lengths")
code("""display(tr.groupby('label').n_words.describe(percentiles=[.1, .5, .9]).round(0))
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for l in LABELS:
    ax[0].hist(np.log10(tr.n_words[tr.label == l]), bins=60, alpha=.5, density=True, label=l, color=COL[l])
ax[0].set_xlabel('log10(words)'); ax[0].legend(); ax[0].set_title('Length by label')
med = tr.groupby(['domain', 'label']).n_words.median().unstack()[LABELS]
med.plot.barh(ax=ax[1], color=[COL[l] for l in LABELS], width=.8); ax[1].set_title('Median words per domain'); ax[1].set_ylabel('')
plt.tight_layout(); plt.savefig(IMG / 'eda_lengths.png', bbox_inches='tight'); plt.show()""")
md("AI texts have a narrow length distribution (LLM outputs rarely exceed ~1000 words), while human and fusion "
   "texts have long tails up to 3000 words. Length alone is a weak signal, but very long texts are almost never ai.")

md("## 3. Tokenisation: how much of a text fits into an encoder")
code("""from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('microsoft/mdeberta-v3-base')
smp = tr.sample(3000, random_state=0)
smp['n_tok'] = [len(x) for x in tok(smp.text.tolist(), add_special_tokens=False)['input_ids']]
smp['tok_per_word'] = smp.n_tok / smp.n_words
display(smp.groupby('lang')[['n_tok', 'tok_per_word']].median().round(2))
print('share of texts longer than 510 tokens:', round((smp.n_tok > 510).mean(), 3))
print('share of tokens beyond the first 510:', round((smp.n_tok - 510).clip(lower=0).sum() / smp.n_tok.sum(), 3))""")
md("More than half of the texts do not fit into 512 tokens and about half of all tokens are beyond the first "
   "window, so plain truncation throws a lot away. This motivates random-window training and averaging over "
   "windows at inference (section 'long-text handling' in the encoder notebook).")

md("## 4. Examples of each class")
code("""def show(t, n=350):
    return t[:n].replace('\\n', ' / ') + ' ...'
for l in LABELS:
    for d in ['cnn_news', 'bundestag']:
        print(f'[{l} | {d}]', show(tr[(tr.label == l) & (tr.domain == d)].text.iloc[3]), '\\n')""")

md("## 5. What makes fusion different\n"
   "The data card says that a human text and its AI / fusion versions stay on the same side of the split, but "
   "no group id is given. `src/groups.py` reconstructs groups: word TF-IDF inside each domain, top-10 cosine "
   "neighbours, union-find over pairs with cosine > 0.4.")
code("""g = tr.groupby('group').label.agg(lambda s: ''.join(sorted(set(x[0] for x in s))))
print('group label-sets:', g.value_counts().to_dict())
print('group sizes:', tr.group.value_counts().value_counts().sort_index().head(6).to_dict())""")
md("Most fusion texts form a pair with exactly one human text (`fh` groups), while ai texts are almost always "
   "alone: an AI version is written from scratch and shares little wording with its source, whereas fusion "
   "keeps a large part of the human text. Below we align human-fusion pairs to measure how much is kept.")
code("""SENT = re.compile(r'(?<=[.!?])\\s+')
pairs = []
for gid, d in tr[tr.group.isin(g[g == 'fh'].index)].groupby('group'):
    if len(d) == 2:
        pairs.append((d[d.label == 'human'].text.iloc[0], d[d.label == 'fusion'].text.iloc[0]))
    if len(pairs) == 1500: break
rows = []
for h, f in pairs:
    hs, fs = set(SENT.split(h)), SENT.split(f)
    kept = np.mean([s in hs for s in fs])
    sm = difflib.SequenceMatcher(None, h.split(), f.split(), autojunk=False)
    rows.append({'kept_sent': kept, 'word_ratio': sm.ratio(), 'len_ratio': len(f.split()) / len(h.split())})
pr = pd.DataFrame(rows)
display(pr.describe(percentiles=[.1, .5, .9]).round(2))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].hist(pr.kept_sent, bins=40, color=COL['fusion']); ax[0].set_title('Fusion: share of sentences copied verbatim from human')
ax[1].hist(pr.word_ratio, bins=40, color=COL['fusion']); ax[1].set_title('Fusion vs its human source: word-level similarity')
plt.tight_layout(); plt.savefig(IMG / 'eda_fusion.png', bbox_inches='tight'); plt.show()""")
code("""h, f = pairs[7]
sm = difflib.SequenceMatcher(None, h.split(), f.split(), autojunk=False)
out = []
for op, i1, i2, j1, j2 in sm.get_opcodes()[:40]:
    w = ' '.join(f.split()[j1:j2])
    out.append(w if op == 'equal' else f'[[{w}]]')
print('fusion text, LLM-changed spans in [[...]]:\\n', ' '.join(out)[:1500])""")
md("**Observation.** A fusion text keeps a part of the human sentences verbatim and replaces / inserts others, "
   "and the changed spans come in contiguous blocks. A fusion text is therefore *locally* human and *locally* "
   "AI, which is why fusion is the hardest class: a short window can look entirely human. The leaked prompt answers (example above, counts in section 6) "
   "(\"reconstructed missing section ... Word Count\") show how fusion was produced: the middle of a human text was "
   "removed and an LLM wrote the gap. Two consequences for "
   "modelling: (1) looking at the whole text (all windows) should help, and per-window disagreement is itself a "
   "fusion signal; (2) features that measure inconsistency inside a text (spread of per-segment perplexity / "
   "style) should separate fusion from both pure classes.")

md("## 6. A few style statistics by class")
code("""st = pd.read_parquet(Path(ROOT) / 'preds' / 'stylo_sub.parquet').merge(tr[['id', 'label']], on='id')
cols = ['sent_mean', 'sent_cv', 'wlen_mean', 'ttr', 'p_dash_em', 'p_quote_curly', 'space_before_punct',
        'w_furthermore', 'w_crucial', 'blk_wlen_std']
display(st.groupby('label')[cols].mean().loc[LABELS].T.round(3))   # means: most rare-token rates have median 0""")
md("Leftover LLM meta-text (share of texts matching a pattern). Fusion was produced by asking an LLM to fill a "
   "removed middle part of a human text, and sometimes the prompt answer leaks into the text:")
code("""pats = {'missing section/part': r'(?i)missing (?:section|part|passage)', 'reconstruct': r'(?i)reconstruct',
        'word count': r'(?i)word count', 'markdown heading': r'(?m)^#+ '}
display(pd.DataFrame({k: tr.text.str.contains(v).groupby(tr.label).mean() for k, v in pats.items()}).loc[LABELS].T.round(4))""")
md("AI texts have more regular sentence lengths (lower coefficient of variation), longer words and more typical "
   "LLM vocabulary; fusion sits between human and ai on most statistics, consistent with a mixture, and has the "
   "largest spread of word length between 60-word blocks (`blk_wlen_std`) - the inconsistency we expected. "
   "Leaked meta-text appears in only ~5% of fusion texts, so it is a useful n-gram cue but not the main signal.")

md("## 7. Validation design")
code("""v = tr[tr.fold == 0]
print('val rows:', len(v), '| label share', v.label.value_counts(normalize=True).round(3).to_dict())
print('groups shared between val and the rest:', len(set(v.group) & set(tr[tr.fold != 0].group)))""")
md("Fold 0 of a 10-fold StratifiedGroupKFold (groups above, strata = domain x label) is the fixed validation "
   "set for all comparisons; transformers and the models compared with them use a fixed 50k subsample of the "
   "remaining folds (whole groups). A random split would put a human text in train and its fusion copy in val, "
   "and the fusion detector would be rewarded for memorising the human text.")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python312-global", "display_name": "Python 3.12 (global)", "language": "python"}
nbf.write(nb, "notebooks/01_eda.ipynb")
print("written notebooks/01_eda.ipynb")
