"""Builds notebooks/03_encoder.ipynb from cached predictions (training: src/train_encoder.py, see scripts/queue_local.sh).
Run:  python notebooks/build_encoder.py && jupyter nbconvert --to notebook --execute --inplace notebooks/03_encoder.ipynb
"""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("# 03 · Fine-tuned multilingual encoder\n"
   "`python src/train_encoder.py --model microsoft/mdeberta-v3-base --name mdeberta --epochs 2 --bs 4 --accum 4`\n\n"
   "* Backbone: mDeBERTa-v3-base (multilingual, 100+ languages incl. German), 3-way classification head.\n"
   "* Data: the same fixed 50k training subsample and the same validation fold as the baseline.\n"
   "* Optimisation: AdamW, lr 2e-5, linear warm-up 6% + decay, effective batch 16, 2 epochs, bf16 autocast. "
   "The 250k-token embedding matrix is frozen (it is 2/3 of the weights; freezing saves ~3 GB so the model trains "
   "on an 8 GB GPU).\n"
   "* **Long texts.** 68% of texts are longer than 510 tokens and half of all tokens lie beyond the first window. "
   "Training: each epoch draws a random 510-token window from every text (a cheap augmentation that exposes the "
   "model to all parts of a document). Inference: overlapping windows (510 tokens, stride 384, at most 8 windows "
   "incl. the tail); per-window logits are cached so aggregations can be compared offline.")
code("""import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, str(Path.cwd().parent / 'src'))
from common import ROOT, PRED, RES, LABELS, fixed_sets, y_of
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
IMG = Path(ROOT) / 'images'
plt.rcParams.update({'figure.dpi': 110})
sub, val, test = fixed_sets(); yv = y_of(val)

def chunk_probs(name, split='val'):
    z = np.load(Path(PRED) / f'{name}_chunks_{split}.npz'); lg, doc = z['logits'], z['doc']
    p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return p, doc

def aggregations(name, n):
    p, doc = chunk_probs(name)
    df = pd.DataFrame(p, columns=LABELS); df['doc'] = doc
    g = df.groupby('doc')[LABELS]
    first, last, mean = g.first().values, g.last().values, g.mean().values
    lmean = np.exp(np.log(df[LABELS].clip(1e-6)).groupby(df.doc).mean().values)
    return {'first window (truncation)': first, 'head + tail': (first + last) / 2,
            'mean over windows': mean, 'geometric mean': lmean}

def f1row(p):
    f = f1_score(yv, p.argmax(1), average=None)
    return {'macro_f1': f.mean(), **{f'f1_{l}': v for l, v in zip(LABELS, f)}}""")

md("## 1. Long-text handling at inference")
code("""rows = []
for name in ['mdeberta', 'xlmr', 'xlmr_head', 'mdeberta_150k', 'xlmr_150k', 'xlmr_full']:
    if not (Path(PRED) / f'{name}_chunks_val.npz').exists(): continue
    for agg, p in aggregations(name, len(val)).items():
        rows.append({'model': name, 'aggregation': agg, **f1row(p)})
agg_tab = pd.DataFrame(rows).round(4); agg_tab""")

md("## 2. Encoder vs baseline on the same split")
code("""pb = np.load(Path(PRED) / 'baseline_val.npy')
pe = aggregations('mdeberta', len(val))['mean over windows']
cmp = pd.DataFrame([{'model': 'TF-IDF word 1-3 + LR (50k)', **f1row(pb)},
                    {'model': 'mDeBERTa-v3-base (50k, mean over windows)', **f1row(pe)}]).round(4)
display(cmp)
print(classification_report(yv, pe.argmax(1), target_names=LABELS, digits=4))
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, p, t in [(ax[0], pb, 'TF-IDF + LR'), (ax[1], pe, 'mDeBERTa')]:
    ConfusionMatrixDisplay(confusion_matrix(yv, p.argmax(1), normalize='true'), display_labels=LABELS).plot(ax=a, colorbar=False, values_format='.3f'); a.set_title(t)
plt.tight_layout(); plt.savefig(IMG / 'encoder_vs_baseline_cm.png', bbox_inches='tight'); plt.show()""")
code("""val['enc'] = pe.argmax(1); val['tfidf'] = pb.argmax(1)
dom = val.groupby('domain').apply(lambda d: pd.Series({'n': len(d),
        'tfidf': f1_score(y_of(d), d.tfidf, average='macro'), 'mdeberta': f1_score(y_of(d), d.enc, average='macro')}), include_groups=False)
dom.round(3)""")

md("## 3. Errors: does the encoder agree with the baseline?")
code("""both = pd.crosstab(val.enc == yv, val.tfidf == yv, rownames=['encoder correct'], colnames=['tfidf correct'])
display(both)
print('oracle (either correct):', round(((val.enc == yv) | (val.tfidf == yv)).mean(), 4))""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python312-global", "display_name": "Python 3.12 (global)", "language": "python"}
nbf.write(nb, "notebooks/03_encoder.ipynb")
print("written notebooks/03_encoder.ipynb")
