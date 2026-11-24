"""Builds notebooks/02_baseline.ipynb from cached results (models are trained by src/baseline.py).
Run:  python notebooks/build_baseline.py && jupyter nbconvert --to notebook --execute --inplace notebooks/02_baseline.ipynb
"""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("# 02 · Classical baseline: TF-IDF + linear models\n"
   "Training: `python src/baseline.py search` (grid) and `python src/baseline.py final --cfg ... [--full]`.\n\n"
   "All vectorizers live inside an sklearn `Pipeline` and are fit on training rows only; every model is "
   "trained on the fixed 50k subsample and scored on the fixed validation fold (27,230 texts, grouped split).")
code("""import sys, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, str(Path.cwd().parent / 'src'))
from common import ROOT, PRED, RES, LABELS, fixed_sets, y_of
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
IMG = Path(ROOT) / 'images'
plt.rcParams.update({'figure.dpi': 110})
M = pd.read_json(Path(RES) / 'val_metrics.jsonl', lines=True).drop_duplicates('name', keep='last')""")

md("## 1. Parameter search\nWord n-grams keep case and punctuation (style signals), char n-grams use `char_wb` 2-5. "
   "Models: Complement NB, logistic regression, linear SVM (calibrated with 3-fold `CalibratedClassifierCV` to get probabilities).")
code("""g = M[M.name.str.startswith('tfidf_') & (M.train == 'sub50k')][['name', 'macro_f1', 'f1_human', 'f1_ai', 'f1_fusion', 'sec']]
g.sort_values('macro_f1', ascending=False).reset_index(drop=True)""")
md("Logistic regression and linear SVM are within ~0.5 pt; adding char n-grams to words gives a small gain at "
   "~6x the cost. Naive Bayes is far behind (its independence assumption over-counts correlated n-grams). "
   "We keep **word 1-3-grams + LR (C=16)** as the reference baseline: almost the best score, simplest model, and "
   "its coefficients are directly interpretable (see bonus section).")

md("## 2. Per-class metrics and confusion matrix (reference baseline, trained on 50k)")
code("""sub, val, test = fixed_sets()
yv = y_of(val)
pv = np.load(Path(PRED) / 'baseline_val.npy')
print(classification_report(yv, pv.argmax(1), target_names=LABELS, digits=4))
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, norm, t in [(ax[0], None, 'counts'), (ax[1], 'true', 'row-normalised')]:
    ConfusionMatrixDisplay(confusion_matrix(yv, pv.argmax(1), normalize=norm), display_labels=LABELS).plot(
        ax=a, colorbar=False, values_format='.2f' if norm else 'd'); a.set_title(t)
plt.tight_layout(); plt.savefig(IMG / 'baseline_cm.png', bbox_inches='tight'); plt.show()""")
md("AI is almost solved (F1 ~0.985). Nearly all errors are human <-> fusion confusions: a fusion text is "
   "mostly copied human text, so bag-of-n-grams sees mostly human evidence.")

md("## 3. Per-domain macro-F1")
code("""val['pred'] = np.array(LABELS)[pv.argmax(1)]
from sklearn.metrics import f1_score
dom = val.groupby('domain').apply(lambda d: pd.Series({'n': len(d), 'macro_f1': f1_score(d.label, d.pred, average='macro'),
                                                       'f1_fusion': f1_score(d.label == 'fusion', d.pred == 'fusion')}), include_groups=False)
dom.round(3)""")

md("## 4. More data\nThe same pipeline fit on all non-validation folds (~245k texts) instead of 50k:")
code("""M[M.name.str.contains('tfidf_featword_modellr_C16_wng1-3')][['name', 'train', 'macro_f1', 'f1_human', 'f1_ai', 'f1_fusion', 'sec']]""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python312-global", "display_name": "Python 3.12 (global)", "language": "python"}
nbf.write(nb, "notebooks/02_baseline.ipynb")
print("written notebooks/02_baseline.ipynb")
