"""Print text outputs of a notebook (for quick checks)."""
import sys, nbformat
nb = nbformat.read(sys.argv[1], 4)
for c in nb.cells:
    if c.cell_type == 'code':
        for o in c.outputs:
            t = o.get('text') or o.get('data', {}).get('text/plain', '') or ''.join(o.get('traceback', []))[-800:]
            if t: print(t[:1500]); print('---')
