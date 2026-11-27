import os
"""Validate our DTE-protocol data pipeline: kNN (k=5, largest) must match DTE's published kNN column."""
import sys, json, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anoddae_repro.data import benchmark_datasets, family_members, make_split, short_name

def knn_file(f, setting, seed):
    s = make_split(f, setting, seed, protocol=PROTO)
    nn = NearestNeighbors(n_neighbors=5, n_jobs=1).fit(s.x_train)
    d, _ = nn.kneighbors(s.x_test)
    sc = d[:, -1]
    return f, roc_auc_score(s.y_test, sc) * 100, average_precision_score(s.y_test, sc) * 100

setting = sys.argv[1]; seed = int(sys.argv[2]); PROTO = sys.argv[3] if len(sys.argv) > 3 else "dte"
jobs = [(ds, f) for ds in benchmark_datasets() for f in family_members(ds)]
res = Parallel(n_jobs=4)(delayed(knn_file)(f, setting, seed) for ds, f in jobs)
per_file = {f: (r, p) for f, r, p in res}
rows = []
for ds in benchmark_datasets():
    fs = family_members(ds)
    rows.append(dict(dataset=short_name(ds), roc=np.mean([per_file[f][0] for f in fs]), pr=np.mean([per_file[f][1] for f in fs]),
                     roc_first=per_file[fs[0]][0], pr_first=per_file[fs[0]][1]))
df = pd.DataFrame(rows).set_index('dataset')
ref_roc = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reference', f'dte_{setting}_ROC_mean.csv'), index_col=0)['kNN']
ref_pr = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reference', f'dte_{setting}_PR_mean.csv'), index_col=0)['kNN']
df['dte_roc'] = ref_roc.reindex(df.index); df['dte_pr'] = ref_pr.reindex(df.index)
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 100)
print(df.round(2))
print('MEAN ours(family avg) ROC %.2f PR %.2f | ours(first file) ROC %.2f PR %.2f | DTE ROC %.2f PR %.2f' % (
    df.roc.mean(), df.pr.mean(), df.roc_first.mean(), df.pr_first.mean(), df.dte_roc.mean(), df.dte_pr.mean()))
print('max |diff| ROC (family avg):', (df.roc - df.dte_roc).abs().max().round(2), ' median:', (df.roc - df.dte_roc).abs().median().round(3))
df.to_csv(os.path.join(os.environ.get('ANODDAE_RESULTS', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')), f'knn_validation_{setting}_seed{seed}' + ('' if PROTO == 'dte' else '_' + PROTO) + '.csv'))
