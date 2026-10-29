"""Recompute every number for the benchmark-led manuscript from results.csv (OASIS-Net = a-priori design, v1)."""
import sys, json, numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp
RES = sys.argv[1]; OUT = sys.argv[2]
R = pd.read_csv(f'{RES}/results.csv')
NAMES = {'gbm': 'Features+GBM', 'wdcnn': 'Raw+WDCNN', 'oasis': 'OASIS-Net', 'oasis_noadv': 'OASIS w/o adversary',
         'oasis_noorder': 'OASIS w/o order branch', 'oasis_noimg': 'OASIS w/o scalogram branch'}
def cfg(r):
    if r.model in NAMES: return NAMES[r.model]
    if r.model.startswith('oasis2'): return 'v2:' + r.model
    return r['transform'] + '+' + {'resnet18': 'ResNet-18', 'swin': 'Swin-T', 'hybrid': 'Hybrid'}[r.model]
R['config'] = R.apply(cfg, axis=1)
clean = R[R.snr_db.isna()].copy()
clean['col'] = clean.dataset + ' ' + np.where(clean.test.str.startswith('X-'), 'X', clean.protocol.replace({'L3X': 'L3'}))
COLS = ['CWRU L0', 'CWRU L1', 'CWRU L2', 'CWRU L3', 'PU L0', 'PU L1', 'PU L3', 'PU X', 'PU A2R']
LF = ['CWRU L3', 'PU L3', 'PU X', 'PU A2R']
MAIN = ['Features+GBM', 'Raw+WDCNN', 'STFT+ResNet-18', 'STFT+Swin-T', 'STFT+Hybrid', 'OASIS-Net']
ABL = ['OASIS-Net', 'OASIS w/o adversary', 'OASIS w/o order branch', 'OASIS w/o scalogram branch']
ps = clean.groupby(['config', 'col', 'seed']).macro_f1.mean().reset_index()
agg = ps.groupby(['config', 'col']).macro_f1.agg(['mean', 'std']).reset_index()
M = agg.pivot(index='config', columns='col', values='mean').reindex(columns=COLS) * 100
S = agg.pivot(index='config', columns='col', values='std').reindex(columns=COLS) * 100
out = {}
out['main_mean'] = M.loc[MAIN].round(1).to_dict(orient='index')
out['main_sd'] = S.loc[MAIN].round(1).to_dict(orient='index')
out['abl_mean'] = M.loc[ABL].round(1).to_dict(orient='index')
out['abl_sd'] = S.loc[ABL].round(1).to_dict(orient='index')
lfm = M[LF].mean(axis=1)
out['lf_mean'] = lfm.loc[MAIN + ABL[1:]].round(1).to_dict()
# fault-only CWRU L3
l3 = clean[(clean.dataset == 'CWRU') & (clean.protocol == 'L3')].copy()
l3['fo'] = l3.f1_per_class.map(lambda s: float(np.mean(json.loads(s)[1:])))
fo = l3.groupby(['config', 'seed']).fo.mean().groupby('config').agg(['mean', 'std']) * 100
out['faultonly'] = fo.loc[MAIN + ABL[1:]].round(1).to_dict(orient='index')
# screening
scr = clean[(clean.model == 'resnet18') & (clean.seed == 0)]
st = (scr.groupby(['transform', 'col']).macro_f1.mean().unstack().reindex(columns=COLS) * 100).round(1)
st['val'] = (scr.drop_duplicates('run_key').groupby('transform').val_macro_f1.mean() * 100).round(1)
st = st.sort_values('val', ascending=False)
out['screen'] = st.to_dict(orient='index')
# inflation
infl = pd.DataFrame({'CWRU': M['CWRU L0'] - M['CWRU L3'], 'PU': M['PU L0'] - M['PU L3'], 'PU_A2R': M['PU L0'] - M['PU A2R']})
out['infl'] = infl.loc[list(dict.fromkeys(MAIN + list(st.index.map(lambda t: t + '+ResNet-18'))))].round(1).to_dict(orient='index')
# stats on leakage-free blocks
lf = clean[clean.config.isin(MAIN) & clean.col.isin(LF)]
B = lf.groupby(['config', 'fold', 'test']).macro_f1.mean().unstack('config')[MAIN].dropna()
fr = friedmanchisquare(*[B[c].values for c in MAIN])
ranks = B.rank(axis=1, ascending=False).mean().sort_values()
nem = sp.posthoc_nemenyi_friedman(B.values); nem.index = nem.columns = B.columns
out['friedman'] = dict(chi2=round(fr.statistic, 2), p=float(fr.pvalue), n_blocks=len(B))
out['ranks'] = ranks.round(2).to_dict()
out['nemenyi_sig'] = [(a, b, round(float(nem.loc[a, b]), 3)) for i, a in enumerate(MAIN) for b in MAIN[i + 1:] if nem.loc[a, b] < 0.05]
wil = {}
for a in MAIN[:-1]:
    w = wilcoxon(B['OASIS-Net'], B[a]); wil[a] = dict(p=float(w.pvalue), diff=round(100 * (B['OASIS-Net'] - B[a]).mean(), 1))
out['wilcoxon_oasis_vs'] = wil
w = wilcoxon(B['STFT+Hybrid'], B['Features+GBM']); out['wil_hybrid_vs_gbm'] = dict(p=float(w.pvalue), diff=round(100 * (B['STFT+Hybrid'] - B['Features+GBM']).mean(), 1))
# noise
nz = R[R.snr_db.notna() & R.config.isin(MAIN)]
nt = (nz.groupby(['dataset', 'config', 'snr_db']).macro_f1.mean().unstack('snr_db') * 100).round(1)
out['noise'] = {f'{d}|{c}': row.to_dict() for (d, c), row in nt.iterrows()}
# cost
co = clean[clean.config.isin(MAIN + ABL[1:])].groupby('config').agg(n_params=('n_params', 'first'), train=('train_time_s', 'mean'), infer=('infer_ms', 'mean'))
tc = pd.read_csv(f'{RES}/tables/transform_cost.csv').set_index('transform').ms_per_window
inp = {'Features+GBM': np.nan, 'Raw+WDCNN': 0.0, 'STFT+ResNet-18': tc['STFT'], 'STFT+Swin-T': tc['STFT'], 'STFT+Hybrid': tc['STFT'],
       'OASIS-Net': tc['CWT'] + tc['order'], 'OASIS w/o adversary': tc['CWT'] + tc['order'], 'OASIS w/o order branch': tc['CWT'],
       'OASIS w/o scalogram branch': tc['order']}
co['input'] = pd.Series(inp)
out['cost'] = co.round(3).to_dict(orient='index'); out['transform_cost'] = tc.round(3).to_dict()
# audit and pools
out['audit'] = pd.read_csv(f'{RES}/tables/leakage_audit.csv').round(4).to_dict(orient='records')
pn = json.load(open(f'{RES}/paper_numbers.json'))
for k in ('N_WINDOWS_CWRU_OVL', 'N_WINDOWS_CWRU_NON', 'N_WINDOWS_PU_OVL', 'N_WINDOWS_PU_NON', 'OVERLAP_FRAC_CWRU_L0', 'OVERLAP_FRAC_PU_L0',
          'PROBE_ACC_OASIS', 'PROBE_ACC_NOADV', 'PROBE_CHANCE', 'ORDER_FIG_BEARING'):
    out[k] = pn[k]
# selection study
sel = pd.read_csv(f'{RES}/oasis2_selection.csv')
st2 = sel.pivot(index='cfg', columns='inner', values='f1') * 100
st2['mean'] = st2.mean(axis=1); st2 = st2.sort_values('mean', ascending=False)
out['selection'] = st2.round(1).reset_index().to_dict(orient='records')
out['selection_held'] = sel.drop_duplicates('inner').set_index('inner').held.to_dict()
# per-factor marginal means of the selection grid
f = sel.copy(); f[['enc', 'img', 'lam', 'mix']] = f.cfg.str.split('|', expand=True)
out['selection_marginal'] = {k: (f.groupby(k).f1.mean() * 100).round(1).to_dict() for k in ['enc', 'img', 'lam', 'mix']}
json.dump(out, open(f'{OUT}/numbers.json', 'w'), indent=1, default=float)
print(json.dumps({k: out[k] for k in ['lf_mean', 'friedman', 'ranks', 'nemenyi_sig', 'wilcoxon_oasis_vs', 'wil_hybrid_vs_gbm', 'selection_marginal']}, indent=1, default=float))
print(M.loc[MAIN + ABL[1:]].round(1).to_string())
