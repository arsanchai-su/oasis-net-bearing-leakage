import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import scikit_posthocs as sp
RES, OUT = sys.argv[1], sys.argv[2]
N = json.load(open(f'{OUT}/numbers.json'))
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9, 'savefig.dpi': 400, 'savefig.bbox': 'tight'})
MAIN = ['Features+GBM', 'Raw+WDCNN', 'STFT+ResNet-18', 'STFT+Swin-T', 'STFT+Hybrid', 'OASIS-Net']
COLS = ['CWRU L0', 'CWRU L1', 'CWRU L2', 'CWRU L3', 'PU L0', 'PU L1', 'PU L3', 'PU X', 'PU A2R']
STY = {'Features+GBM': ('#1f77b4', 'o', '-'), 'Raw+WDCNN': ('#ff7f0e', 's', '-'), 'STFT+ResNet-18': ('#d62728', 'v', '-'),
       'STFT+Swin-T': ('#9467bd', '^', '-'), 'STFT+Hybrid': ('#2ca02c', 'D', '-'), 'OASIS-Net': ('black', 'o', '--')}

# ---------- Figure 1: study design ----------
def box(ax, x, y, w, h, kind, lw=1.2):
    ls = {'data': (0, (5, 3)), 'model': '-', 'analysis': (0, (1, 2))}[kind]
    ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=lw, ls=ls, ec='black'))
def txt(ax, x, y, s, **k):
    ax.text(x, y, s, ha='center', va='center', fontsize=k.pop('fs', 8), **k)
def arrow(ax, pts):
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color='black', lw=1.1)
    ax.annotate('', xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle='-|>', color='black', lw=1.1, mutation_scale=9))
fig, ax = plt.subplots(figsize=(7.2, 4.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 61); ax.axis('off')
box(ax, 1, 34, 27, 26, 'model'); txt(ax, 14.5, 57.5, 'Input data', fs=9)
box(ax, 3, 45.5, 23, 9.5, 'data'); txt(ax, 14.5, 50.2, 'CWRU, 12 kHz\n10 classes\nloads 0–3 hp', fs=7.5)
box(ax, 3, 35.5, 23, 9, 'data'); txt(ax, 14.5, 40, 'Paderborn, 16 kHz\n27 bearings, 3 classes\n4 operating settings', fs=7.5)
box(ax, 1, 5, 27, 27, 'model'); txt(ax, 14.5, 29.5, 'Windows and protocols', fs=8.5)
box(ax, 3, 18, 23, 9, 'data'); txt(ax, 14.5, 22.5, '1024-sample windows;\noverlapped and\nnon-overlapping pools', fs=7.5)
box(ax, 3, 6.5, 23, 9.5, 'analysis'); txt(ax, 14.5, 11.2, 'Leakage ladder + audit\nCWRU: L0, L1, L2, L3\nPU: L0, L1, L3, X, A2R', fs=7.5)
arrow(ax, [(14.5, 34), (14.5, 32.2)]); arrow(ax, [(14.5, 18), (14.5, 16.2)])
box(ax, 33, 26, 66, 34, 'model'); txt(ax, 66, 57.5, 'Models trained on identical splits', fs=9)
for (x, y, t) in [(35, 41, 'A. Signal-to-image screening\n6 encodings (STFT, CWT, GASF,\nGADF, MTF, RP) × ResNet-18;\nencoding chosen on validation'),
                  (67, 41, 'B. Image backbones\nselected encoding ×\nResNet-18, Swin-T,\nCNN–Swin hybrid'),
                  (35, 27.5, 'C. Non-image baselines\nWDCNN on raw windows;\ngradient boosting on\n46 hand-crafted features'),
                  (67, 27.5, 'D. Invariance-driven design\nOASIS-Net: scalogram, order\nspectrum, specimen adversary;\nablations, selection study')]:
    box(ax, x, y, 30, 12.5, 'model'); txt(ax, x + 15, y + 6.25, t, fs=7.0)
arrow(ax, [(28, 11.2), (30.5, 11.2), (30.5, 43), (33, 43)])
box(ax, 33, 1, 66, 22, 'model'); txt(ax, 66, 20.5, 'Shared inference', fs=9)
box(ax, 35, 2.5, 30, 15, 'analysis'); txt(ax, 50, 10, 'Performance\nmacro-F1 per protocol,\nmean ± SD over 3 seeds;\ninflation L0 − L3', fs=7.3)
box(ax, 67, 2.5, 30, 15, 'analysis'); txt(ax, 82, 10, 'Statistics and diagnostics\nFriedman, Nemenyi, Wilcoxon\n(8 leakage-free blocks); noise\n−4…10 dB; probe; cost', fs=7.3)
arrow(ax, [(50, 26), (50, 23.2)]); arrow(ax, [(82, 26), (82, 23.2)])
for i, (k, lab) in enumerate([('data', 'Data'), ('model', 'Model'), ('analysis', 'Analysis')]):
    ax.plot([1 + 16 * i, 6 + 16 * i], [-1.5, -1.5], color='black', lw=1.2, ls={'data': (0, (5, 3)), 'model': '-', 'analysis': (0, (1, 2))}[k])
    ax.text(7 + 16 * i, -1.5, lab, va='center', fontsize=8)
ax.set_ylim(-3, 61)
fig.savefig(f'{OUT}/figs/fig1_design.png'); plt.close(fig)

# ---------- Figure 2: OASIS-Net ----------
fig, ax = plt.subplots(figsize=(7.4, 3.1)); ax.set_xlim(0, 104); ax.set_ylim(0, 44); ax.axis('off')
def b2(x, y, w, h, t, dash=False, bold=False):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=1.8 if bold else 1.1, ls=(0, (4, 3)) if dash else '-', ec='black'))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=7.6)
b2(0, 17, 14, 10, 'Window\n1024 samples\n+ shaft speed')
b2(17, 29, 19, 10, 'CWT scalogram\nMorlet, 64 scales\n→ 224 × 224')
b2(17, 5, 19, 10, 'Envelope order\nspectrum\n0.25–12 orders')
b2(40, 29, 15, 10, 'Swin-T\n(ImageNet)\n768-d')
b2(40, 5, 15, 10, '1-D CNN\n3 blocks\n128-d')
b2(58, 17, 17, 10, 'Gated fusion\nf ⊙ σ(Wf + b)')
b2(80, 26, 23, 9, 'Classifier\nLayerNorm, linear\nfault class', bold=True)
b2(58, 1, 17, 9, 'Gradient\nreversal (−λ)', dash=True)
b2(80, 1, 23, 9, 'Specimen adversary:\nrecording or bearing\ngiven the class', dash=True)
for p in ([(14, 24), (15.5, 24), (15.5, 34), (17, 34)], [(14, 20), (15.5, 20), (15.5, 10), (17, 10)], [(36, 34), (40, 34)], [(36, 10), (40, 10)],
          [(55, 34), (56.5, 34), (56.5, 24), (58, 24)], [(55, 10), (56.5, 10), (56.5, 20), (58, 20)], [(75, 22), (77.5, 22), (77.5, 30.5), (80, 30.5)]):
    arrow(ax, p)
ax.plot([66.5, 66.5], [17, 12.5], color='black', lw=1.1, ls=(0, (4, 3)))
ax.annotate('', xy=(66.5, 10), xytext=(66.5, 12.6), arrowprops=dict(arrowstyle='-|>', color='black', lw=1.1, mutation_scale=9))
ax.annotate('', xy=(80, 5.5), xytext=(75, 5.5), arrowprops=dict(arrowstyle='-|>', color='black', lw=1.1, mutation_scale=9, ls=(0, (4, 3))))
ax.text(45, 3, 'training only →', fontsize=7, ha='center', style='italic')
fig.savefig(f'{OUT}/figs/fig2_oasis.png'); plt.close(fig)

# ---------- Figure 4: leakage ladder ----------
M = {c: N['main_mean'][c] for c in MAIN}; S = {c: N['main_sd'][c] for c in MAIN}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True, gridspec_kw=dict(width_ratios=[4, 5]))
for ax, cols, t in zip(axes, [COLS[:4], COLS[4:]], ['(a) CWRU', '(b) Paderborn']):
    for c in MAIN:
        col, mk, ls = STY[c]
        ax.errorbar(range(len(cols)), [M[c][k] for k in cols], yerr=[S[c][k] for k in cols], color=col, marker=mk, ls=ls,
                    lw=1.6 if c == 'OASIS-Net' else 1.0, ms=4, capsize=2, label=c)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([k.split()[1] for k in cols]); ax.set_title(t); ax.grid(alpha=.3)
axes[0].set_ylabel('Macro-F1 (%)'); axes[0].set_ylim(25, 102)
axes[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
fig.savefig(f'{OUT}/figs/fig4_ladder.png'); plt.close(fig)

# ---------- Figure 5: L0 vs leakage-free (does a random split predict generalisation?) ----------
R = pd.read_csv(f'{RES}/results.csv'); c = R[R.snr_db.isna()].copy()
c['col'] = c.dataset + ' ' + np.where(c.test.str.startswith('X-'), 'X', c.protocol.replace({'L3X': 'L3'}))
names = {'gbm': 'Features+GBM', 'wdcnn': 'Raw+WDCNN', 'oasis': 'OASIS-Net', 'oasis_noadv': 'OASIS w/o adversary',
         'oasis_noorder': 'OASIS w/o order branch', 'oasis_noimg': 'OASIS w/o scalogram branch'}
c = c[~c.model.str.startswith('oasis2')]
c['config'] = [names.get(m, t + '+' + {'resnet18': 'ResNet-18', 'swin': 'Swin-T', 'hybrid': 'Hybrid'}.get(m, m)) for m, t in zip(c.model, c['transform'])]
pc = c.groupby(['config', 'col']).macro_f1.mean().unstack() * 100
pts = pd.DataFrame({'L0': pc[['CWRU L0', 'PU L0']].mean(axis=1), 'LF': pc[['CWRU L3', 'PU L3', 'PU X', 'PU A2R']].mean(axis=1)}).dropna()
fig, ax = plt.subplots(figsize=(4.6, 3.4))
for k, r in pts.iterrows():
    mk = 'D' if k in MAIN else ('s' if k.startswith('OASIS') else 'o')
    ax.scatter(r.L0, r.LF, s=26, marker=mk, color='black' if k in MAIN else ('#777777' if k.startswith('OASIS') else 'white'), edgecolor='black', zorder=3)
    lab = k.replace('+ResNet-18', '') if (k not in MAIN and k != 'STFT+ResNet-18') else k
    lab = 'OASIS-Lite (order only)' if k == 'OASIS w/o scalogram branch' else lab
    pos = {'STFT+Hybrid': (86, 56.6), 'STFT+Swin-T': (86, 54.3), 'CWT+ResNet-18': (86, 53.4), 'Features+GBM': (86, 52.6),
           'OASIS-Net': (86, 51.8), 'OASIS w/o adversary': (86, 51.0), 'OASIS w/o order branch': (84, 50.2),
           'STFT+ResNet-18': (92.5, 46.9), 'Raw+WDCNN': (85.5, 48.6), 'OASIS w/o scalogram branch': (75.5, 49.0)}
    if k in pos:
        ax.annotate(lab, (r.L0, r.LF), xytext=pos[k], fontsize=6.3, va='center',
                    arrowprops=dict(arrowstyle='-', lw=0.5, color='#555555'))
    else:
        ax.annotate(lab, (r.L0, r.LF), xytext=(3, 3), textcoords='offset points', fontsize=6.3)
from scipy.stats import spearmanr
rho, p = spearmanr(pts.L0, pts.LF)
ax.set_xlabel('Mean macro-F1 under L0, random split (%)'); ax.set_ylabel('Mean leakage-free macro-F1 (%)')
ax.set_title(f'Spearman ρ = {rho:.2f} (p < 0.001, n = {len(pts)})' if p < 0.001 else f'Spearman ρ = {rho:.2f} (p = {p:.3f}, n = {len(pts)})', fontsize=8.5); ax.grid(alpha=.3)
fig.savefig(f'{OUT}/figs/fig5_l0_vs_lf.png'); plt.close(fig)
N['spearman_L0_LF'] = dict(rho=round(float(rho), 2), p=round(float(p), 3), n=len(pts), points=pts.round(1).to_dict(orient='index'))

# ---------- Figure 6: critical-difference diagram ----------
lf = c[c.config.isin(MAIN) & c.col.isin(['CWRU L3', 'PU L3', 'PU X', 'PU A2R'])]
B = lf.groupby(['config', 'fold', 'test']).macro_f1.mean().unstack('config')[MAIN].dropna()
ranks = B.rank(axis=1, ascending=False).mean()
nem = sp.posthoc_nemenyi_friedman(B.values); nem.index = nem.columns = B.columns
fig, ax = plt.subplots(figsize=(6.4, 2.0))
sp.critical_difference_diagram(ranks, nem, ax=ax, label_fmt_left='{label} ({rank:.2f})  ', label_fmt_right='  ({rank:.2f}) {label}',
                               text_h_margin=0.3, label_props={'fontsize': 7.5})
fig.savefig(f'{OUT}/figs/fig6_cd.png'); plt.close(fig)

# ---------- Figure 9: noise ----------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for ax, ds, t in zip(axes, ['CWRU', 'PU'], ['(a) CWRU L1 test set', '(b) Paderborn L3 (unseen bearings)']):
    for cfg in MAIN:
        d = N['noise'][f'{ds}|{cfg}']; xs = sorted(map(float, d)); col, mk, ls = STY[cfg]
        ax.plot(xs, [d[str(x)] if str(x) in d else d[x] for x in xs], color=col, marker=mk, ls=ls, ms=3.5, lw=1.6 if cfg == 'OASIS-Net' else 1, label=cfg)
    ax.set_title(t); ax.set_xlabel('Test-time SNR (dB)'); ax.grid(alpha=.3)
axes[0].set_ylabel('Macro-F1 (%)'); axes[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
fig.savefig(f'{OUT}/figs/fig9_noise.png'); plt.close(fig)
json.dump(N, open(f'{OUT}/numbers.json', 'w'), indent=1, default=float)
print('rho', rho, p, len(pts)); print(pts.round(1).sort_values('LF').to_string())
