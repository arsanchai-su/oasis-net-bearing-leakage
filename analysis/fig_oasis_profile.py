import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
N=json.load(open('numbers.json'))
MM,MS,IN,NZ,CO,LF,AM,AS=N['main_mean'],N['main_sd'],N['infl'],N['noise'],N['cost'],N['lf_mean'],N['abl_mean'],N['abl_sd']
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8.5,'savefig.dpi':400,'savefig.bbox':'tight'})
MAIN=['Features+GBM','Raw+WDCNN','STFT+ResNet-18','STFT+Swin-T','STFT+Hybrid','OASIS-Net']
lab={'Features+GBM':'GBM','Raw+WDCNN':'WDCNN','STFT+ResNet-18':'ResNet-18','STFT+Swin-T':'Swin-T','STFT+Hybrid':'Hybrid','OASIS-Net':'OASIS-Net','order':'OASIS-Lite'}
OC='#c0392b'; G='#9aa5b1'
fig,ax=plt.subplots(2,2,figsize=(7.2,5.6))
# (a) noise PU
a=ax[0,0]; sn=[-4,-2,0,2,4,6,8,10]
for c,col,ls in [('STFT+Hybrid','#34495e','--'),('STFT+Swin-T','#7f8c8d',':'),('OASIS-Net',OC,'-')]:
    d=NZ[f'PU|{c}']; a.plot(sn,[d[str(float(x))] for x in sn],ls,color=col,marker='o',ms=3,lw=1.6 if c=='OASIS-Net' else 1.2,label=lab[c])
a.set_xlabel('Test-time SNR (dB)'); a.set_ylabel('Macro-F1 (%)'); a.set_title('(a) Noise, unseen PU bearings',loc='left',fontsize=9)
a.legend(frameon=False,fontsize=7.5); a.grid(alpha=.3)
# (b) PU inflation
b=ax[0,1]; vals=[IN[c]['PU'] for c in MAIN]+[AM['OASIS w/o scalogram branch']['PU L0']-AM['OASIS w/o scalogram branch']['PU L3']]
names=[lab[c] for c in MAIN]+[lab['order']]; cols=[G]*5+[OC,OC]
o=np.argsort(vals); b.barh(np.arange(len(vals)),np.array(vals)[o],color=np.array(cols)[o])
b.set_yticks(np.arange(len(vals))); b.set_yticklabels(np.array(names)[o],fontsize=7.5)
for i,v in enumerate(np.array(vals)[o]): b.text(v+0.8,i,f'{v:.1f}',va='center',fontsize=7)
b.set_xlabel('PU inflation, L0 − L3 (points; lower is better)'); b.set_title('(b) Inflation on Paderborn',loc='left',fontsize=9); b.set_xlim(0,58)
# (c) seed SD PU L3 / X
c_=ax[1,0]; ks=['PU L3','PU X']; x=np.arange(len(MAIN)+1); w=.38
sd=lambda c,k: AS['OASIS w/o scalogram branch'][k] if c=='order' else MS[c][k]
mods=MAIN+['order']
for i,k in enumerate(ks):
    c_.bar(x+(i-.5)*w,[sd(m,k) for m in mods],w,color=[OC if m in('OASIS-Net','order') else G for m in mods],alpha=1 if i==0 else .6,label=k)
c_.set_xticks(x); c_.set_xticklabels([lab[m].replace('\n',' ') for m in mods],rotation=35,ha='right',fontsize=7.5)
c_.set_ylabel('SD over seeds (points)'); c_.set_title('(c) Stability on unseen PU bearings / settings',loc='left',fontsize=9)
c_.legend(['PU L3 (solid)','PU X (light)'],frameon=False,fontsize=7.5)
# (d) params vs PU leakage-free mean
d=ax[1,1]
pu=lambda m: (np.mean([AM['OASIS w/o scalogram branch'][k] for k in ['PU L3','PU X','PU A2R']]) if m=='order' else np.mean([MM[m][k] for k in ['PU L3','PU X','PU A2R']]))
par=lambda m: CO['OASIS w/o scalogram branch']['n_params'] if m=='order' else CO[m]['n_params']
for m in ['Raw+WDCNN','STFT+ResNet-18','STFT+Swin-T','STFT+Hybrid','OASIS-Net','order']:
    col=OC if m in('OASIS-Net','order') else G
    d.scatter(par(m),pu(m),s=40,color=col,edgecolor='k',lw=.4,zorder=3)
    d.annotate(lab[m].replace('\n',' '),(par(m),pu(m)),xytext=(5,-10 if m in ('STFT+Swin-T',) else 4),textcoords='offset points',fontsize=7.5)
d.set_xscale('log'); d.set_xlabel('Parameters (log scale)'); d.set_ylabel('Mean PU leakage-free macro-F1 (%)')
d.set_title('(d) Accuracy per parameter on Paderborn',loc='left',fontsize=9); d.grid(alpha=.3)
fig.tight_layout(); fig.savefig('fig10_oasis_profile.png')
print({m:round(pu(m),1) for m in ['Raw+WDCNN','STFT+ResNet-18','STFT+Swin-T','STFT+Hybrid','OASIS-Net','order']})
