# Plot-only: main Fig 3 panels b,c (KO x ChIP) from T9_bind/full_v2. Run on a server from repo root.
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R='results/signed/r2fix_2026_11_01/'; O=R+'MAIN_DRAFT_2026_11_03/'; os.makedirs(O,exist_ok=True)
G1,G2,G3='#555555','#888888','#bbbbbb'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.5,
 'axes.linewidth':0.7,'axes.edgecolor':G2,'xtick.color':G1,'ytick.color':G1,'axes.grid':True,'grid.color':'#e6e6e6','grid.linewidth':0.5,'axes.axisbelow':True,
 'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'savefig.dpi':300})
pc=pd.read_csv(R+'T9_bind/full_v2/per_contrast.tsv',sep='\t'); po=pd.read_csv(R+'T9_bind/full_v2/pooled.tsv',sep='\t')
cons=[(t,d) for t in ['TBX5','NKX2-5','RFX6'] for d in ['KO-down','KO-up']]
rl=['%s %s'%c for c in cons]+['Pooled']
au=pc[pc.metric=='AUROC_unsigned']; pq=po[po.quantity=='pooled unsigned AUROC (bound vs unbound)']
def A(sc):
    r=[au[(au.score==sc)&(au.TF==t)&(au.direction==d)].iloc[0] for t,d in cons]+[pq[pq.score==sc].iloc[0]]
    return np.array([[x.value,x.lo,x.hi] for x in r])
oth=['|Pearson|','GENIE3','GRNBoost2','RegVelo |J|','Velorama','RegDiffusion','DeepSEM','PIDC','FastTENET']
best=max(oth,key=lambda s:pq[pq.score==s].value.iloc[0]); print('best other (pooled):',best)
S=[('GeoBRIDGE','GeoBRIDGE','k',True),('GeoBRIDGE k=0','GeoBRIDGE k=0','k',False),(best,'Best other: '+best,'#888888',True),('label: |log2FC|','|log2FC|','#bbbbbb','D')]
f=plt.figure(figsize=(7.2,3.6)); gs=f.add_gridspec(1,2,width_ratios=[1,1.1],wspace=0.55,bottom=0.28)
b=f.add_subplot(gs[0]); n=len(rl)
for j,(sc,l,c,fl) in enumerate(S):
    v=A(sc); y=np.arange(n)+(j-1.5)*.17
    b.errorbar(v[:,0],y,xerr=[v[:,0]-v[:,1],v[:,2]-v[:,0]],fmt='D' if fl=='D' else 'o',color=c,mfc='white' if fl is False else c,ms=3.2,lw=1,capsize=0,label=l)
b.axvline(.5,color='k',lw=.8); b.axhline(n-1.5,color=G3,lw=.6); b.set_yticks(range(n)); b.set_yticklabels(rl); b.invert_yaxis(); b.set_xlabel('Bound-vs-unbound AUROC (unsigned), 95% module bootstrap CI')
b.text(-0.02,1.04,'b',transform=b.transAxes,fontweight='bold',fontsize=9,ha='right')
b.legend(loc='upper center',bbox_to_anchor=(.45,-0.2),ncol=2)
c=f.add_subplot(gs[1]); sd=pc[pc.metric=='sign_agree_diff'].set_index(['TF','direction'])
def D(sc):
    r=[pc[(pc.metric=='signcand_agree_diff')&(pc.score==sc)&(pc.TF==t)&(pc.direction==d)].iloc[0] for t,d in cons]
    p=po[(po.quantity=='pooled signcand_agree_diff')&(po.score==sc)].iloc[0]
    return np.array([[x.value,x.lo,x.hi] for x in r]+[[p.value,p.lo,p.hi]])
for j,(sc,l,col,mk) in enumerate([('iso_C','GeoBRIDGE sign rule','k','o'),('cellMsMu','Cell-Pearson sign (reference)','#888888','s')]):
    v=D(sc); y=np.arange(n)+(j-.5)*.2
    c.errorbar(v[:,0],y,xerr=[v[:,0]-v[:,1],v[:,2]-v[:,0]],fmt=mk,color=col,ms=3.2,lw=1,capsize=0,label=l)
c.axvline(0,color='k',lw=.8); c.axhline(n-1.5,color=G3,lw=.6); c.set_yticks(range(n)); c.set_yticklabels(rl); c.invert_yaxis()
c.set_xlabel('Sign agreement with KO direction:\nbound minus unbound, 95% CI'); c.text(-0.02,1.04,'c',transform=c.transAxes,fontweight='bold',fontsize=9,ha='right')
c.legend(loc='upper center',bbox_to_anchor=(.45,-0.25),ncol=1)
f.savefig(O+'Fig3_bc_draft.png',dpi=300,bbox_inches='tight')
