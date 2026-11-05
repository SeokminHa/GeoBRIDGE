# Plot-only: reads already-computed tables; no statistics beyond arithmetic means noted in README.
import os, shutil, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
R='results/signed/r2fix_2026_11_01/'; O=R+'SI_FIGS_2026_11_03/'; os.makedirs(O,exist_ok=True)
PAL=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#6250d6','#e34948']
CL=['None','Temporal (A)','Branch (B)','A+B']; CC=['C0','C1','C2','C3']
G1,G2,G3='#555555','#888888','#bbbbbb'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
 'legend.fontsize':6.5,'axes.linewidth':0.7,'lines.linewidth':1.1,'axes.edgecolor':G2,'axes.labelcolor':'black','text.color':'black','xtick.color':G1,'ytick.color':G1,
 'axes.grid':True,'grid.color':'#e6e6e6','grid.linewidth':0.5,'axes.axisbelow':True,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'savefig.dpi':300})
def T(p,**k): return pd.read_csv(R+p,sep='\t',**k)
def lab(ax,s): ax.text(-0.02,1.04,s,transform=ax.transAxes,fontweight='bold',fontsize=9,ha='right',va='bottom')
def save(f,n): f.savefig(O+n,dpi=300,bbox_inches='tight'); plt.close(f); print('wrote',n)
def eb(ax,x,y,lo,hi,c,m='o',fill=True,ms=3.2,horiz=False,**k):
    x,y,lo,hi=map(np.asarray,(x,y,lo,hi)); fc=c if fill else 'white'
    if horiz: ax.errorbar(y,x,xerr=[y-lo,hi-y],fmt=m,color=c,mfc=fc,ms=ms,lw=1,capsize=0,elinewidth=1,**k)
    else: ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt=m,color=c,mfc=fc,ms=ms,lw=1,capsize=0,elinewidth=1,**k)


CP=['#2a78d6','#eb6834','#1baf7a','#eda100']          # conditions only
CL=['None','Temporal (A)','Branch (B)','A+B']; CC=['C0','C1','C2','C3']
EPS=['PANC_RFX6','BM_GATA1_acute','HEART_TBX5','HEART_NKX2-5']; EL=['Pancreas RFX6','Bone marrow GATA1','Heart TBX5','Heart NKX2-5']
EC=['#2a78d6','#eb6834','#eda100','#1baf7a']           # endpoints only (real-data figures)
K='black'; DG='#555555'; GR='#9a9a9a'
O1,O2,O3='#e87ba4','#6250d6','#008300'
def H(c=K,m='o',fill=True,ls='',label='',ms=3.2,lw=1): return Line2D([],[],marker=m,color=c,mfc=c if fill else 'white',ls=ls,ms=ms,lw=lw,label=label)

# ---------- S_kreal
w=T('T8_kwin_real/ALL_windows.tsv'); w=w[(w.direction=='ALL')&(w.metric=='AUROC')]
pv=T('T8_kwin_real/ALL_pairs_vs_cum0-8.tsv'); pv=pv[(pv.direction=='ALL')&(pv.metric=='AUROC')]
f,(a,b)=plt.subplots(1,2,figsize=(7.2,3.2),gridspec_kw={'width_ratios':[1,1.15],'wspace':0.5})
Ks=[0,4,6,8,9,10,12,16,30]; M=[]
for i,(e,l) in enumerate(zip(EPS,EL)):
    d=w[(w.endpoint==e)].set_index('subset').loc[['cum0-%d'%k for k in Ks]]
    xo=np.array(Ks)+(i-1.5)*0.25; eb(a,xo,d.value,d.lo,d.hi,EC[i],ms=2.5); a.plot(xo,d.value,color=EC[i],lw=1); M.append(d.value.values)
a.plot(Ks,np.mean(M,0),color=DG,lw=1.4,marker='s',ms=2.5)
a.axvline(8,color=K,ls=':',lw=1); a.axhline(0.5,color=G3,lw=0.8)
a.set_xlabel('Maximum depth K (depth range 0..K)'); a.set_ylabel('AUROC (all targets)'); lab(a,'a')
wins=['cum0-0','cum0-6','cum0-9','cum0-10','cum0-12','w1-8','w1-10','w2-8','pct1-8']
wl=['No conditioning\n(k=0)','0..6','0..9','0..10','0..12','1..8','1..10','2..8','Percentile\nmean 1..8']
for i,(e,l) in enumerate(zip(EPS,EL)):
    d=pv[pv.endpoint==e].set_index('contrast'); y=np.arange(len(wins))+(i-1.5)*0.15; d=d.loc[[x+' - cum0-8' for x in wins]]
    eb(b,y,d.value,d.lo,d.hi,EC[i],ms=2.5,horiz=True)
b.axvline(0,color=K,lw=0.8); b.set_yticks(range(len(wins))); b.set_yticklabels(wl); b.invert_yaxis()
b.set_xlabel('AUROC difference vs 0..8 (production), 95% CI'); b.set_xticks([-0.1,-0.05,0,0.05]); lab(b,'b')
f.legend(handles=[H(EC[i],label=EL[i]) for i in range(4)]+[H(DG,'s',ls='-',label='Equal-weight mean of endpoints'),H(K,'',ls=':',label='Production (depth 0..8)')],
  title='Colour = real-data endpoint',loc='upper center',bbox_to_anchor=(0.5,0.0),ncol=3)
save(f,'S_kreal.png')

# ---------- S_matched
s=T('T10_matched/full/summary.tsv'); s=s[s.scope=='pooled']
arms=['CONC_k0','COS_k0','PEAR_k0','CONC_k08','COS_k08','PEAR_k08','PCPART_cell_k08']
al=['Concordance k=0','Cosine k=0','Pearson k=0','Concordance k=0..8','Cosine k=0..8','Pearson k=0..8','Cell-space partial\ncorrelation k=0..8']
rd=pd.concat([T('T11_matched_real/%s/depth_multi.tsv'%p) for p in ['panc','bm','heart']]); rd=rd[(rd.direction=='ALL')&(rd.metric=='AUROC')]
f,ax2=plt.subplots(2,2,figsize=(7.2,6.6),gridspec_kw={'hspace':1.0,'wspace':0.3,'bottom':0.2}); ax=ax2.ravel()
a,b,c,d_=ax
for A,m,ttl in [(a,'AP','AP (mean of 6 GRNs)'),(b,'DvI23_AUROC','DvI23 AUROC (mean of 6 GRNs)')]:
    d=s[s.metric==m].set_index('arm').loc[arms]; A.scatter(range(7),d.arm_mean,color=K,s=12,zorder=3); A.set_ylabel(ttl)
for j,(m,col) in enumerate([('AP',K),('DvI23_AUROC',GR)]):
    d=s[s.metric==m].set_index('arm').loc[arms]; eb(c,np.arange(7)+(j-.5)*.2,d.prod_minus_arm,d.lo,d.hi,col,ms=2.5)
c.axhline(0,color=K,lw=0.8); c.set_ylabel('Production (GeoBRIDGE) minus arm, 95% CI')
sub=['cum0-0','cum0-8','COS_k0','COS_k08','PEAR_k0','PEAR_k08']; sl=['Concordance k=0','Concordance k=0..8','Cosine k=0','Cosine k=0..8','Pearson k=0','Pearson k=0..8']
for i,e in enumerate(EPS):
    d=rd[rd.endpoint==e].set_index('subset').loc[sub]; eb(d_,np.arange(6)+(i-1.5)*.15,d.value,d.lo,d.hi,EC[i],ms=2.5)
d_.axhline(0.5,color=G3,lw=0.8); d_.set_ylabel('AUROC (real data, all targets)')
for A in (a,b,c): A.set_xticks(range(7)); A.set_xticklabels(al,rotation=45,ha='right')
d_.set_xticks(range(6)); d_.set_xticklabels(sl,rotation=45,ha='right')
for A,l in zip(ax,'abcd'): lab(A,l)
f.legend(handles=[H(K,label='c: AP'),H(GR,label='c: DvI23 AUROC')],loc='lower left',bbox_to_anchor=(0.08,0.0),ncol=1)
f.legend(handles=[H(EC[i],label=EL[i]) for i in range(4)],title='d: colour = real-data endpoint',loc='lower left',bbox_to_anchor=(0.45,0.0),ncol=2)
save(f,'S_matched.png')

# ---------- S_binding
pc=T('T9_bind/full_v2/per_contrast.tsv'); po=T('T9_bind/full_v2/pooled.tsv')
cons=[('TBX5','KO-down'),('TBX5','KO-up'),('NKX2-5','KO-down'),('NKX2-5','KO-up'),('RFX6','KO-down'),('RFX6','KO-up')]
rows=cons+[('pooled','')]; rl=['%s %s'%(t,d) if t!='pooled' else 'Pooled' for t,d in rows]
au=pc[pc.metric=='AUROC_unsigned']; pq=po[po.quantity=='pooled unsigned AUROC (bound vs unbound)']
def getA(sc,t,d):
    r=(pq[pq.score==sc] if t=='pooled' else au[(au.score==sc)&(au.TF==t)&(au.direction==d)]).iloc[0]; return r.value,r.lo,r.hi
others=[x for x in au.score.unique() if x not in ('GeoBRIDGE','GeoBRIDGE k=0','label: |log2FC|')]
f=plt.figure(figsize=(7.2,6.4)); gs=f.add_gridspec(2,2,width_ratios=[1.2,1],height_ratios=[1.15,0.8],hspace=0.55,wspace=0.45)
a=f.add_subplot(gs[0,0]); b=f.add_subplot(gs[0,1]); c=f.add_subplot(gs[1,:])
for i,(t,d) in enumerate(rows):
    for o in others: a.plot(getA(o,t,d)[0],i,'o',color=GR,ms=2.5,mew=0,zorder=1)
    for sc,col,m,fill,off in [('GeoBRIDGE',K,'o',True,-.22),('GeoBRIDGE k=0',K,'o',False,0),('label: |log2FC|',DG,'D',True,.22)]:
        v,lo,hi=getA(sc,t,d); eb(a,[i+off],[v],[lo],[hi],col,m=m,fill=fill,horiz=True,ms=3)
a.axvline(0.5,color=G1,lw=0.8); a.set_yticks(range(7)); a.set_yticklabels(rl); a.invert_yaxis(); a.set_xlabel('Unsigned AUROC, bound vs unbound (95% CI)'); lab(a,'a')
a.legend(handles=[H(K,label='GeoBRIDGE'),H(K,fill=False,label='GeoBRIDGE k=0'),H(DG,'D',label='Label: |log2FC|'),H(GR,label='Other methods (no CI shown)')],loc='upper center',bbox_to_anchor=(0.5,-0.14),ncol=2)
sb=pc[pc.metric=='sign_agree_bound'].set_index(['TF','direction']); su=pc[pc.metric=='sign_agree_unbound'].set_index(['TF','direction'])
for k,(pp,tab,fill) in enumerate([(po[po.quantity=='pooled sign_agree_bound'].iloc[0],sb,True),(po[po.quantity=='pooled sign_agree_unbound'].iloc[0],su,False)]):
    v=[tab.loc[c_] for c_ in cons]+[pp]
    eb(b,np.arange(7)+(k-.5)*.25,[x.value for x in v],[x.lo for x in v],[x.hi for x in v],K,fill=fill,ms=2.8)
b.axhline(0.5,color=G1,lw=0.8); b.set_xticks(range(7)); b.set_xticklabels(rl,rotation=45,ha='right'); b.set_ylabel('Sign agreement'); lab(b,'b')
b.set_ylim(0,1.3); b.set_yticks([0,.25,.5,.75,1]); b.legend(handles=[H(K,label='Bound targets'),H(K,fill=False,label='Unbound targets')],loc='upper center',ncol=2,bbox_to_anchor=(0.5,1.03))
rules=['iso_C','iso_PP','fwd_C0','fwdPP','cellMsMu','rew_C']
rn=['Near-isochronic concordance (adopted)','Near-isochronic pair Pearson','Forward concordance, k=0','Forward pair Pearson','Cell Pearson (spliced–unspliced)','Reference-shifted concordance']
sr=pc[pc.metric=='signcand_agree_responders']
for t,d in cons:
    v=sr[(sr.TF==t)&(sr.direction==d)].set_index('score').loc[rules].value; c.plot(v.values,range(6),'o',color=GR,ms=2.5,mew=0)
pp=po[po.quantity=='pooled signcand_agree_responders'].set_index('score').loc[rules]
eb(c,range(6),pp.value,pp.lo,pp.hi,K,horiz=True,ms=3); c.axvline(0.5,color=G1,lw=0.8); c.axhspan(-.4,.4,color='#eeeeee',zorder=0)
c.set_yticks(range(6)); c.set_yticklabels(rn); c.invert_yaxis(); c.set_xlabel('Sign agreement among responders (specification curve over sign rules)'); lab(c,'c')
c.legend(handles=[H(K,label='Pooled (95% CI)'),H(GR,label='Per contrast')],loc='upper center',bbox_to_anchor=(0.5,-0.25),ncol=2)
save(f,'S_binding.png')

# ---------- S_degradation
e1=T('T12_story_E1E2E3E7/E1_degradation.tsv'); e1=e1[e1.topo=='all']
nm={'GB':'GeoBRIDGE','GB_k0':'GeoBRIDGE k=0'}
f,ax=plt.subplots(1,3,figsize=(7.2,3.7),sharey=True)
order=e1[(e1.contrast=='C3-C0')&(e1.metric=='DvI23_native')].sort_values('mean').method.tolist()
cl={'C1-C0':'Temporal (A) − None','C2-C0':'Branch (B) − None','C3-C0':'A+B − None'}
for a,cn,l in zip(ax,['C1-C0','C2-C0','C3-C0'],'abc'):
    for j,(mt,fill) in enumerate([('DvI23_native',True),('DvI23_z',False)]):
        d=e1[(e1.contrast==cn)&(e1.metric==mt)].set_index('method').loc[order]
        for i,m in enumerate(order):
            col=K if m=='GB' else DG if m=='GB_k0' else GR
            eb(a,[i+(j-.5)*.3],[d.loc[m,'mean']],[d.loc[m,'lo']],[d.loc[m,'hi']],col,fill=fill,horiz=True,ms=3)
    a.axvline(0,color=K,lw=0.8); a.set_yticks(range(len(order))); a.set_yticklabels([nm.get(m,m) for m in order]); a.set_xlabel('Δ DvI23 AUROC, '+cl[cn]); lab(a,l)
for t in ax[0].get_yticklabels():
    if t.get_text() in nm.values(): t.set_fontweight('bold'); t.set_color('black')
ax[0].invert_yaxis()
f.legend(handles=[H(DG,label='Native'),H(DG,fill=False,label='Per-source z'),H(K,'',ls='',label='')][:2]+[Line2D([],[],color=K,lw=3,label='GeoBRIDGE (black)'),Line2D([],[],color=DG,lw=3,label='GeoBRIDGE k=0 (dark grey)'),Line2D([],[],color=GR,lw=3,label='Other methods (grey)')],loc='upper center',ncol=5,bbox_to_anchor=(0.5,0.0))
save(f,'S_degradation.png')

# ---------- S_classrank
e9=T('E9_class_restore/E9_summary.tsv'); e9=e9[e9.topo=='all']; cl_=['d1','d2','d3','d4plus','unreach']; cn=['Direct (d=1)','d=2','d=3','d=4+','Unconnected']
e3=T('T12_story_E1E2E3E7/E3_class_means.tsv'); e3=e3[(e3.quantity=='rank_pct')&(e3.topo=='all')]
f,(a,b)=plt.subplots(1,2,figsize=(7.2,3.1),gridspec_kw={'wspace':0.35})
for i,c_ in enumerate(CC):
    d=e9[e9.contrast=='Delta_'+c_].set_index('cls').loc[cl_]; eb(a,np.arange(5)+(i-1.5)*.15,d['mean'],d.lo,d.hi,CP[i],ms=2.5)
    for m,fill,ls in [('GB',True,'-'),('GB_k0',False,'--')]:
        d=e3[(e3.method==m)&(e3.cond==c_)].iloc[0]; v=[d['d1'],d['d2'],d['d3'],d['d4plus'],d['unreach']]
        b.plot(np.arange(5)+(i-1.5)*.08,v,ls=ls,marker='o',ms=2.8,mfc=CP[i] if fill else 'white',color=CP[i],lw=0.9)
a.axhline(0,color=K,lw=0.8); b.axhline(0.5,color=G3,lw=0.8)
for A in (a,b): A.set_xticks(range(5)); A.set_xticklabels(cn,rotation=30,ha='right'); A.set_xlabel('Distance class (to nearest perturbed regulator)')
a.set_ylabel('Δ rank percentile (GeoBRIDGE − GeoBRIDGE k=0)'); b.set_ylabel('Mean rank percentile'); lab(a,'a'); lab(b,'b')
f.legend(handles=[H(CP[i],label=CL[i]) for i in range(4)]+[H(K,ls='-',label='GeoBRIDGE (b, filled)'),H(K,fill=False,ls='--',label='GeoBRIDGE k=0 (b, open)')],title='Colour = condition',loc='upper center',ncol=6,bbox_to_anchor=(0.5,-0.14))
save(f,'S_classrank.png')

# ---------- S_oracle
fu=T('E10_oracle/full/summary_full.tsv'); rk=T('E10_oracle/rank_audit/summary_rank.tsv')
arms_=[('GB_k0','GeoBRIDGE k=0',K,False),('ORACLE','Injected-profile oracle',O1,True),('ORACLE_RANK','Hidden-subspace oracle',O2,True),('ORACLE_RANK_Q08','Hidden-subspace oracle + PCs (k=0..8)',O3,True)]
def get(arm,m,c_):
    if arm in('ORACLE_RANK','ORACLE_RANK_Q08'):
        r=rk[(rk.metric==m)&(rk.cond==c_)&(rk.arm==arm)&(rk.ref=='GB_k08')]
        if len(r)==0: return None
        r=r.iloc[0]; return r['diff'],r.lo,r.hi
    r=fu[(fu.metric==m)&(fu.cond==c_)&(fu.arm==arm)].iloc[0]; return r.d_vs_GB_k08,r.lo_GB_k08,r.hi_GB_k08
def getd(arm,m):
    if arm in('ORACLE_RANK','ORACLE_RANK_Q08'): r=rk[(rk.metric==m)&(rk.cond=='C3-C0')&(rk.arm==arm)].iloc[0]; return r['diff'],r.lo,r.hi
    r=fu[(fu.metric==m)&(fu.cond=='C3-C0')&(fu.arm==arm)].iloc[0]; return r['mean'],r['lo'],r['hi']
f,ax=plt.subplots(1,4,figsize=(7.2,3.6),gridspec_kw={'width_ratios':[1,1,.8,.8],'wspace':0.75})
for k,(m,ml) in enumerate([('AP','AP'),('DvI23_AUROC','DvI23 AUROC')]):
    a=ax[k]
    for j,(arm,l,col,fill) in enumerate(arms_):
        pts=[(i,get(arm,m,c_)) for i,c_ in enumerate(CC)]; pts=[(i,p) for i,p in pts if p is not None]
        eb(a,[i+(j-1.5)*.15 for i,_ in pts],[p[0] for _,p in pts],[p[1] for _,p in pts],[p[2] for _,p in pts],col,fill=fill,ms=2.8)
    a.axhline(0,color=K,lw=0.8); a.set_xticks(range(4)); a.set_xticklabels(CL,rotation=30,ha='right'); a.set_ylabel('%s difference vs GeoBRIDGE (reference)'%ml); a.set_xlabel('Condition'); lab(a,'ab'[k])
for k,(m,ml) in enumerate([('AP','AP'),('DvI23_AUROC','DvI23 AUROC')]):
    a=ax[2+k]
    for j,(arm,l,col,fill) in enumerate([('GB_k08','GeoBRIDGE',K,True)]+arms_):
        if arm=='GB_k08': r=fu[(fu.metric==m)&(fu.cond=='C3-C0')&(fu.arm=='GB_k08')].iloc[0]; v=(r['mean'],r['lo'],r['hi'])
        else: v=getd(arm,m)
        eb(a,[j],[v[0]],[v[1]],[v[2]],col,fill=fill,ms=3)
    a.axhline(0,color=K,lw=0.8); a.set_xticks([]); a.set_xlabel('A+B vs None'); a.set_ylabel('Change in %s, A+B − None'%ml); lab(a,'cd'[k])
f.legend(handles=[H(K,label='GeoBRIDGE (reference; c,d only)')]+[H(c_,fill=fl,label=l) for _,l,c_,fl in arms_],title='Colour/marker = arm (95% CI)',loc='upper center',ncol=2,bbox_to_anchor=(0.5,-0.08))
save(f,'S_oracle.png')

# ---------- S_mech
e5=T('T13_story_E4E5/full/E5_energy_by_k_summary.tsv'); es=T('T13b_E4_strat/full/E4strat_summary_t95.tsv')
f,ax=plt.subplots(1,3,figsize=(7.2,3.4),gridspec_kw={'width_ratios':[1,1,1.1],'wspace':0.5})
for a,tp,l in zip(ax[:2],['tree','bifurcation'],'ab'):
    a.axvspan(0,8,color='#eeeeee',zorder=0)
    for i,c_ in enumerate(CC):
        for q,ls,lw in [('Mu','-',1.2),('AB','--',1.0),('GRN',':',1.4)]:
            d=e5[(e5.topo==tp)&(e5.cond==c_)&(e5.qty==q)].sort_values('k'); a.plot(d.k,d.frac_total,ls=ls,color=CP[i],lw=lw)
    a.set_xlabel('Depth k (removed components)'); a.set_ylabel('Energy captured (fraction of total)'); lab(a,l); a.text(.97,.04,tp.capitalize()+' topology',transform=a.transAxes,ha='right',color=G1)
st=['d2','d3','d4p','unreach','pooled']; stl=['d=2','d=3','d=4+','Unconnected','All non-direct']; b=ax[2]
for j,(ct,fill) in enumerate([('GB',True),('GB_k0',False)]):
    d=es[(es.cond=='C3')&(es.assoc=='corr')&(es.contrast==ct)].set_index('stratum').loc[st]; eb(b,np.arange(5)+(j-.5)*.2,d['mean'],d.lo95,d.hi95,K,fill=fill,ms=3)
b.axhline(0,color=K,lw=0.8); b.set_xticks(range(5)); b.set_xticklabels(stl,rotation=40,ha='right'); b.set_xlabel('Distance stratum'); b.set_ylabel('Score\u2013shared-signal Spearman (A+B)'); lab(b,'c')
f.legend(handles=[H(CP[i],'',ls='-',label=CL[i]) for i in range(4)],title='a,b: colour = condition',loc='upper left',bbox_to_anchor=(0.02,-0.15),ncol=2)
f.legend(handles=[Line2D([],[],color=DG,ls='-',label='Observed unspliced (Mu)'),Line2D([],[],color=DG,ls='--',label='Injected signal (A, B or A+B)'),Line2D([],[],color=DG,ls=':',lw=1.4,label='GRN production'),Patch(color='#eeeeee',label='Production range k=0..8')],title='a,b: line style',loc='upper center',bbox_to_anchor=(0.5,-0.15),ncol=1)
f.legend(handles=[H(K,label='GeoBRIDGE'),H(K,fill=False,label='GeoBRIDGE k=0')],title='c',loc='upper right',bbox_to_anchor=(1.0,-0.2),ncol=1)
save(f,'S_mech.png')

# ---------- S_fg
fg=T('T14_fg_direction/fg_auroc.tsv'); ar=T('T14_fg_direction/fg_arm_reslice.tsv')
FGL='Fraction of target change\nin removed subspace'
f,(a,b)=plt.subplots(1,2,figsize=(7.2,3.4),gridspec_kw={'width_ratios':[1,1.3],'wspace':0.4})
cs=[('KOdown','KO-down vs\nnon-responders'),('KOup','KO-up vs\nnon-responders'),('KOdown_vs_KOup','KO-down vs\nKO-up')]
for j,(c_,l) in enumerate(cs):
    d=fg[fg.contrast==c_].set_index('endpoint').loc[EPS]; eb(a,np.arange(4)*0+j+(np.arange(4)-1.5)*.15,d.auroc_f,d.lo,d.hi,'black',ms=0.1)
    for i in range(4): eb(a,[j+(i-1.5)*.15],[d.auroc_f.iloc[i]],[d.lo.iloc[i]],[d.hi.iloc[i]],EC[i],ms=3)
a.axhline(0.5,color=K,lw=0.8); a.set_xticks(range(3)); a.set_xticklabels([l for _,l in cs]); a.set_ylabel('AUROC of '+FGL.replace('\n',' ')); lab(a,'a')
a.set_ylabel('AUROC of fraction of target change\nin removed subspace')
arm=[('K0','k=0',K,False),('RND','Random directions',GR,True),('SPL','Pseudotime splines',O2,True),('PS','Mature-target (spliced) substitution',O3,True)]
for i,(A,al_,col,fill) in enumerate(arm):
    for dj,(dr,m) in enumerate([('KOdown','o'),('KOup','^')]):
        d=ar[(ar.arm==A)&(ar.direction==dr)].set_index('level').loc[EPS]; eb(b,np.arange(4)+(i-1.5)*.17+(dj-.5)*.06,d.value,d.lo,d.hi,col,m=m,fill=fill,ms=2.8)
b.axhline(0,color=K,lw=0.8); b.set_xticks(range(4)); b.set_xticklabels(EL,rotation=30,ha='right'); b.set_ylabel('GeoBRIDGE − control AUROC'); lab(b,'b')
f.legend(handles=[H(EC[i],label=EL[i]) for i in range(4)],title='a: colour = real-data endpoint',loc='upper left',bbox_to_anchor=(0.02,-0.12),ncol=1)
f.legend(handles=[H(c_,fill=fl,label=al_) for _,al_,c_,fl in arm]+[H(DG,'o',label='KO-down'),H(DG,'^',label='KO-up')],title='b: colour = control arm; shape = direction',loc='upper left',bbox_to_anchor=(0.45,-0.2),ncol=2)
save(f,'S_fg.png')


# ---------- S_pairwindow
pw=T('E11_pairwindow/summary.tsv'); LO=[0.0,0.01,0.02,0.05]; HI=[0.1,0.15,0.25,0.4,0.6]
cols=[('AP','C0','AP\nNone'),('AP','C3','AP\nA+B'),('DvI23_AUROC','C0','Direct-vs-indirect AUROC\nNone'),('DvI23_AUROC','C3','Direct-vs-indirect AUROC\nA+B')]
f=plt.figure(figsize=(7.2,7.4)); gs=f.add_gridspec(3,4,height_ratios=[1,1,0.9],hspace=0.8,wspace=0.6,bottom=0.14)
for r,(arm,rl) in enumerate([('GB_k08','GeoBRIDGE (k=0..8)'),('GB_k0','GeoBRIDGE k=0 (SI)')]):
    for c,(m,cd,tt) in enumerate(cols):
        a=f.add_subplot(gs[r,c]); d=pw[(pw.cap==40)&(pw.arm==arm)&(pw.cond==cd)]
        Z=d.pivot(index='lo',columns='hi',values=m).loc[LO,HI].values
        rg=np.nanmax(Z)-np.nanmin(Z); im=a.imshow(Z,cmap='Greys',aspect='auto',vmin=np.nanmin(Z)-0.2*rg,vmax=np.nanmax(Z)+2.5*rg)
        for i in range(4):
            for j in range(5): a.text(j,i,'%.2f'%Z[i,j],ha='center',va='center',fontsize=5.5,color='black')
        a.add_patch(plt.Rectangle((HI.index(0.25)-.5,LO.index(0.02)-.5),1,1,fill=False,ec=PAL[3],lw=1.8))
        a.set_xticks(range(5)); a.set_xticklabels(HI,fontsize=5.5); a.set_yticks(range(4)); a.set_yticklabels(LO); a.grid(False)
        for sp in a.spines.values(): sp.set_visible(False)
        if r==0: a.set_title(tt,fontsize=6.5)
        a.set_xlabel('upper bound'); 
        if c==0: a.set_ylabel(rl+'\nlower bound of Δpseudotime')
        if r==0 and c==0: lab(a,'a')
        if r==1 and c==0: lab(a,'b')
pm=pw[pw.lo.eq(0.02)&pw.hi.eq(0.25)]
for k,(m,ml) in enumerate([('AP','AP'),('DvI23_AUROC','Direct-vs-indirect AUROC')]):
    a=f.add_subplot(gs[2,2*k:2*k+2]); caps=[10,20,40,80]
    for i,cd in enumerate(CC):
        for arm,fill in [('GB_k08',True),('GB_k0',False)]:
            d=pm[(pm.cond==cd)&(pm.arm==arm)].set_index('cap').loc[caps]
            x=np.arange(4)+(i-1.5)*.07
            a.plot(x,d[m],color=CP[i],marker='o',ms=3,mfc=CP[i] if fill else 'white',ls='-' if fill else '--',lw=0.9)
    a.set_xticks(range(4)); a.set_xticklabels(caps); a.set_xlabel('Pair cap per cell'); a.set_ylabel(ml); lab(a,'cd'[k])
f.legend(handles=[H(CP[i],ls='-',label=CL[i]) for i in range(4)]+[H(K,ls='-',label='GeoBRIDGE (c,d)'),H(K,fill=False,ls='--',label='GeoBRIDGE k=0 (c,d)'),Patch(fc='white',ec=PAL[3],lw=1.8,label='Production window (0.02, 0.25)')],
  title='a,b: grey scale = value; c,d: colour = condition',loc='upper center',ncol=4,bbox_to_anchor=(0.5,0.07))
save(f,'S_pairwindow.png')

# ---------- S_isowindow
sm=T('E12_isowindow/SUMMARY_BY_METHOD_COND.tsv'); pr=T('E12_isowindow/PAIRED_VS_0p02.tsv'); pr=pr[pr.metric=='sign_BA']
Wn=[0.005,0.01,0.02,0.04,0.08]; f,(a,b)=plt.subplots(1,2,figsize=(7.2,3.3),gridspec_kw={'wspace':0.38})
for i,cd in enumerate(CC):
    d=sm[sm.cond==cd].set_index('method'); y=[d.loc['iso_w%g'%w,'sign_BA_mean'] for w in Wn]; sd=[d.loc['iso_w%g'%w,'sign_BA_std'] for w in Wn]
    a.plot(Wn,y,color=CP[i],marker='o',ms=3,lw=1.1); a.axhline(d.loc['fwd_k0_centred','sign_BA_mean'],color=CP[i],ls='--',lw=0.9)
    p=pr[pr.cond==cd].set_index('method'); ys=[p.loc['iso_w%g'%w,'diff_vs_0p02'] for w in Wn]
    xs=np.array(Wn)*(1+0.04*(i-1.5)); b.errorbar(xs,ys,yerr=[[p.loc['iso_w%g'%w,'diff_vs_0p02']-p.loc['iso_w%g'%w,'ci_lo'] for w in Wn],[p.loc['iso_w%g'%w,'ci_hi']-p.loc['iso_w%g'%w,'diff_vs_0p02'] for w in Wn]],color=CP[i],marker='o',ms=2.8,lw=1,elinewidth=1,capsize=0)
for A in (a,b): A.set_xscale('log'); A.set_xticks(Wn); A.set_xticklabels(Wn); A.minorticks_off(); A.axvline(0.02,color=K,ls=':',lw=1); A.set_xlabel('Near-isochronic window |Δpseudotime|')
a.set_ylabel('Sign balanced accuracy (mean over 6 GRNs)'); b.set_ylabel('Δ sign balanced accuracy vs 0.02, 95% CI'); b.axhline(0,color=K,lw=0.8); lab(a,'a'); lab(b,'b')
f.legend(handles=[H(CP[i],ls='-',label=CL[i]) for i in range(4)]+[Line2D([],[],color=DG,ls='--',label='Forward-pair k=0 (centred), per condition'),Line2D([],[],color=K,ls=':',label='Production window 0.02')],title='Colour = condition',loc='upper center',ncol=3,bbox_to_anchor=(0.5,-0.14))
save(f,'S_isowindow.png')

# ---------- S_kcurve_synth
kc=T('T5_kcurve/kcurve_summary.tsv'); kc=kc[kc.kind=='cumulative']; dk=T('T6_k0delta/delta_vs_k0.tsv'); lg=T('T7_kjustify/logo_cv.tsv'); pl=T('T7_kjustify/plateau.tsv')
f=plt.figure(figsize=(7.2,7.6)); gs=f.add_gridspec(4,3,width_ratios=[1,1,0.7],hspace=0.5,wspace=0.45,bottom=0.12)
TP=['tree','bifurcation']; rows=[('kc','AP','Cumulative AP'),('kc','DvI23_AUROC','Cumulative DvI AUROC'),('dk','AP','Δ AP vs K=0'),('dk','DvI23_AUROC','Δ DvI AUROC vs K=0')]
for r,(src,m,yl) in enumerate(rows):
    for c,tp in enumerate(TP):
        a=f.add_subplot(gs[r,c])
        for i,cd in enumerate(CC):
            d=(kc[(kc.topo==tp)&(kc.cond==cd)&(kc.metric==m)] if src=='kc' else dk[(dk.topo==tp)&(dk.cond==cd)&(dk.metric==m)]).sort_values('K')
            a.fill_between(d.K,d.lo,d.hi,color=CP[i],alpha=0.15,lw=0); a.plot(d.K,d['mean'],color=CP[i],lw=1.1)
        a.axvline(8,color=K,ls=':',lw=1)
        if src=='dk': a.axhline(0,color=K,lw=0.8)
        a.set_xlabel('Maximum depth K'); a.set_ylabel(yl if c==0 else ''); 
        if r==0: a.set_title(tp.capitalize()+' topology',fontsize=7)
        if c==0: lab(a,'abcd'[r])
a=f.add_subplot(gs[0:2,2]); mm=['AP','DvI23_AUROC','AUROC']; ml=['AP','DvI AUROC','AUROC']
for j,m in enumerate(mm):
    d=lg[lg.metric==m]; a.scatter(j+np.linspace(-.18,.18,len(d)),d.K_selected,color=K,s=9,zorder=3)
a.axhline(8,color=K,ls=':',lw=1); a.set_xticks(range(3)); a.set_xticklabels(ml,rotation=30,ha='right'); a.set_ylim(5,10); a.set_ylabel('K selected by leave-one-GRN-out CV'); lab(a,'e')
a.text(.5,.04,'6 folds (held-out GRN)\nper metric',transform=a.transAxes,ha='center',fontsize=6,color=G1)
f.legend(handles=[H(CP[i],ls='-',label=CL[i]) for i in range(4)]+[Line2D([],[],color=K,ls=':',label='K=8 (production)'),H(K,label='LOGO-CV selected K (e)')],title='Colour = condition (a-d); band = 95% CI over 6 GRNs',loc='upper center',ncol=3,bbox_to_anchor=(0.5,0.07))
save(f,'S_kcurve_synth.png')
