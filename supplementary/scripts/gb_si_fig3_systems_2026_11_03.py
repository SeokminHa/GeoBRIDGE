"""SI: Fig 3 d-g for PANC / HEART / BM. Plot-only; reads existing outputs (PANC dev/geom parity-rerun verified identical). usage: script OUT.png"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=(__import__("os").environ["GEOBRIDGE_ROOT"] + '/results/signed/'); FB=R+'fig3d_rebuild_2026_10_29/'; PA=R+'r2fix_2026_11_01/SI_fig3_other_systems/'
plt.rcParams.update({'font.family':'sans-serif','font.size':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6})
GREEN,PURP,GREY='#1b9e77','#7570b3','#7f7f7f'
PAL=['#e7298a','#7570b3','#e6ab02','#66a61e','#d95f02','#1f78b4']
SYS=[('Pancreas',PA+'parity_panc_dev/panc_devstructure.tsv',PA+'parity_panc_geom/geom30_pooled.tsv','depth_multi_B9999/panc',[('PANC_RFX6','RFX6','#7570b3')],['Alpha','Beta','Delta','Epsilon']),
 ('Heart',FB+'multisys/devstructure_CARDIACV/cardiacv_devstructure.tsv',FB+'multisys/geom30_CARDIACV/geom30_pooled.tsv','depth_multi_B9999/heart',[('HEART_TBX5','TBX5','#d95f02'),('HEART_NKX2-5','NKX2-5','#e7298a')],[]),
 ('Bone marrow',FB+'multisys/devstructure_BONEMARROW/bonemarrow_devstructure.tsv',FB+'multisys/geom30_BONEMARROW/geom30_pooled.tsv','depth_multi_B9999/bm',[('BM_GATA1_acute','GATA1','#999999')],['Ery_2','Mono_2','Mega','DCs','CLP'])]
fig,AX=plt.subplots(3,4,figsize=(11,7.2),gridspec_kw=dict(width_ratios=[.7,1.2,1.2,1.2]))
for i,(nm,dv,gm,dm,eps,fates) in enumerate(SYS):
    DS=pd.read_csv(dv,sep='\t'); G=pd.read_csv(gm,sep='\t'); a=AX[i]
    cur=lambda s:np.r_[0,G[G.signal==s].set_index('k').captured.loc[np.arange(1,31)].to_numpy()]
    T=DS[DS.block=='T_temporal']; I=DS[DS.quantity=='incremental CV R2 of fate one-hot beyond spline']
    a[0].bar(0,T.value.mean(),.6,color=GREEN,alpha=.85); a[0].plot([-.28,.28],[T.null_mean.mean()]*2,'k',lw=1); a[0].text(0,T.value.mean()+.03,f'{T.value.mean():.2f}',ha='center')
    if len(I): a[0].bar(1,I.value.mean(),.6,color=PURP,alpha=.85); a[0].plot([.72,1.28],[I.null_mean.mean()]*2,'k',lw=1); a[0].text(1,I.value.mean()+.03,f'{I.value.mean():.2f}',ha='center')
    else: a[0].text(1,.05,'no cell-type\nannotation',ha='center',fontsize=6,color=GREY)
    a[0].set_xticks([0,1]); a[0].set_xticklabels(['Pseudotime','Terminal\nidentity']); a[0].set_ylim(0,.85); a[0].set_ylabel(f'{nm}\nCross-validated R²')
    k=np.arange(11)
    for ax in a[1:3]: ax.axvspan(0,8,color='k',alpha=.06,lw=0); ax.plot(k,cur('Pi')[:11],':',color=GREY,lw=1.1,label='Pair-difference energy'); ax.set_xlim(0,10); ax.set_ylim(0,1); ax.set_xlabel('Conditioning depth k'); ax.set_ylabel('Fraction captured')
    a[1].plot(k,cur('tau')[:11],color=GREEN,lw=1.3,label='Pseudotime'); a[1].legend(frameon=False,fontsize=6,loc='lower right')
    for j,f in enumerate(fates): a[2].plot(k,cur(f'fate_{f}|spline')[:11],color=PAL[j],lw=1.1,label=f)
    if fates: a[2].legend(frameon=False,fontsize=5.8,loc='upper left',ncol=2)
    else: a[2].text(5,.5,'no terminal-identity annotation',ha='center',fontsize=6.5,color=GREY)
    D=pd.read_csv(FB+dm+'/depth_multi.tsv',sep='\t'); D=D[(D.metric=='AUROC')&(D.direction=='ALL')]; ax=a[3]; ax.axvspan(0,8,color='k',alpha=.06,lw=0)
    for ep,tf,c in eps:
        q=D[D.endpoint==ep].set_index('subset'); v=[q.loc[f'cum0-{x}','value'] for x in range(17)]; ax.plot(range(17),v,color=c,lw=1.2,label=tf); ax.plot(8,v[8],'D',color=c,ms=4)
        print(nm,tf,'AUROC k0=%.3f cum0-8=%.3f'%(v[0],v[8]))
    ax.axhline(.5,color='k',lw=.5,ls=':'); ax.set_xlim(0,16); ax.set_ylim(.45,.70); ax.set_xlabel('Maximum aggregation depth K'); ax.set_ylabel('AUROC'); ax.legend(frameon=False,fontsize=6,loc='upper left')
    if i==0:
        for ax,t in zip(a,'defg'): ax.set_title(t,loc='left',fontweight='bold',fontsize=9)
    print(nm,'R2 roots',T.value.round(3).tolist(),'dR2 roots',I.value.round(3).tolist(),'| k=1: tau %.3f Pi %.3f'%(cur('tau')[1],cur('Pi')[1]))
    for f in fates: print('  ',f,'k1 %.3f k8 %.3f'%(cur(f'fate_{f}|spline')[1],cur(f'fate_{f}|spline')[8]))
fig.tight_layout(); fig.savefig(sys.argv[1],dpi=300,bbox_inches='tight')
