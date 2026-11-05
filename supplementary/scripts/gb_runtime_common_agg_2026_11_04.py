import json, glob, numpy as np, pandas as pd
from scipy.stats import spearmanr
R=(__import__("os").environ["GEOBRIDGE_ROOT"] + '/results/sergio_tf32rep_2026_10_22'); N=R+'/runtime_common_2026_11_04'
rows=[]; GPU='NVIDIA RTX A6000'
for f in sorted(glob.glob(N+'/rt/*.json')):
    j=json.load(open(f)); n=f.split('/')[-1][:-5]; m,g,t=n.rsplit('_',2)
    if m=='GB':
        for nm,k in (('GeoBRIDGE (preprocess+score)','gb_pre_score_s'),('GeoBRIDGE (score only)','gb_score_only_s')): rows.append((nm,'CPU',j['host'],'CPU node 72 cores, OMP/MKL=8',t,g,j[k]))
        rows.append(('GeoBRIDGE (total incl. SERGIO sim)','CPU',j['host'],'CPU node 72 cores, OMP/MKL=8',t,g,j['total_wall_s']))
    else:
        dev='GPU' if m in('deepsem','regdiffusion','fasttenet','velorama') else 'CPU'
        hwd=(j.get('gpu') or GPU)+' (1 GPU)' if dev=='GPU' else {'genie3':'CPU node dask 8 workers x1 thr','grnboost2':'CPU node dask 8 workers x1 thr','pidc':'CPU node torch 4 threads','pearson':'CPU node numpy default threads'}[m]
        rows.append((m,dev,j['host'],hwd,t,g,j['method_s'])); rows.append((m+' [total wall incl. load]',dev,j['host'],hwd,t,g,j['total_wall_s']))
D=pd.DataFrame(rows,columns='method device host hardware topo dataset seconds'.split()); D.to_csv(N+'/RUNTIME_COMMON.tsv',sep='\t',index=False)
S=D.groupby(['method','device','topo']).seconds.mean().round(2).unstack('topo'); S.to_csv(N+'/RUNTIME_COMMON_SUMMARY.tsv',sep='\t'); print(S.to_string())
# sanity vs stored scores
for m in ('genie3','grnboost2','pidc','pearson','deepsem','regdiffusion','fasttenet'):
    for f in sorted(glob.glob(f'{N}/scores/{m}/*.npz'))[:4]:
        a=np.load(f)['score']; b=np.load(f.replace(N,R))['score']; print('SANITY',m,f.split('/')[-1],'exact' if np.array_equal(a,b) else 'rho=%.4f'%spearmanr(a.ravel(),b.ravel())[0])
for f in sorted(glob.glob(N+'/g*/*/C3/*/raw/geobridge_raw.npz')):
    a=np.load(f); b=np.load(f.replace(N,R))
    print('SANITY GB',f.split('/')[-5:-3],{k:('exact' if np.array_equal(a[k],b[k]) else 'maxabs=%.2e'%np.abs(a[k].astype(float)-b[k]).max()) for k in a.files if a[k].dtype.kind in 'fi' and k in b.files and a[k].shape==b[k].shape})
for f in sorted(glob.glob(N+'/velorama_rt0/*/*.npz')):
    a=np.load(f)['score']; b=np.load(f.replace(N+'/velorama_rt0/'+f.split('/')[-2],R+'/g'+f.split('/')[-2].split('_')[0][1:]+'/'+f.split('/')[-2].split('_')[1]+'/velorama_rt0/C3'))['score']
    print('SANITY velorama',f.split('/')[-2],'exact' if np.array_equal(a,b) else 'rho=%.4f'%spearmanr(a.ravel(),b.ravel())[0])
