#!/usr/bin/env python3
"""New synthetic experiment for paper 51; it is not the lost original experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / 'results'
DATA_SEED, MC_SEED, BOOT_SEED = 20261106, 42042, 8214
N_RUNS, N_BOOT = 1000, 10000
CRITERIA = [f'C{i}' for i in range(1, 10)]
METHODS = ['AHP-SAW', 'Entropy-SAW', 'Hybrid-SAW', 'AHP-weighted TOPSIS']
BASE_IMPORTANCE = np.array([.16,.20,.13,.16,.06,.08,.07,.07,.07])
# Category: count, (mean, sd, lower, upper) for C1, C2, C3, distance in metres,
# criticality, followed by ordinal means C5-C9 (SD=0.7, rounded and clipped to 1..5).
SPECS = {
 'Agricultural': (8, (100,35,5,200),(90,80,0,400),(3,2,0,10),(4500,2000,200,10000),(1.5,.7,.3,5), [2,4,2,5,2]),
 'Forest': (6, (85,35,5,180),(15,20,0,80),(3,2,0,10),(5000,2500,200,12000),(1.5,.8,.3,5), [5,2,4,2,1]),
 'Settlement-adjacent': (6, (22,15,5,65),(420,140,100,800),(5,2,0,12),(700,400,50,2000),(4.2,.5,2.5,5), [2,4,2,4,5]),
 'Critical infrastructure': (4, (18,10,5,50),(350,130,80,750),(4,2,0,10),(350,200,50,1000),(4.7,.3,3,5), [2,5,2,5,4]),
 'Open or abandoned': (3, (45,20,5,90),(20,20,0,100),(2,1.5,0,8),(4000,2000,200,10000),(1.4,.6,.3,4), [4,2,4,1,2]),
 'Wetland': (3, (25,12,5,60),(10,12,0,60),(2,1.5,0,8),(6000,2500,500,12000),(1.2,.5,.3,4), [5,1,5,1,2]),
}

def draw(rng, spec):
    mu, sd, lo, hi = spec
    return float(np.clip(rng.normal(mu, sd), lo, hi))

def generate():
    rng = np.random.default_rng(DATA_SEED); rows=[]
    for category, (count, area, pop, inc, dist, crit, ordinal) in SPECS.items():
        for _ in range(count):
            d, v = draw(rng,dist), draw(rng,crit)
            vals=[round(draw(rng,area),1), round(draw(rng,pop)), round(draw(rng,inc)), round(v*np.exp(-d/4000),4)]
            vals += list(np.clip(np.rint(rng.normal(ordinal,.7)),1,5).astype(int))
            rows.append(dict(ID=f'T{len(rows)+1:02d}', Category=category, Distance_m=round(d,6), Criticality=round(v,6), **dict(zip(CRITERIA,vals))))
    return pd.DataFrame(rows)

def ahp():
    # Round preference ratios to nearest Saaty intensity 1..9; enforce reciprocity.
    matrix=np.ones((9,9))
    for i in range(9):
        for j in range(i+1,9):
            ratio=BASE_IMPORTANCE[i]/BASE_IMPORTANCE[j]
            value=float(np.clip(np.floor(max(ratio,1/ratio)+.5),1,9))
            matrix[i,j]=value if ratio>=1 else 1/value; matrix[j,i]=1/matrix[i,j]
    eigvals,eigvecs=np.linalg.eig(matrix); k=int(np.argmax(eigvals.real))
    w=np.abs(eigvecs[:,k].real); w=w/w.sum()
    cr=(eigvals[k].real-9)/(9-1)/1.45
    return matrix,w,float(cr)

def normalize(x):
    lo=x.min(axis=0); span=x.max(axis=0)-lo
    r=np.divide(x-lo,span,out=np.zeros_like(x,dtype=float),where=span>0)
    for j in (4,6):
        if span[j]>0: r[:,j]=1-r[:,j]
    return r

def entropy(r):
    totals=r.sum(axis=0); p=np.divide(r,totals,out=np.zeros_like(r),where=totals>0)
    logp=np.zeros_like(p); np.log(p,out=logp,where=p>0)
    e=-(p*logp).sum(axis=0)/np.log(len(r)); divergence=1-e
    # Constant input has no discriminatory value, including all-zero columns.
    divergence[np.ptp(r,axis=0)<1e-12]=0
    return divergence/divergence.sum() if divergence.sum()>0 else np.ones(r.shape[1])/r.shape[1]

def scores(x, wa, alpha=.5):
    r=normalize(x); we=entropy(r); wh=alpha*wa+(1-alpha)*we
    weighted=r*wa; ideal=weighted.max(axis=0); anti=weighted.min(axis=0)
    dp=np.linalg.norm(weighted-ideal,axis=1); dm=np.linalg.norm(weighted-anti,axis=1)
    topsis=np.divide(dm,dp+dm,out=np.full(len(x),.5),where=dp+dm>0)
    return {'AHP-SAW':r@wa,'Entropy-SAW':r@we,'Hybrid-SAW':r@wh,'AHP-weighted TOPSIS':topsis},we,wh

def ranks(s):
    # Ties resolved deterministically by input order; ranks are unique positions.
    order=np.argsort(-s,kind='stable'); rank=np.empty(len(s),dtype=int);rank[order]=np.arange(1,len(s)+1)
    return rank

def rho(a,b): return float(np.corrcoef(a,b)[0,1])
def perturb(x,rng):
    y=x.copy()
    for j,sigma in ((0,.10),(1,.15),(3,.10)):
        y[:,j]=np.maximum(0,y[:,j]*(1+rng.normal(0,sigma,len(x))))
    y[:,2]=np.maximum(0,np.rint(y[:,2]+rng.normal(0,1,len(x))))
    for j,prob in ((4,.15),(5,.10),(6,.15),(7,.10),(8,.10)):
        changed=rng.random(len(x))<prob
        steps=rng.choice([-1,1],len(x)); y[:,j]=np.clip(y[:,j]+changed*steps,1,5)
    return y

def main():
    OUT.mkdir(exist_ok=True); df=generate(); df.to_csv(OUT/'synthetic_dataset.csv',index=False)
    x=df[CRITERIA].to_numpy(dtype=float); matrix,wa,cr=ahp()
    pd.DataFrame(matrix,index=CRITERIA,columns=CRITERIA).to_csv(OUT/'ahp_matrix.csv')
    base,we,wh=scores(x,wa); pd.DataFrame({'criterion':CRITERIA,'AHP':wa,'Entropy':we,'Hybrid':wh}).to_csv(OUT/'weights.csv',index=False)
    rank0={m:ranks(base[m]) for m in METHODS}; ranking=df[['ID','Category']].copy()
    for m in METHODS: ranking[m+'_score']=base[m]; ranking[m+'_rank']=rank0[m]
    ranking.sort_values('Hybrid-SAW_rank').to_csv(OUT/'rankings.csv',index=False)
    alpha_rows=[]
    for alpha in (0,.25,.5,.75,1):
        s,_,_=scores(x,wa,alpha); rr=ranks(s['Hybrid-SAW']); top=set(np.flatnonzero(rr<=5)); ref=set(np.flatnonzero(rank0['Hybrid-SAW']<=5))
        alpha_rows.append({'alpha':alpha,'spearman_vs_0_5':rho(rr,rank0['Hybrid-SAW']),'top5_overlap':len(top&ref)/5,'top5':', '.join(df.ID.iloc[np.argsort(rr)[:5]])})
    pd.DataFrame(alpha_rows).to_csv(OUT/'alpha_sensitivity.csv',index=False)
    rng=np.random.default_rng(MC_SEED); rows=[]
    for run in range(1,N_RUNS+1):
        xp=perturb(x,rng); ss,_,_=scores(xp,wa)
        for m in METHODS:
            rr=ranks(ss[m]); ref=rank0[m]; overlap=len(set(np.flatnonzero(rr<=5))&set(np.flatnonzero(ref<=5)))/5
            rows.append({'run':run,'method':m,'spearman':rho(rr,ref),'top5_overlap':overlap,'exact_top5':int(overlap==1),'mean_rank_change':float(np.abs(rr-ref).mean())})
    runs=pd.DataFrame(rows);runs.to_csv(OUT/'monte_carlo_runs.csv',index=False)
    results=[]
    for m,g in runs.groupby('method',sort=False):
        results.append({'method':m,'mean_spearman':g.spearman.mean(),'sd_spearman':g.spearman.std(ddof=1),'mean_top5_overlap':g.top5_overlap.mean(),'exact_top5_stability':g.exact_top5.mean(),'mean_rank_change':g.mean_rank_change.mean()})
    summary=pd.DataFrame(results);summary.to_csv(OUT/'robustness_summary.csv',index=False)
    # Paired bootstrap on run-wise differences, using shared perturbations.
    brng=np.random.default_rng(BOOT_SEED); a=runs[runs.method=='Hybrid-SAW'].set_index('run'); b=runs[runs.method=='AHP-weighted TOPSIS'].set_index('run'); boot=[]
    for metric in ('spearman','top5_overlap','exact_top5','mean_rank_change'):
        delta=(a[metric]-b[metric]).to_numpy(); sample=brng.integers(0,N_RUNS,size=(N_BOOT,N_RUNS)); means=delta[sample].mean(axis=1); lo,hi=np.quantile(means,[.025,.975]); boot.append({'comparison':'Hybrid-SAW minus AHP-weighted TOPSIS','metric':metric,'mean_difference':delta.mean(),'ci95_low':lo,'ci95_high':hi,'bootstrap_replicates':N_BOOT})
    pd.DataFrame(boot).to_csv(OUT/'paired_bootstrap.csv',index=False)
    meta={'dataset_seed':DATA_SEED,'perturbation_seed':MC_SEED,'bootstrap_seed':BOOT_SEED,'runs':N_RUNS,'bootstrap_replicates':N_BOOT,'AHP_CR_RI_1_45':cr,'numpy':np.__version__,'pandas':pd.__version__,'note':'New synthetic experiment; not the original data or original results.'}
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('AHP CR:',round(cr,6)); print('\n',summary.to_string(index=False)); print('\nSensitivity\n',pd.DataFrame(alpha_rows).to_string(index=False)); print('\nPaired bootstrap\n',pd.DataFrame(boot).to_string(index=False))
if __name__=='__main__': main()
