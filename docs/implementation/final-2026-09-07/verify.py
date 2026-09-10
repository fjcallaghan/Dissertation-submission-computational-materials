"""Independent numerical and source checks for the final revision."""
from pathlib import Path
import hashlib
import json
import math
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from src.detect.psy import recursive_adf, date_stamp

OUT = Path(__file__).resolve().parent
source=(ROOT/'thesis/main.tex').read_text()
before=(OUT/'before/main.tex').read_text()
labels=re.findall(r'\\label\{([^}]+)\}',source)
refs=set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',source))
assert len(labels)==len(set(labels)) and not refs-set(labels)
assert set(re.findall(r'\\label\{([^}]+)\}',before))<=set(labels)
cites={k.strip() for group in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}',source) for k in group.split(',')}
keys=set(re.findall(r'@\w+\s*\{\s*([^,]+),',(ROOT/'thesis/references.bib').read_text()))
assert not cites-keys
assert not re.search(r'\\(?:todo|vcite|vsup|vmath|vwrite|chk)\b',source)
assert 'university-ordinances/' not in source
assert source.index(r'\section{A structural comparison') < source.index(r'\section{The wider discrete')
result={'labels':len(labels),'citations':len(cites),'all_original_labels_preserved':True}
new=json.loads((OUT/'results.json').read_text())
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in new['input_sha256'].items())
result['preserved_observation_files']=len(new['input_sha256'])

df=pd.read_parquet(ROOT/'data/processed/aligned_daily_full.parquet').set_index('date').sort_index().asfreq('D')
baseline=np.load(ROOT/'docs/reviews/verification/baseline.npz')
draws=np.load(ROOT/'docs/reviews/verification/fresh_wild_199.npz')
y=np.log(df.close.to_numpy())
calculated=recursive_adf(y,p=1)
assert np.allclose(calculated.bsadf,baseline['bsadf'],atol=1e-10,rtol=0,equal_nan=True)
result['recomputed_GSADF']=calculated.gsadf_stat
result['minimum_window_observations']=calculated.min_obs
increments=np.diff(y);increments-=increments.mean()
rng=np.random.default_rng(12345)
for i in range(199):
    sample=np.r_[0,np.cumsum(increments*rng.choice((-1.,1.),len(increments)))]
    if i in (0,198):
        got=recursive_adf(sample,p=1)
        assert np.allclose(got.bsadf,draws['bsadf'][i],atol=1e-10,rtol=0,equal_nan=True)
        assert abs(got.gsadf_stat-draws['scalars'][i])<1e-10
result['bootstrap_paths_replayed']=[0,198]
valid=np.all(np.isfinite(draws['bsadf']),axis=0)
cv=np.full(len(df),np.nan);cv[valid]=np.quantile(draws['bsadf'][:,valid],.95,axis=0)
eps=date_stamp(calculated.bsadf,cv,min_duration=8)
assert len(eps)==15 and sum(b-a+1 for a,b in eps)==538
result['episodes']={'count':len(eps),'days':sum(b-a+1 for a,b in eps)}
df['B']=0
for a,b in eps:df.iloc[a:b+1,df.columns.get_loc('B')]=1
df['v']=df['return'].rolling(30,min_periods=10).std(ddof=1)
manual={}
for proxy in ['borrow_daily','funding_daily']:
    X=pd.DataFrame({'constant':1.,'Z':df[proxy].shift(1)*10000,
                    'r':df['return'].shift(1),'v':df.v.shift(1)})
    ok=X.notna().all(axis=1)&df.B.notna()
    fit=sm.Probit(df.loc[ok,'B'].to_numpy(),X.loc[ok].to_numpy()).fit(disp=False,maxiter=200)
    xx=X.loc[ok].to_numpy();yy=df.loc[ok,'B'].to_numpy()
    eta=xx@fit.params;pi=stats.norm.cdf(eta)
    g=stats.norm.pdf(eta)[:,None]*(yy-pi)[:,None]/(pi*(1-pi))[:,None]*xx
    assert np.allclose(g,fit.model.score_obs(fit.params),rtol=1e-7,atol=1e-8)
    g=pd.DataFrame(g,index=df.index[ok]).reindex(pd.date_range(df.index[ok][0],df.index[ok][-1],freq='D'),fill_value=0).to_numpy()
    omega=g.T@g
    for lag in range(1,31):
        term=g[lag:].T@g[:-lag]
        omega+=(1-lag/31)*(term+term.T)
    bread=np.linalg.inv(-fit.model.hessian(fit.params))
    covariance=bread@omega@bread
    se=math.sqrt(covariance[1,1])
    ame=float(np.mean(stats.norm.pdf(eta))*fit.params[1])
    saved=new['probit_effects'][proxy]
    assert abs(fit.params[1]-saved['coef_proxy'])<1e-10
    assert abs(se-saved['se_proxy'])<1e-9
    assert abs(ame-saved['average_marginal_effect'])<1e-10
    manual[proxy]={'coefficient':fit.params[1],'manual_HAC_se':se,'AME':ame}
result['manual_probit_score_and_covariance']=manual

option=[]
for expiry in ['2026-09-25','2026-10-30']:
    folder=ROOT/'pilot/option_reliability/outputs/reviewed_20260907'/expiry
    for filename in ['witness_A_0.json','witness_A_4.json']:
        law=json.loads((folder/filename).read_text())
        xs,ws=law['support'],law['probabilities']
        total=math.fsum(ws);mean=math.fsum(x*w for x,w in zip(xs,ws))
        violations=[]
        for quote in law['validation']['repriced']:
            price=math.fsum(w*max(quote['k']-x,0) for x,w in zip(xs,ws))
            lo,hi=quote['bid_normalised'],quote['ask_normalised']
            violations.append(max(0,(lo-price) if lo is not None else 0,(price-hi) if hi is not None else 0))
        error=max(abs(total-1),abs(mean-1+law['target_defect']),max(violations))
        assert error<1e-8
        option.append({'expiry':expiry,'file':filename,'mean':mean,'max_residual':error})
result['independent_scalar_option_repricing']=option
assert abs(1-140000/79630.82+60540/79630.82-.0021451493)<1e-10
result['worked_put_bound']=1-140000/79630.82+60540/79630.82
result['Bessel_mean']=2*stats.norm.cdf(2)-1
result['drawdown_recovery']=.5
result['source_sha256']=hashlib.sha256((ROOT/'thesis/main.tex').read_bytes()).hexdigest()
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
