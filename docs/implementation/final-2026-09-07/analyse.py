"""Offline final-thesis exhibits. Preserves observations and original analyses."""
from pathlib import Path
import hashlib
import json
import math
import sys
import warnings

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.detect.link import daily_frame, predictive_regression, realized_volatility, lead_lag_correlation
from src.detect.psy import date_stamp
from src.detect.slm import local_volatility, volatility_elasticity

OUT = Path(__file__).resolve().parent
THESIS = ROOT / 'thesis'
AUDIT = ROOT / 'docs/reviews/verification'
PILOT = ROOT / 'pilot/option_reliability/outputs/reviewed_20260907'
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 240, 'axes.labelsize': 11, 'legend.fontsize': 10})
report = {}
frame = pd.read_parquet(ROOT / 'data/processed/aligned_daily_full.parquet').set_index('date').sort_index()
frame = daily_frame(frame)
base = np.load(AUDIT / 'baseline.npz')
draws = np.load(AUDIT / 'fresh_wild_199.npz')
assert np.array_equal(np.log(frame.close.to_numpy()), base['y'])
frame['bsadf'] = base['bsadf']
frame['realized_vol'] = realized_volatility(frame['return'])
with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    wild_cv = np.nanquantile(draws['bsadf'], .95, axis=0)
schemes = {'wild': wild_cv, 'mc': base['cv']}

def label_frame(df, cv):
    df = df.copy()
    df['in_episode'] = 0
    for a, b in date_stamp(base['bsadf'], cv, min_duration=8):
        df.iloc[a:b+1, df.columns.get_loc('in_episode')] = 1
    return df

def fit(df, proxy):
    scaled = df.copy()
    scaled[proxy] *= 10000
    result = predictive_regression(scaled, regime_col='in_episode', proxy_col=proxy,
                                  control_cols=['return', 'realized_vol'], covariance='HAC', hac_lags=30)
    assert 'error' not in result
    return result

effects = {}
for proxy in ['borrow_daily', 'funding_daily']:
    df = label_frame(frame, wild_cv)
    result = fit(df, proxy)
    cols = [proxy, 'return', 'realized_vol']
    data = df[['in_episode', *cols]].copy()
    data[proxy] *= 10000
    data[cols] = data[cols].shift(1)
    data = data.dropna()
    x = sm.add_constant(data[cols].to_numpy())
    model = sm.Probit(data.in_episode.to_numpy(), x).fit(disp=False, maxiter=200)
    assert abs(model.params[1] - result['coef_proxy']) < 1e-12
    index = x @ model.params
    ame = float(np.mean(stats.norm.pdf(index)) * model.params[1])
    q1, q3 = np.quantile(x[:, 1], [.25, .75])
    xlo, xhi = x.copy(), x.copy()
    xlo[:, 1], xhi[:, 1] = q1, q3
    plo, phi = np.mean(stats.norm.cdf(xlo @ model.params)), np.mean(stats.norm.cdf(xhi @ model.params))
    effects[proxy] = {**result, 'all_parameters': model.params.tolist(),
        'average_marginal_effect': ame, 'ame_percentage_points_per_bp': ame*100,
        'quartile_rates_bp': [q1, q3], 'standardised_probabilities': [plo, phi],
        'quartile_contrast_percentage_points': 100*(phi-plo)}
report['probit_effects'] = effects

# Coefficients and covariance sensitivity on the actual estimation windows.
saved = json.loads((ROOT / 'docs/implementation/ch5/results.json').read_text())
rows = []
for proxy, name in [('borrow_daily', 'Borrow'), ('funding_daily', 'Funding')]:
    for scheme in ['mc', 'wild']:
        for window in (['all', 'common_2019', 'exclude_2017'] if proxy == 'borrow_daily' else ['all']):
            r = next(v for v in saved if (v['scheme'], v['window'], v['proxy']) == (scheme, window, proxy))
            f, f60 = r['fits']['30'], r['fits']['60']
            win = {'all': 'Full', 'common_2019': 'Common', 'exclude_2017': 'Without 2017'}[window]
            p = lambda v: f'{v:.4f}' if v >= .0001 else r'$<0.0001$'
            rows.append(f"{name} & {scheme.upper()} & {win} & {f['coef_proxy']:.4f} & "
                        f"[{f['ci95_proxy'][0]:.4f}, {f['ci95_proxy'][1]:.4f}] & "
                        f"{p(f['p_proxy'])} & {p(f60['p_proxy'])} " + r'\\')
(THESIS / 'ch5_final_sensitivity_rows.tex').write_text('\n'.join(rows)+'\n')

# Raw-rate and matched-venue comparisons, recomputed under both date schemes.
raw_specs = [('borrow_raw', 'bitfinex_BTC_borrow', 'borrow_time', 'borrow_rate', 'mean'),
             ('binance_raw', 'binance_BTCUSDT_funding', 'funding_time', 'funding_rate', 'sum')]
for name, file, stamp, value, method in raw_specs:
    raw = pd.read_parquet(ROOT / f'data/raw/{file}.parquet')
    frame[name] = raw.groupby(raw[stamp].dt.normalize())[value].agg(method).reindex(frame.index)
bybit = pd.read_parquet(ROOT / 'data/processed/bybit_funding_clean.parquet')
report['bybit_columns'] = bybit.columns.tolist()
rate_col = 'funding_rate_clean' if 'funding_rate_clean' in bybit else 'funding_rate'
assert rate_col == 'funding_rate_clean', bybit.columns
frame['bybit_clean'] = bybit.groupby(bybit['funding_time'].dt.normalize())[rate_col].sum().reindex(frame.index)
extra = []
for scheme, cv in schemes.items():
    df = label_frame(frame, cv)
    for proxy, name, window in [('borrow_raw', 'Borrow, raw', 'Full'),
                                ('binance_raw', 'Binance, raw', 'Full'),
                                ('funding_daily', 'Binance, winsorised', 'Matched'),
                                ('bybit_clean', 'Bybit, winsorised', 'Matched')]:
        sub = df.loc[df.bybit_clean.first_valid_index():] if window == 'Matched' else df
        result = fit(sub, proxy)
        curve = lead_lag_correlation(sub, proxy_col=proxy, stat_col='bsadf')
        peak = curve.loc[curve['corr'].idxmax()]
        extra.append({'scheme': scheme, 'name': name, 'window': window, **result,
                      'peak_lag': int(peak.lag), 'peak_corr': float(peak['corr'])})
report['raw_and_venue_sensitivity'] = extra
rows=[]
for r in extra:
    p = f"{r['p_proxy']:.4f}" if r['p_proxy'] >= .0001 else r'$<0.0001$'
    rows.append(f"{r['scheme'].upper()} & {r['name']} & {r['window']} & {r['coef_proxy']:.4f} & "
                f"[{r['ci95_proxy'][0]:.4f}, {r['ci95_proxy'][1]:.4f}] & {p} & {r['n']:,} " + r'\\')
(THESIS / 'app_final_rate_rows.tex').write_text('\n'.join(rows)+'\n')

# Conditional Monte Carlo precision, not a detector-size validation.
scalar = draws['scalars']
observed = float(np.nanmax(base['bsadf']))
m, exceed = len(scalar), int(np.sum(scalar > observed))
tail_interval = [stats.beta.ppf(.025, exceed, m-exceed+1), stats.beta.ppf(.975, exceed+1, m-exceed)]
lo_rank = int(stats.binom.ppf(.025, m, .95))
hi_rank = int(stats.binom.ppf(.975, m, .95)) + 1
qinterval = np.sort(scalar)[[lo_rank-1, hi_rank-1]].tolist()
mc = {'draws': m, 'exceedances': exceed, 'plus_one_p': (exceed+1)/(m+1),
      'conditional_exceedance_probability_Clopper_Pearson_95': tail_interval,
      'population_q95_order_ranks': [lo_rank, hi_rank], 'q95_MC_interval': qinterval,
      'resampling_repetitions': 1000, 'resampling_seed': 20260907}
rng = np.random.default_rng(20260907)
stability = []
valid = np.all(np.isfinite(draws['bsadf']), axis=0)
vals = draws['bsadf'][:,valid]
mask = np.zeros(len(frame), dtype=int)
baseline_mask = np.zeros(len(frame), dtype=bool)
for a,b in date_stamp(base['bsadf'], wild_cv, min_duration=8): baseline_mask[a:b+1] = True
for i in range(1000):
    ids = rng.integers(0, m, size=m)
    cv = np.full(len(frame), np.nan)
    cv[valid] = np.quantile(vals[ids], .95, axis=0)
    eps = date_stamp(base['bsadf'], cv, min_duration=8)
    current = np.zeros(len(frame), dtype=bool)
    for a,b in eps: current[a:b+1] = True
    mask += current
    stability.append([len(eps), int(current.sum()), int(np.sum(current != baseline_mask)), float(np.quantile(scalar[ids], .95))])
    if (i+1) % 250 == 0: print('Monte Carlo resamples:', i+1, flush=True)
stability = np.array(stability)
np.savez_compressed(OUT/'calibration_stability.npz', summaries=stability, flag_frequency=mask/1000)
mc['resampling_quantiles_025_50_975'] = {
    name: np.quantile(stability[:,j], [.025,.5,.975]).tolist()
    for j,name in enumerate(['episode_count','flagged_days','days_changed_from_baseline','scalar_q95'])}
mc['global_rejection_resample_fraction'] = float(np.mean(stability[:,3] < observed))
report['monte_carlo_precision'] = mc

# Explain the last principal episode through actual exceedance runs.
post = frame.index > pd.Timestamp('2024-04-01', tz=frame.index.tz)
all_runs = date_stamp(base['bsadf'], wild_cv, min_duration=1)
report['after_last_episode'] = {'raw_exceedance_runs': [
    [str(frame.index[a].date()), str(frame.index[b].date()), b-a+1]
    for a,b in all_runs if post[a]],
    'max_bsadf_minus_cv': float(np.nanmax((base['bsadf']-wild_cv)[post]))}

# Volatility fit and observation support.
prices = frame.close.to_numpy()
levels = prices[:-1]
n = len(levels)
h = max(0.9*min(levels.std(), np.subtract(*np.quantile(levels,[.75,.25]))/1.34)*n**(-.2), 1e-8)
grid, sigma2 = local_volatility(prices, dt=1.0)
tail = np.log(grid) >= np.quantile(np.log(grid), .6)
coef = np.polyfit(np.log(grid[tail]),np.log(sigma2[tail]),1)
eff = []
for g in grid:
    w = np.exp(-.5*((levels-g)/h)**2)
    eff.append(float(w.sum()**2/(w@w)))
report['volatility'] = {**volatility_elasticity(prices), 'bandwidth_USD': h,
    'grid_minmax_USD': [float(grid[0]),float(grid[-1])],
    'fitted_minmax_USD': [float(grid[tail][0]),float(grid[-1])],
    'effective_weight_count_fitted_minmax': [min(np.array(eff)[tail]),max(np.array(eff)[tail])],
    'dt_days': 1, 'increments': n}
fig, (ax, support_ax) = plt.subplots(2,1,figsize=(8,5.1),sharex=True,gridspec_kw={'height_ratios':[2.4,1]})
ax.plot(grid/1000,np.sqrt(sigma2),color='#17648b',lw=2,label='Kernel estimate')
ax.plot(grid[tail]/1000,np.exp(np.polyval(coef,np.log(grid[tail]))/2),color='#c06426',lw=2,
        label=r'Upper-grid fit: $\widehat\rho=0.123$')
ax.axvspan(grid[tail][0]/1000,grid[-1]/1000,color='#c06426',alpha=.08)
ax.set(ylabel='Daily volatility (USD)')
ax.legend(loc='upper left'); ax.grid(alpha=.2)
support_ax.plot(grid/1000,eff,color='#17648b')
support_ax.set(xlabel='Price level (thousand USD)',ylabel='Weight count')
support_ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(THESIS/'fig_final_volatility.png');plt.close(fig)

# Funding plateau, with both sign and magnitude visible.
p = np.linspace(-.12,.14,600)
f = p + np.clip(.01-p,-.05,.05)
fig,ax=plt.subplots(figsize=(7.6,3.6))
ax.plot(p,f,color='#17648b',lw=2.5)
ax.axvspan(-.04,.06,color='#17648b',alpha=.1,label='Premium values on the plateau')
ax.axhline(0,color='grey',lw=.7);ax.axvline(0,color='grey',lw=.7)
ax.scatter([-.04,0,.06],[.01]*3,color='#c06426',zorder=3)
ax.annotate('Zero premium, positive funding',xy=(0,.01),xytext=(-.108,.072),
            arrowprops={'arrowstyle':'->','color':'#444444'},fontsize=10)
ax.text(.009,-.02,'Funding = 0.01%',fontsize=10)
ax.set(xlabel='Average premium (%)',ylabel='Funding rate per interval (%)',xlim=(-.12,.14))
ax.set_xticks([-.12,-.08,-.04,0,.06,.10,.14]);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(THESIS/'fig_final_funding_plateau.png');plt.close(fig)

# Show cumulative mean differences, whose terminal value is the mean defect.
laws=[json.loads((PILOT/'2026-09-25'/name).read_text()) for name in ['witness_A_0.json','witness_A_4.json']]
xs = sorted(set([0.] + sum([law['support'] for law in laws], [])))
cum=[]
for law in laws:
    cum.append([math.fsum(x*w for x,w in zip(law['support'],law['probabilities']) if x<=g) for g in xs])
diff=100*(np.array(cum[0])-cum[1])
defect=100*laws[1]['target_defect']
upper=100*json.loads((PILOT/'2026-09-25/analysis.json').read_text())['bounds']['U_put']
fig,(ax,bax)=plt.subplots(1,2,figsize=(10,3.9),gridspec_kw={'width_ratios':[1.2,1]})
ax.step(xs,diff,where='post',lw=2,color='#17648b')
ax.axhline(0,color='grey',lw=.7)
ax.scatter(xs[-1],diff[-1],color='#c06426',zorder=3)
ax.annotate(f'Total mean difference\n{defect:.4f}% of reference',xy=(xs[-1],diff[-1]),
            xytext=(.70,.73),textcoords='axes fraction',ha='center',fontsize=10,
            arrowprops={'arrowstyle':'->','color':'#444444'})
ax.set(xlabel='Normalised terminal index X',ylabel='Cumulative mean difference (% of reference)',
       title='Where the two means differ',xlim=(0,1.72))
ax.grid(alpha=.2)
bax.barh([1,0],[defect,upper],color=['#17648b','#d8e0e5'],height=.3)
bax.scatter([0,defect],[1,1],color='#c06426',s=30,zorder=3)
bax.text(defect,1.25,f'{defect:.4f}%',ha='center')
bax.text(upper,.25,f'{upper:.4f}%',ha='center')
bax.set(yticks=[1,0],yticklabels=['Constructed\nfeasible range','Necessary\nouter range'],
        xlabel='Defect (% of reference)',title='Inner and outer ranges',xlim=(-.008,.255),ylim=(-.45,1.6))
bax.grid(axis='x',alpha=.2)
fig.tight_layout(w_pad=2);fig.savefig(THESIS/'fig_final_option_witness.png');plt.close(fig)
rows=[]
for x in xs[1:]:
    weights=[math.fsum(w for g,w in zip(law['support'],law['probabilities']) if g==x) for law in laws]
    rows.append(f'{x:.10f} & {weights[0]:.10f} & {weights[1]:.10f} '+r'\\')
(THESIS/'app_final_witness_rows.tex').write_text('\n'.join(rows)+'\n')
report['option_laws']=[{k:v for k,v in law.items() if k not in ['validation','attempts']} for law in laws]
report['input_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
    for folder in [ROOT/'data/raw',ROOT/'pilot/option_reliability/raw']
    for p in sorted(folder.rglob('*')) if p.is_file()}
(OUT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['input_sha256','option_laws','bybit_columns']},indent=2))
