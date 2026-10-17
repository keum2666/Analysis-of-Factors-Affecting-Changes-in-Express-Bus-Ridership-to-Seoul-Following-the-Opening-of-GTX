"""S4b. 주모형에서 IC 도로거리(ic_access)만 제외하고 재추정, 7변수 주모형과 비교 (이용인원·건수 기준)."""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'tables' / 'S1_126격자_건수기준_기초표.csv'
OUT = ROOT / '04_OLS' / 'tables'

RESIDENTIAL = '주거지역 비율(%)'
FLOOR = '2024년 건축물 연면적(천㎡)'
FULL = ['log_walk', 'pop_20_50', 'log_floor', RESIDENTIAL, 'competing_share', 'bus_time_mean', 'ic_access']
DROP = [v for v in FULL if v != 'ic_access']
LABELS = {'log_walk': 'log(1+GTX역 보행시간)', 'pop_20_50': '20~50대 인구(명)',
          'log_floor': 'log(1+건축물 연면적)', RESIDENTIAL: RESIDENTIAL,
          'competing_share': 'GTX 경쟁 목적지 비율(%)', 'bus_time_mean': '평균 버스 통행시간(분)',
          'ic_access': 'IC 도로거리(km)'}

raw = pd.read_csv(SRC, encoding='utf-8-sig')
assert len(raw) == 126


def build(person):
    p = '[이용인원]' if person else ''
    d = raw[['walk_min', 'pop_20_50', FLOOR, RESIDENTIAL, 'ic_access']].copy()
    d['log_walk'] = np.log1p(d['walk_min'])
    d['log_floor'] = np.log1p(d[FLOOR])
    d['change'] = (raw[f'{p}board_2025_all' if person else 'board_2025']
                   - raw[f'{p}board_2024_all' if person else 'board_2024']).astype(float)
    d['competing_share'] = raw[f'{p}competing_share']
    d['bus_time_mean'] = raw[f'{p}bus_time_mean']
    return d


def fit(d, feats):
    X = sm.add_constant(d[feats].astype(float), has_constant='add')
    return sm.OLS(d['change'], X).fit(cov_type='HC3')


rows, fit_rows = [], []
for basis, person in [('이용인원', True), ('건수', False)]:
    d = build(person)
    f7, f6 = fit(d, FULL), fit(d, DROP)
    for v in FULL:
        rows.append({
            '기준': basis, '변수': LABELS[v],
            '7변수 계수': f7.params[v], '7변수 p값': f7.pvalues[v],
            '6변수 계수': f6.params.get(v, np.nan), '6변수 p값': f6.pvalues.get(v, np.nan),
        })
    for name, f in [('7변수(기존)', f7), ('6변수(IC 제외)', f6)]:
        fit_rows.append({'기준': basis, '모형': name, '표본 수': int(f.nobs), 'R²': f.rsquared,
                         '수정 R²': f.rsquared_adj, 'AIC': f.aic, 'BIC': f.bic})

coef = pd.DataFrame(rows)
fits = pd.DataFrame(fit_rows)
coef.to_csv(OUT / 'S4b_1_IC제외_계수비교.csv', index=False, encoding='utf-8-sig')
fits.to_csv(OUT / 'S4b_2_IC제외_적합도비교.csv', index=False, encoding='utf-8-sig')

pd.set_option('display.width', 250)
print(coef.round(4).to_string(index=False))
print()
print(fits.round(4).to_string(index=False))
