"""S4. 기존 11-9 부분 로그변환 OLS를 승차 건수 기준으로 재실행하고 이용인원 기준과 비교한다.

- 기존 노트북: 11-9_부분로그변환_최종모형_OLS.ipynb (수정하지 않음)
- 모형 설정은 기존과 동일: 126개 격자, 7개 설명변수, 보행시간·연면적 log(1+x), 상수항, HC3
- 바꾸는 것: 종속변수(2025-2024 변화량), competing_share, bus_time_mean 의 집계 기준
  A. 이용인원 기준 (기존 재현)    : utztn_nope 합/가중
  B. 건수 기준 (주결과)           : 종속변수 + 교통카드 설명변수 2개 모두 건수
  C. 종속변수만 건수 (참고)       : 설명변수 2개는 이용인원 기준 유지
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor

ROOT = Path(__file__).resolve().parents[2]          # k/건수기준_재분석
SRC = ROOT / 'tables' / 'S1_126격자_건수기준_기초표.csv'
OUT = ROOT / '04_OLS' / 'tables'
OUT.mkdir(parents=True, exist_ok=True)

ID = '셀 ID'
RESIDENTIAL = '주거지역 비율(%)'
FLOOR = '2024년 건축물 연면적(천㎡)'
FINAL_FEATURES = ['log_walk', 'pop_20_50', 'log_floor', RESIDENTIAL,
                  'competing_share', 'bus_time_mean', 'ic_access']
LABELS = {
    'log_walk': 'log(1+GTX역 보행시간)', 'pop_20_50': '20~50대 인구(명)',
    'log_floor': 'log(1+건축물 연면적)', RESIDENTIAL: RESIDENTIAL,
    'competing_share': 'GTX 경쟁 목적지 비율(%)', 'bus_time_mean': '평균 버스 통행시간(분)',
    'ic_access': 'IC 도로거리(km)', 'log_pre_boarding': 'log(1+2024년 승차)',
    'board_2024': '2024년 승차',
}

# 기존 노트북 출력값 (재현 검증용)
ORIGINAL = {
    'R2': 0.3462, 'adjR2': 0.3074,
    'coef': {'log_walk': 148.8107, 'pop_20_50': 0.0370, 'log_floor': -72.1137,
             RESIDENTIAL: 1.6974, 'competing_share': -1.1164,
             'bus_time_mean': -0.1797, 'ic_access': -3.2943},
}

raw = pd.read_csv(SRC, encoding='utf-8-sig')
assert len(raw) == 126 and raw[ID].is_unique

common = raw[[ID, 'nearest_station', 'walk_min', 'pop_20_50', FLOOR, RESIDENTIAL, 'ic_access']].copy()
common['log_walk'] = np.log1p(common['walk_min'])
common['log_floor'] = np.log1p(common[FLOOR])


def build(dv_basis, x_basis):
    d = common.copy()
    pre = '[이용인원]' if dv_basis == 'person' else ''
    b24 = raw[f'{pre}board_2024_all' if pre else 'board_2024']
    b25 = raw[f'{pre}board_2025_all' if pre else 'board_2025']
    d['board_2024'], d['board_2025'] = b24.astype(float), b25.astype(float)
    d['change'] = d['board_2025'] - d['board_2024']
    xp = '[이용인원]' if x_basis == 'person' else ''
    d['competing_share'] = raw[f'{xp}competing_share']
    d['bus_time_mean'] = raw[f'{xp}bus_time_mean']
    return d


SPECS = {
    'A_이용인원': build('person', 'person'),
    'B_건수': build('count', 'count'),
    'C_종속변수만건수': build('count', 'person'),
}

# 결측·무한대 확인 (자동 제외하지 않고 보고)
check_rows = []
for name, d in SPECS.items():
    cols = ['change', 'board_2024', 'board_2025'] + FINAL_FEATURES
    vals = d[cols].astype(float)
    check_rows.append({'모형': name, '행 수': len(d), '결측 수': int(vals.isna().sum().sum()),
                       '무한대 수': int(np.isinf(vals).sum().sum())})
check = pd.DataFrame(check_rows)
assert (check['결측 수'] == 0).all() and (check['무한대 수'] == 0).all(), check


def ols(y, X_df):
    X = sm.add_constant(X_df.astype(float), has_constant='add')
    return sm.OLS(y.astype(float), X).fit(cov_type='HC3'), X


def stars(p):
    return '***' if p < .01 else '**' if p < .05 else '*' if p < .10 else ''


fits, coef_rows, model_rows, diag_rows, vif_rows, rob_rows, infl_rows = {}, [], [], [], [], [], []
for name, d in SPECS.items():
    fit, X = ols(d['change'], d[FINAL_FEATURES])
    fits[name] = fit
    sd_y = d['change'].std()
    ci = fit.conf_int()
    for v in FINAL_FEATURES:
        coef_rows.append({
            '모형': name, '변수': LABELS[v], 'key': v, '계수': fit.params[v], 'HC3 표준오차': fit.bse[v],
            'z값': fit.tvalues[v], 'p값': fit.pvalues[v], '95% CI 하한': ci.loc[v, 0],
            '95% CI 상한': ci.loc[v, 1], '유의성': stars(fit.pvalues[v]),
            '표준화 계수': fit.params[v] * d[v].std() / sd_y,
        })
    model_rows.append({'모형': name, '표본 수': int(fit.nobs), 'R²': fit.rsquared,
                       '수정 R²': fit.rsquared_adj, '모형 F값': float(fit.fvalue),
                       '모형 p값': float(fit.f_pvalue), 'AIC': fit.aic, 'BIC': fit.bic,
                       '종속변수 평균': d['change'].mean(), '종속변수 SD': sd_y})

    # 진단
    plain = sm.OLS(d['change'].astype(float), X).fit()
    bp = het_breuschpagan(plain.resid, X)
    jb = jarque_bera(plain.resid)
    reset = linear_reset(plain, power=2, use_f=True)
    diag_rows += [
        {'모형': name, '검정': 'Breusch–Pagan', '통계량': bp[0], 'p값': bp[1]},
        {'모형': name, '검정': 'Jarque–Bera', '통계량': jb.statistic, 'p값': jb.pvalue},
        {'모형': name, '검정': 'Ramsey RESET', '통계량': float(reset.fvalue), 'p값': float(reset.pvalue)},
    ]
    for v in FINAL_FEATURES:
        vif_rows.append({'모형': name, '변수': LABELS[v],
                         'VIF': variance_inflation_factor(X.to_numpy(), X.columns.get_loc(v))})

    # 영향력 격자 (Cook > 4/N)
    cook = OLSInfluence(plain).cooks_distance[0]
    thr = 4 / len(d)
    flagged = d.loc[cook > thr, [ID, 'nearest_station', 'board_2024', 'board_2025', 'change']].copy()
    flagged['적합값'] = plain.fittedvalues[cook > thr]
    flagged['잔차'] = plain.resid[cook > thr]
    flagged["Cook's distance"] = cook[cook > thr]
    flagged.insert(0, '모형', name)
    infl_rows.append(flagged.sort_values("Cook's distance", ascending=False))

    # 강건성 분석 (기존 노트북 8·9·12·14·15절과 동일 구성)
    def add_rob(label, f, extra=None):
        row = {'모형': name, '분석': label, '표본 수': int(f.nobs), 'R²': f.rsquared,
               '수정 R²': f.rsquared_adj, 'AIC': f.aic}
        for v in FINAL_FEATURES + ([extra] if extra else []):
            row[f'{LABELS[v]} 계수'] = f.params[v]
            row[f'{LABELS[v]} p값'] = f.pvalues[v]
        rob_rows.append(row)

    add_rob('주모형', fit)
    dummies = pd.get_dummies(d['nearest_station'], prefix='역', drop_first=True, dtype=float)
    f_fe, _ = ols(d['change'], pd.concat([d[FINAL_FEATURES], dummies], axis=1))
    add_rob('+역 고정효과', f_fe)
    for st in sorted(d['nearest_station'].unique()):
        sub = d[d['nearest_station'] != st]
        f_lo, _ = ols(sub['change'], sub[FINAL_FEATURES])
        add_rob(f'{st} 제외', f_lo)
    d2 = d.assign(log_pre_boarding=np.log1p(d['board_2024']))
    f_bl, _ = ols(d2['change'], d2[FINAL_FEATURES + ['log_pre_boarding']])
    add_rob('+log(1+2024 승차)', f_bl, 'log_pre_boarding')
    f_an, _ = ols(d['board_2025'], d[FINAL_FEATURES + ['board_2024']])
    add_rob('ANCOVA(2025 승차)', f_an, 'board_2024')
    f_lan, _ = ols(np.log1p(d2['board_2025']), d2[FINAL_FEATURES + ['log_pre_boarding']])
    add_rob('로그 ANCOVA', f_lan, 'log_pre_boarding')
    keep = d[~(cook > thr)]
    f_red, _ = ols(keep['change'], keep[FINAL_FEATURES])
    add_rob(f'영향력 {int((cook > thr).sum())}개 제외', f_red)

coef = pd.DataFrame(coef_rows)
models = pd.DataFrame(model_rows)

# 기존 노트북 재현 검증
a = fits['A_이용인원']
repro = pd.DataFrame([{'항목': LABELS[k], '기존 노트북': v, '재현값': round(a.params[k], 4),
                       '차이': round(a.params[k] - v, 4)} for k, v in ORIGINAL['coef'].items()]
                     + [{'항목': 'R²', '기존 노트북': ORIGINAL['R2'], '재현값': round(a.rsquared, 4),
                         '차이': round(a.rsquared - ORIGINAL['R2'], 4)},
                        {'항목': '수정 R²', '기존 노트북': ORIGINAL['adjR2'], '재현값': round(a.rsquared_adj, 4),
                         '차이': round(a.rsquared_adj - ORIGINAL['adjR2'], 4)}])
repro_ok = bool((repro['차이'].abs() <= 1e-4).all())

# 이용인원(A) vs 건수(B, C) 비교표
comp_rows = []
for v in FINAL_FEATURES:
    ra = coef[(coef['모형'] == 'A_이용인원') & (coef['key'] == v)].iloc[0]
    row = {'변수': LABELS[v], '이용인원 계수': ra['계수'], '이용인원 p값': ra['p값'],
           '이용인원 유의성': ra['유의성'], '이용인원 표준화계수': ra['표준화 계수']}
    for tag, nm in [('건수', 'B_건수'), ('종속만건수', 'C_종속변수만건수')]:
        rb = coef[(coef['모형'] == nm) & (coef['key'] == v)].iloc[0]
        row[f'{tag} 계수'] = rb['계수']
        row[f'{tag} p값'] = rb['p값']
        row[f'{tag} 유의성'] = rb['유의성']
        row[f'{tag} 표준화계수'] = rb['표준화 계수']
        row[f'{tag} 부호 일치'] = np.sign(ra['계수']) == np.sign(rb['계수'])
        sa, sb = ra['p값'] < .05, rb['p값'] < .05
        row[f'{tag} 5% 유의성 변화'] = '유지(유의)' if sa and sb else '유지(비유의)' if not sa and not sb \
            else '유의→비유의' if sa else '비유의→유의'
    comp_rows.append(row)
comp = pd.DataFrame(comp_rows)

coef = coef.drop(columns='key')
files = {
    'S4_0_자료점검.csv': check,
    'S4_1_기존결과_재현검증.csv': repro,
    'S4_2_계수표_전체모형.csv': coef,
    'S4_3_모형적합도.csv': models,
    'S4_4_이용인원vs건수_비교.csv': comp,
    'S4_5_진단검정.csv': pd.DataFrame(diag_rows),
    'S4_6_VIF.csv': pd.DataFrame(vif_rows),
    'S4_7_강건성분석.csv': pd.DataFrame(rob_rows),
    'S4_8_영향력격자.csv': pd.concat(infl_rows, ignore_index=True),
}
for fn, df in files.items():
    df.to_csv(OUT / fn, index=False, encoding='utf-8-sig')

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', None)
print('재현 성공:', repro_ok)
print(repro.to_string(index=False))
print()
print(models.round(4).to_string(index=False))
print()
print(comp.round(4).to_string(index=False))
