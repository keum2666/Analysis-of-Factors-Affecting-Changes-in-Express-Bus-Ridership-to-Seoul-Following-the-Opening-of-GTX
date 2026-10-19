"""⑩-12 GTX역행 마을·시내버스 분석 최종 정리 (126격자, 건수 기준, 하루치 비교).

입력: H11a (s10_11 — 격자별 GTX역행 2024·2025·Δ (출근/그 외), 광역 2024·2025·Δ, 마을버스 공급구조 판정), H6a (s10_6 — 마을·시내버스 승차 합),
      H14 (s10_14 — 20~50대 인구 2025)
(1) OLS (HC3) 종속 = 2025 GTX역행 마을·시내버스 승차(건) = 격자 정류장 승차 → 역명 정류장 하차 (개통 후 이용 수준)
    2024는 126격자 중 113개가 0건(운정은 역 정류장이 없었음)이라 변화량보다 개통 후 수준이 질문("개통 후 어디서 타나")에 맞음
    주모형: log(1+GTX역 보행시간), 20~50대 인구 2025, log(1+건축물 연면적), 주거지역 비율,
            1km 안 기존 철도역 수, 운정 더미(운정중앙 = 1, 대곡·킨텍스 = 0)
      인구만 2025: 종속이 2025 교통카드라 시점을 맞춤 (2024와 격자 순위상관 0.98). 나머지는 개통 전 값
      철도역: GTX 전용역(운정중앙·킨텍스) 제외 OSM 철도역 (05_변수확장 S5_후보변수의 n_rail_1km, 읽기만 함)
      운정 더미: 운정중앙역에는 환승센터가 있고(현장 답사) 연계 노선 신설·경로 변경도 운정에 몰림
    점검: ① 기존 4변수 (역 고정효과·철도역 없음)  ② 마을·시내버스 승차가 있는 91격자  ③ 마을·시내버스 공급구조 유지 92격자
          ④ 종속 = Δ (2025 − 2024): 개통 전부터 있던 역행 이용을 뺀 결과  ⑤ 인구 = 2024 (개통 전)
          ⑧ 운정 더미 대신 권역 더미 2개(대곡·킨텍스, 기준 운정)
(2) 순위상관 (Spearman): Δ광역 vs 2025 GTX역행, 종일 / 출근(06~09시) / 그 외 시간, 권역별
    주: 126격자 / 점검: 공급구조 유지 92격자. 두 변수 모두 개통 후 값이라 "함께 나타났나"(연관)만 본다
(3) 겹치는 격자: 2025 GTX역행 승차가 있는 격자(43) vs 없는 격자(83)의 광역버스 변화 비교 + 격자 목록 + 지도용 좌표표(H12j)
(4) 보행시간 구간별 분포(H12e) + 보행시간 더미 점검(H12f, 기준 30·40·45·50·60분 모두 제시)
    + 3구간 더미 점검(H12g, 가까움/중간/먼, 주 기준 20/50분 + 기준 바꾼 16가지)

실행 (k 폴더에서): python 건수기준_재분석/code/s10_12_feeder_final.py
"""

# UROP 통합: 실행 위치에 독립적인 경로 설정 (계산 방식은 원본 유지)
import sys as _fb_sys
from pathlib import Path as _FBPath
_fb_dir = _FBPath(__file__).resolve().parent
if str(_fb_dir) not in _fb_sys.path:
    _fb_sys.path.insert(0, str(_fb_dir))
from feeder_paths import transfer_path as _fb_path, station_path as _fb_station_path, korean_font as _fb_font

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

TAB = Path(_fb_path('건수기준_재분석/10_가설검증/tables'))
STATIONS = ["운정중앙", "킨텍스", "대곡"]
d = pd.read_csv(TAB / "H11a_격자별_공급구조_GTX역행.csv", encoding="utf-8-sig").set_index("셀 ID")
h6 = pd.read_csv(TAB / "H6a_격자별_연계버스_전후.csv", encoding="utf-8-sig").set_index("셀 ID").reindex(d.index)
info = pd.read_csv(_fb_path('건수기준_재분석/tables/S1_126격자_건수기준_기초표.csv'), encoding="utf-8-sig").set_index("셀 ID").reindex(d.index)
assert len(d) == 126
d["마을·시내버스 승차 있음"] = (h6["연계버스 승차 2024"] + h6["연계버스 승차 2025"]) > 0
d["공급구조 유지(운정 전부)"] = (d.권역 == "운정중앙") | d["마을·시내버스 공급구조 유지"]
pop = pd.read_csv(TAB / "H14_격자별_20~50대인구_2025.csv", encoding="utf-8-sig").set_index("셀 ID").reindex(d.index)
d["20~50대 인구 2025(명)"] = pop["20~50대 인구 2025(명)"]
osm = pd.read_csv(_fb_path('건수기준_재분석/05_변수확장/tables/S5_후보변수_126격자.csv'), encoding="utf-8-sig").set_index("셀 ID").reindex(d.index)
d["1km 안 기존 철도역 수"] = osm["n_rail_1km"]  # 0·1·2개뿐이라 log 없이 그대로
d["운정 더미"] = (d.권역 == "운정중앙").astype(int)

Y = "GTX역행 2025"
X4 = ["log(1+GTX역 보행시간)", "20~50대 인구 2025(명)", "log(1+건축물 연면적)", "주거지역 비율(%)"]
X = X4 + ["1km 안 기존 철도역 수", "운정 더미"]  # 보행시간이 첫 번째 (더미 점검에서 X[1:]를 통제로 씀)
X24 = [v.replace("20~50대 인구 2025(명)", "20~50대 인구(명)") for v in X]  # 점검⑤: 2024 인구
ALL = pd.Series(True, index=d.index)
SPECS = {"주모형 (126)": (ALL, X, Y),
         "점검① 기존 4변수 (126)": (ALL, X4, Y),
         "점검② 마을·시내버스 승차 있는 격자 (91)": (d["마을·시내버스 승차 있음"], X, Y),
         "점검③ 공급구조 유지 격자 (92)": (d["공급구조 유지(운정 전부)"], X, Y),
         "점검④ 종속 = Δ(2025−2024) (126)": (ALL, X, "Δ GTX역행"),
         "점검⑤ 인구 = 2024 (126)": (ALL, X24, Y),
         "점검⑧ 권역 더미 2개 대곡·킨텍스 (126)": (ALL, X4 + ["1km 안 기존 철도역 수", "대곡", "킨텍스"], Y)}
rows, fits = [], []
for nm, (keep, xs, y) in SPECS.items():
    x = d[keep]
    m = sm.OLS(x[y], sm.add_constant(x[xs])).fit(cov_type="HC3")
    for v in xs:
        rows.append({"모형": nm, "종속": y, "변수": v, "계수": m.params[v], "HC3 표준오차": m.bse[v], "p값": m.pvalues[v],
                     "유의(p<0.05)": "✅" if m.pvalues[v] < .05 else ""})
    fits.append({"모형": nm, "종속": y, "n": int(m.nobs), "R²": m.rsquared, "수정 R²": m.rsquared_adj,
                 "F": m.fvalue, "모형 p값": m.f_pvalue})
Xv = sm.add_constant(d[X])
vif = pd.DataFrame({"변수": X, "VIF": [variance_inflation_factor(Xv.values, i) for i in range(1, Xv.shape[1])]})

# (2) 순위상관: Δ광역 vs 2025 GTX역행
cors = []
for sn, keep in [("126격자", ALL), ("공급구조 유지 92격자", d["공급구조 유지(운정 전부)"])]:
    for lab, nm in (("", "종일"), (" 출근", "출근(06~09시)"), (" 그 외", "그 외 시간")):
        for st in ["전체"] + STATIONS:
            y = d[keep] if st == "전체" else d[keep & (d.권역 == st)]
            a, b = y[f"Δ광역{lab}"], y[f"GTX역행 2025{lab}"]
            r = stats.spearmanr(a, b) if b.nunique() > 1 else (np.nan, np.nan)
            cors.append({"표본": sn, "시간대": nm, "권역": st, "격자 수": len(y), "Δ광역 합": int(a.sum()),
                         "GTX역행 2025 합": int(b.sum()), "순위상관": r[0], "p값": r[1]})

# (3) 겹치는 격자: 2025 GTX역행 승차가 있는 격자 vs 없는 격자의 광역버스 변화
d["2025 GTX역행 있음"] = d[Y] > 0
d["보행시간(분)"] = info.walk_min
ov = []
for st in STATIONS + ["전체"]:
    for has in (True, False):
        x = d[(d["2025 GTX역행 있음"] == has) & ((d.권역 == st) if st != "전체" else True)]
        ov.append({"권역": st, "구분": "GTX역행 있음" if has else "GTX역행 없음", "격자 수": len(x),
                   "보행시간 중앙값(분)": x["보행시간(분)"].median(),
                   "광역 2024": int(x["광역 2024"].sum()), "광역 2025": int(x["광역 2025"].sum()),
                   "Δ광역": int(x["Δ광역"].sum()), "광역 변화율(%)": 100 * x["Δ광역"].sum() / x["광역 2024"].sum(),
                   "Δ광역 출근": int(x["Δ광역 출근"].sum()),
                   "GTX역행 2025": int(x[Y].sum()), "GTX역행 2025 출근": int(x[f"{Y} 출근"].sum()),
                   "GTX역행 출근 비율(%)": 100 * x[f"{Y} 출근"].sum() / x[Y].sum() if x[Y].sum() else np.nan})
OV = pd.DataFrame(ov)
LIST = d[d["2025 GTX역행 있음"]].sort_values(Y, ascending=False)[
    ["권역", "보행시간(분)", "광역 2024", "광역 2025", "Δ광역", "Δ광역 출근", "GTX역행 2024", Y, f"{Y} 출근"]].reset_index()
LIST["광역 변화율(%)"] = 100 * LIST["Δ광역"] / LIST["광역 2024"]

# 지도용 표 (지도는 카카오맵으로 따로 그림): 126격자 중심점 좌표 + 광역 변화 + 2025 GTX역행
MAP = d[["권역", "보행시간(분)", "광역 2024", "광역 2025", "Δ광역", "Δ광역 출근", Y, f"{Y} 출근", "2025 GTX역행 있음"]].copy()
MAP.insert(1, "중심점 위도", info["중심점 위도"])
MAP.insert(2, "중심점 경도", info["중심점 경도"])
MAP["광역 변화율(%)"] = 100 * MAP["Δ광역"] / MAP["광역 2024"]

# (4) 보행시간 구간별 분포(EDA) + 보행시간 더미 점검 (기준 30~60분 모두 제시, 주모형은 연속 log)
walk = np.expm1(d["log(1+GTX역 보행시간)"])
bins = pd.cut(walk, [0, 15, 20, 30, 40, 50, 60, np.inf], right=False,
              labels=["15분 미만", "15~20분", "20~30분", "30~40분", "40~50분", "50~60분", "60분 이상"])
WB = d.groupby(bins, observed=True)[Y].agg(격자수="size", 승차합계="sum", 격자당평균="mean", 승차있는격자=lambda s: int((s > 0).sum()))
for st in STATIONS:
    WB[f"{st} 합계"] = d[d.권역 == st].groupby(bins[d.권역 == st], observed=True)[Y].sum()
WB = WB.fillna(0).reset_index(names="보행시간")
dum = [("log 보행시간 (주모형)", d["log(1+GTX역 보행시간)"])] + [(f"{c}분 미만 = 1", (walk < c).astype(float)) for c in (30, 40, 45, 50, 60)]
DM = []
for nm, w in dum:
    m = sm.OLS(d[Y], sm.add_constant(pd.concat([w.rename("보행"), d[X[1:]]], axis=1))).fit(cov_type="HC3")
    DM.append({"보행시간 넣는 방식": nm, "계수": m.params["보행"], "HC3 표준오차": m.bse["보행"], "p값": m.pvalues["보행"],
               "수정 R²": m.rsquared_adj, "1인 격자 수": int(w.sum()) if "미만" in nm else np.nan})
# 3구간 더미 (가까움 / 중간 / 먼 = 기준). 주 기준 20/50분, 기준 시간을 바꾼 결과도 모두 제시
D3 = []
for a, b in [(a, b) for a in (15, 20, 25, 30) for b in (40, 45, 50, 60)]:
    near, mid = (walk < a).astype(float).rename("가까움"), ((walk >= a) & (walk < b)).astype(float).rename("중간")
    m = sm.OLS(d[Y], sm.add_constant(pd.concat([near, mid, d[X[1:]]], axis=1))).fit(cov_type="HC3")
    D3.append({"기준(가까움/먼, 분)": f"{a}/{b}", "주 기준": "◎" if (a, b) == (20, 50) else "",
               "가까움 격자": int(near.sum()), "중간 격자": int(mid.sum()), "먼 격자(기준)": int(len(d) - near.sum() - mid.sum()),
               "가까움 계수": m.params["가까움"], "가까움 p": m.pvalues["가까움"],
               "중간 계수": m.params["중간"], "중간 p": m.pvalues["중간"], "수정 R²": m.rsquared_adj})

R, Fi, C = pd.DataFrame(rows), pd.DataFrame(fits), pd.DataFrame(cors)
R.round(4).to_csv(TAB / "H12a_GTX역행마을버스_OLS_계수.csv", index=False, encoding="utf-8-sig")
Fi.round(4).to_csv(TAB / "H12b_GTX역행마을버스_OLS_적합도.csv", index=False, encoding="utf-8-sig")
vif.round(2).to_csv(TAB / "H12c_GTX역행마을버스_OLS_VIF.csv", index=False, encoding="utf-8-sig")
C.round(4).to_csv(TAB / "H12d_광역vsGTX역행_순위상관.csv", index=False, encoding="utf-8-sig")
WB.round(2).to_csv(TAB / "H12e_보행시간구간별_GTX역행.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(DM).round(4).to_csv(TAB / "H12f_보행시간_더미점검.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(D3).round(4).to_csv(TAB / "H12g_보행시간_3구간더미점검.csv", index=False, encoding="utf-8-sig")
OV.round(1).to_csv(TAB / "H12h_겹치는격자_광역비교.csv", index=False, encoding="utf-8-sig")
LIST.round(1).to_csv(TAB / "H12i_겹치는격자_목록.csv", index=False, encoding="utf-8-sig")
MAP.round(6).reset_index().to_csv(TAB / "H12j_지도용_126격자_좌표.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print("2025 0건 격자:", int((d[Y] == 0).sum()))
print(R.round(3).to_string(index=False))
print(Fi.round(3).to_string(index=False))
print(vif.round(2).to_string(index=False))
print(C.round(3).to_string(index=False))
print(OV.round(1).to_string(index=False))
print(LIST.head(15).round(1).to_string(index=False))
print(WB.round(1).to_string(index=False))
print(pd.DataFrame(DM).round(3).to_string(index=False))
print(pd.DataFrame(D3).round(3).to_string(index=False))
