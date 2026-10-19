"""⑩-8 정류장 이름 기준 GTX역행 마을·시내버스 승차를 챕터 1 OLS에 추가.

GTX역 하차 = 하차 정류장 이름에 '운정중앙역' · '킨텍스역' · '대곡역'이 들어간 경우 (승차 정류장이 역 정류장이면 제외)
승차 = 126개 격자 정류장 (연도별 격자-정류장 매핑, data/processed/16)
정류장 이름: 2024 STTN 원자료 + 국토교통부 정류장 API(2025-11-06) + 2025 신설 정류장(GBIS)

모형 (챕터 1 건수 기준과 같은 틀: 126격자, 종속 = 2025 − 2024 광역 승차, 7개 변수, HC3)
  M0 챕터 1 그대로
  M1 + 2025 GTX역행 마을버스 승차(건)          ← 주
  M2 + Δ GTX역행 마을버스 승차(건) (2025 − 2024) ← 점검
  M3 M1 + 역 고정효과(대곡·킨텍스)
주의: 마을버스 변수는 개통 후 값이라 "함께 나타났다"(연관)로만 해석한다.

실행 (k 폴더에서): python 건수기준_재분석/code/s10_8_station_name_ols.py <STTN_20241017.parquet>
"""

# UROP 통합: 실행 위치에 독립적인 경로 설정 (계산 방식은 원본 유지)
import sys as _fb_sys
from pathlib import Path as _FBPath
_fb_dir = _FBPath(__file__).resolve().parent
if str(_fb_dir) not in _fb_sys.path:
    _fb_sys.path.insert(0, str(_fb_dir))
from feeder_paths import transfer_path as _fb_path, station_path as _fb_station_path, korean_font as _fb_font

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

STTN = _fb_station_path(sys.argv)
RAW = Path(_fb_path('건수기준_재분석/10_가설검증/raw'))
STOPS = Path(_fb_path('건수기준_재분석/10_가설검증/stops'))
TAB = Path(_fb_path('건수기준_재분석/10_가설검증/tables'))
PAT = "운정중앙역|킨텍스역|대곡역"


def norm(s):
    return s.astype(str).str.replace(".0", "", regex=False).str.lstrip("0")


# 정류장 이름
a = pq.read_table(STTN, columns=["정류장 ID", "정류장 명칭"]).to_pandas()
a = pd.Series(a["정류장 명칭"].values, index=norm(a["정류장 ID"]))
b = pd.read_csv(STOPS / "BusStop_20251106_파주고양서울일부.csv", dtype=str)
b = pd.Series(b.sttn_nm.values, index=norm(b.sttn_id))
c = pd.read_csv(STOPS / "2025_신설정류장_좌표_GBIS.csv", dtype=str)
c = pd.Series(c["정류장명"].values, index=norm(c["정류장 ID"]))
name = pd.concat([c, b, a])
name = name[~name.index.duplicated()]
station_stops = set(name[name.str.contains(PAT, na=False)].index)

# 격자-정류장 매핑
cells = pd.read_csv(next(Path(_fb_path('data/processed')).glob("16_*")), dtype=str)
s2c = {y: {sid.lstrip("0"): int(cid) for cid, s in zip(cells["셀 ID"], cells[f"{y}년 정류장 ID"])
           for sid in re.findall(r"\d{6,}", s)} for y in ("2024", "2025")}

F = pd.concat([pd.read_csv(f, dtype=str, usecols=["opr_ymd", "rte_id", "ride_sttn_id", "goff_sttn_id"])
               for f in sorted(RAW.glob("경기_*.csv")) if f.stat().st_size > 10], ignore_index=True)
F["year"] = F.opr_ymd.str[:4]
F["on"], F["off"] = norm(F.ride_sttn_id), norm(F.goff_sttn_id)
F["cell"] = [s2c[y].get(o) for y, o in zip(F.year, F.on)]
F = F[F.cell.notna() & F.off.isin(station_stops) & ~F.on.isin(station_stops)].copy()
F["cell"] = F.cell.astype(int)
F["하차 정류장"] = F.off.map(name)

used = F.groupby(["year", "하차 정류장"]).size().unstack(0).fillna(0).astype(int)
used.reset_index().to_csv(TAB / "H8a_역이름_하차정류장별_건수.csv", index=False, encoding="utf-8-sig")

# 챕터 1 자료 (s4_ols_count_basis.py 의 B_건수와 같은 변수)
raw = pd.read_csv(_fb_path('건수기준_재분석/tables/S1_126격자_건수기준_기초표.csv'), encoding="utf-8-sig").set_index("셀 ID")
assert len(raw) == 126
d = pd.DataFrame(index=raw.index)
d["change"] = raw.board_2025 - raw.board_2024
d["log(1+GTX역 보행시간)"] = np.log1p(raw.walk_min)
d["20~50대 인구(명)"] = raw.pop_20_50
d["log(1+건축물 연면적)"] = np.log1p(raw["2024년 건축물 연면적(천㎡)"])
d["주거지역 비율(%)"] = raw["주거지역 비율(%)"]
d["GTX 경쟁 목적지 비율(%)"] = raw.competing_share
d["평균 버스 통행시간(분)"] = raw.bus_time_mean
d["IC 도로거리(km)"] = raw.ic_access
BASE = list(d.columns[1:])
for y in ("2024", "2025"):
    d[f"GTX역행 마을버스 {y}"] = F[F.year == y].groupby("cell").size().reindex(d.index).fillna(0)
d["2025 GTX역행 마을버스 승차(건)"] = d["GTX역행 마을버스 2025"]
d["Δ GTX역행 마을버스 승차(건)"] = d["GTX역행 마을버스 2025"] - d["GTX역행 마을버스 2024"]
d["대곡"] = (raw.nearest_station == "대곡").astype(float)
d["킨텍스"] = (raw.nearest_station == "킨텍스").astype(float)
d["권역"] = raw.nearest_station
d.reset_index().round(4).to_csv(TAB / "H8b_격자별_역이름_마을버스.csv", index=False, encoding="utf-8-sig")

SPECS = {"M0 챕터1": BASE,
         "M1 +2025 마을버스": BASE + ["2025 GTX역행 마을버스 승차(건)"],
         "M2 +Δ마을버스": BASE + ["Δ GTX역행 마을버스 승차(건)"],
         "M3 M1+역 고정효과": BASE + ["2025 GTX역행 마을버스 승차(건)", "대곡", "킨텍스"]}
rows, fits = [], []
for nm, xs in SPECS.items():
    m = sm.OLS(d["change"], sm.add_constant(d[xs])).fit(cov_type="HC3")
    for v in xs:
        rows.append({"모형": nm, "변수": v, "계수": m.params[v], "HC3 표준오차": m.bse[v], "p값": m.pvalues[v],
                     "유의(p<0.05)": "✅" if m.pvalues[v] < .05 else ""})
    fits.append({"모형": nm, "n": int(m.nobs), "R²": m.rsquared, "수정 R²": m.rsquared_adj})
R, Fi = pd.DataFrame(rows), pd.DataFrame(fits)
Xv = sm.add_constant(d[SPECS["M1 +2025 마을버스"]])
vif = pd.DataFrame({"변수": Xv.columns[1:], "VIF": [variance_inflation_factor(Xv.values, i)
                                                  for i in range(1, Xv.shape[1])]})
R.round(4).to_csv(TAB / "H8c_역이름_OLS_계수.csv", index=False, encoding="utf-8-sig")
Fi.round(4).to_csv(TAB / "H8d_역이름_OLS_적합도.csv", index=False, encoding="utf-8-sig")
vif.round(2).to_csv(TAB / "H8e_역이름_OLS_VIF.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 250)
print(used.to_string())
print(d.groupby("권역")[["GTX역행 마을버스 2024", "GTX역행 마을버스 2025"]].sum())
print("2025 1건 이상 격자:", int((d["GTX역행 마을버스 2025"] > 0).sum()))
print(R.round(3).to_string(index=False))
print(Fi.round(3).to_string(index=False))
print(vif.round(2).to_string(index=False))
