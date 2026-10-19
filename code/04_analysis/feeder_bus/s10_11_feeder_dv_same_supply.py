"""⑩-11 GTX역행 마을·시내버스 승차 변화 OLS (마을버스가 종속변수) + 같은 공급 격자 기준 순위상관 재계산.

종속 = Δ GTX역행 마을·시내버스 승차(건) = 격자 정류장 승차 → 역명 정류장(운정중앙역·킨텍스역·대곡역) 하차, 2025 − 2024
독립 = 챕터 1 개통 전 변수 7개 (보행시간·20~50대 인구·연면적·주거비율·경쟁 목적지·버스통행시간·IC)
표본
  운정중앙: 63격자 전부 (GTX역행 마을버스가 신설 노선으로 생겼으므로)
  대곡·킨텍스: 마을·시내버스 공급 구조가 2024·2025 같은 격자만 (광역버스 공급구조 판정과 같은 기준, S1)
    노선 구성 동일   — 두 날짜에 격자 정류장 승차기록이 있는 노선 집합이 같음
    정류장 구성 동일 — 노선별 승차 정류장이 대응됨 (이름 우선, 좌표 차이 100m 이내까지 허용)
    (광역 기준의 '양쪽 날짜 승차기록 존재' 조건은 쓰지 않음 → 두 날짜 모두 승차기록이 없는 격자도 유지로 본다.
     한쪽 날짜에만 기록이 있으면 노선 구성이 달라 제외된다)
  노선·정류장은 교통카드 승차기록에서 구한다 (실제 운행표가 아님)
모형 (HC3): O1 7개 변수 / O2 + log(1+개통 전 광역 승차) + 권역(대곡·킨텍스)
순위상관: 같은 표본에서 Δ광역 vs Δ GTX역행, 출근(06~09시) / 그 외

실행 (k 폴더에서): python 건수기준_재분석/code/s10_11_feeder_dv_same_supply.py <STTN_20241017.parquet>
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
from scipy import stats

STTN = _fb_station_path(sys.argv)
RAW = Path(_fb_path('건수기준_재분석/10_가설검증/raw'))
STOPS = Path(_fb_path('건수기준_재분석/10_가설검증/stops'))
TAB = Path(_fb_path('건수기준_재분석/10_가설검증/tables'))
CARD = Path(_fb_path('gtx_a_seoul_bus_outputs/transport_card'))
PAT = "운정중앙역|킨텍스역|대곡역"
STATIONS = ["운정중앙", "킨텍스", "대곡"]


def norm(s):
    return s.astype(str).str.replace(".0", "", regex=False).str.lstrip("0")


a = pq.read_table(STTN, columns=["정류장 ID", "정류장 명칭"]).to_pandas()
b = pd.read_csv(STOPS / "BusStop_20251106_파주고양서울일부.csv", dtype=str)
c = pd.read_csv(STOPS / "2025_신설정류장_좌표_GBIS.csv", dtype=str)
name = pd.concat([pd.Series(c["정류장명"].values, index=norm(c["정류장 ID"])),
                  pd.Series(b.sttn_nm.values, index=norm(b.sttn_id)),
                  pd.Series(a["정류장 명칭"].values, index=norm(a["정류장 ID"]))])
name = name[~name.index.duplicated()]
st = pq.read_table(STTN, columns=["정류장 ID", "정류장 X 좌표", "정류장 Y 좌표"]).to_pandas()
st.index = norm(st["정류장 ID"])
st = st[~st.index.duplicated()]
c.index = norm(c["정류장 ID"])
lat = pd.concat([c["위도"].astype(float), st["정류장 X 좌표"].astype(float)])
lon = pd.concat([c["경도"].astype(float), st["정류장 Y 좌표"].astype(float)])
lat, lon = lat[~lat.index.duplicated()], lon[~lon.index.duplicated()]
station_stops = set(name[name.str.contains(PAT, na=False)].index)

info = pd.read_csv(_fb_path('건수기준_재분석/tables/S1_126격자_건수기준_기초표.csv'), encoding="utf-8-sig").set_index("셀 ID")
cells = pd.read_csv(next(Path(_fb_path('data/processed')).glob("16_*")), dtype=str)
cells.index = cells["셀 ID"].astype(int)
assert (cells.reindex(info.index)["정류장 구성 동일"] == "1").all()
s2c = {y: {sid.lstrip("0"): int(cid) for cid, s in zip(cells.index, cells[f"{y}년 정류장 ID"])
           for sid in re.findall(r"\d{6,}", s)} for y in ("2024", "2025")}

F = pd.concat([pd.read_csv(f, dtype=str, usecols=["opr_ymd", "rte_id", "ride_sttn_id", "goff_sttn_id", "ride_dt"])
               for f in sorted(RAW.glob("경기_*.csv")) if f.stat().st_size > 10], ignore_index=True)
F["year"] = F.opr_ymd.str[:4]
F["on"], F["off"] = norm(F.ride_sttn_id), norm(F.goff_sttn_id)
F["cell"] = [s2c[y].get(o) for y, o in zip(F.year, F.on)]
F["cell_off"] = [s2c[y].get(o) for y, o in zip(F.year, F.off)]
F["출근"] = F.ride_dt.str[8:10].astype(int).between(6, 9)
F["gtx"] = F.off.isin(station_stops) & ~F.on.isin(station_stops)


Fg = F[F.cell.notna()].copy()
Fg["cell"] = Fg.cell.astype(int)


def stop_match(s, t):
    """정류장 s와 t가 같은 정류장인가: ID 같음 → 이름 같음 → 좌표 100m 이내"""
    if s == t or (s in name.index and t in name.index and name[s] == name[t]):
        return True
    if s in lat.index and t in lat.index:
        p1, l1, p2, l2 = map(np.radians, (lat[s], lon[s], lat[t], lon[t]))
        h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
        return 6371008.8 * 2 * np.arcsin(np.sqrt(h)) <= 100
    return False


def supply_same(cell):
    """노선 구성 동일 + 노선별 정류장 구성 대응 → (판정, 사유)"""
    x = Fg[Fg.cell == cell]
    r = {y: x[x.year == y].groupby("rte_id").on.agg(set).to_dict() for y in ("2024", "2025")}
    if not r["2024"] and not r["2025"]:
        return True, "유지(두 날짜 모두 승차기록 없음)"
    if set(r["2024"]) != set(r["2025"]):
        return False, "노선 구성 다름"
    for rid in r["2024"]:
        a24, a25 = r["2024"][rid], r["2025"][rid]
        if not (all(any(stop_match(s, t) for t in a25) for s in a24)
                and all(any(stop_match(t, s) for s in a24) for t in a25)):
            return False, "정류장 구성 다름"
    return True, "유지"


judge = pd.DataFrame([supply_same(cid) for cid in info.index], index=info.index, columns=["판정", "사유"])

d = pd.DataFrame(index=info.index)
d["권역"] = info.nearest_station
d["Δ광역"] = info.board_2025 - info.board_2024
G = Fg[Fg.gtx]
for lab, m in [("", slice(None)), (" 출근", G.출근), (" 그 외", ~G.출근)]:
    t = G[m].groupby(["cell", "year"]).size().unstack("year").reindex(info.index).fillna(0)
    d[f"Δ GTX역행{lab}"] = t.get("2025", 0) - t.get("2024", 0)
    d[f"GTX역행 2024{lab}"], d[f"GTX역행 2025{lab}"] = t.get("2024", 0), t.get("2025", 0)  # s10_12 주모형 종속 = 2025
card = pd.concat([pd.read_csv(CARD / f, dtype=str, usecols=["opr_ymd", "승차정류장ID", "ride_dt"]) for f in
                  ["gtx_a_transport_card_20241017_raw_with_coords_api_filled.csv",
                   "gtx_a_transport_card_20251106_raw_with_coords.csv"]], ignore_index=True)
card["year"] = card.opr_ymd.str[:4]
card["cell"] = [s2c[y].get(o) for y, o in zip(card.year, norm(card["승차정류장ID"]))]
card = card[card.cell.notna()].assign(cell=lambda x: x.cell.astype(int), 출근=lambda x: x.ride_dt.str[8:10].astype(int).between(6, 9))
for lab, m in [(" 출근", card.출근), (" 그 외", ~card.출근)]:
    t = card[m].groupby(["cell", "year"]).size().unstack("year").reindex(info.index).fillna(0)
    d[f"Δ광역{lab}"] = t["2025"] - t["2024"]
d["광역 2024"], d["광역 2025"] = info.board_2024, info.board_2025
d["마을·시내버스 공급구조 유지"] = judge.판정
d["판정 사유"] = judge.사유
d["log(1+GTX역 보행시간)"] = np.log1p(info.walk_min)
d["20~50대 인구(명)"] = info.pop_20_50
d["log(1+건축물 연면적)"] = np.log1p(info["2024년 건축물 연면적(천㎡)"])
d["주거지역 비율(%)"] = info["주거지역 비율(%)"]
d["GTX 경쟁 목적지 비율(%)"] = info.competing_share
d["평균 버스 통행시간(분)"] = info.bus_time_mean
d["IC 도로거리(km)"] = info.ic_access
d["log(1+개통 전 광역 승차)"] = np.log1p(info.board_2024)
d["대곡"] = (d.권역 == "대곡").astype(float)
d["킨텍스"] = (d.권역 == "킨텍스").astype(float)
d.reset_index().round(4).to_csv(TAB / "H11a_격자별_공급구조_GTX역행.csv", index=False, encoding="utf-8-sig")

BASE = ["log(1+GTX역 보행시간)", "20~50대 인구(명)", "log(1+건축물 연면적)", "주거지역 비율(%)",
        "GTX 경쟁 목적지 비율(%)", "평균 버스 통행시간(분)", "IC 도로거리(km)"]
SAMPLES = {
    "S0 전체 126": pd.Series(True, index=d.index),
    "S1 운정 전부 + 대곡·킨텍스 공급구조 유지": (d.권역 == "운정중앙") | d["마을·시내버스 공급구조 유지"],
}
rows, fits, cors, ns = [], [], [], []
for sn, keep in SAMPLES.items():
    x = d[keep]
    ns.append({"표본": sn, **x.권역.value_counts().reindex(STATIONS).fillna(0).astype(int).to_dict(), "합계": len(x)})
    for mn, xs in [("O1 7개 변수", BASE), ("O2 +개통전 광역·권역", BASE + ["log(1+개통 전 광역 승차)", "대곡", "킨텍스"])]:
        xs = [v for v in xs if x[v].nunique() > 1]
        m = sm.OLS(x["Δ GTX역행"], sm.add_constant(x[xs])).fit(cov_type="HC3")
        for v in xs:
            rows.append({"표본": sn, "모형": mn, "변수": v, "계수": m.params[v], "HC3 표준오차": m.bse[v],
                         "p값": m.pvalues[v], "유의(p<0.05)": "✅" if m.pvalues[v] < .05 else ""})
        fits.append({"표본": sn, "모형": mn, "n": int(m.nobs), "R²": m.rsquared, "수정 R²": m.rsquared_adj})
    for lab in ("", " 출근", " 그 외"):
        for st in STATIONS + ["전체"]:
            y = x if st == "전체" else x[x.권역 == st]
            r = stats.spearmanr(y[f"Δ광역{lab}" if lab else "Δ광역"], y[f"Δ GTX역행{lab}"])
            cors.append({"표본": sn, "시간대": lab.strip() or "종일", "권역": st, "격자 수": len(y),
                         "Δ광역 합": int(y[f"Δ광역{lab}" if lab else "Δ광역"].sum()),
                         "Δ GTX역행 합": int(y[f"Δ GTX역행{lab}"].sum()), "순위상관": r[0], "p값": r[1]})
R, Fi, C, N = map(pd.DataFrame, (rows, fits, cors, ns))
R.round(4).to_csv(TAB / "H11b_GTX역행_OLS_계수.csv", index=False, encoding="utf-8-sig")
Fi.round(4).to_csv(TAB / "H11c_GTX역행_OLS_적합도.csv", index=False, encoding="utf-8-sig")
C.round(4).to_csv(TAB / "H11d_같은공급_순위상관.csv", index=False, encoding="utf-8-sig")
N.to_csv(TAB / "H11e_표본구성.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 250)
print(pd.crosstab(d.권역, d["판정 사유"]))
print(N.to_string(index=False))
print(R.round(3).to_string(index=False))
print(Fi.round(3).to_string(index=False))
print(C[C.시간대 != "종일"].round(3).to_string(index=False))
