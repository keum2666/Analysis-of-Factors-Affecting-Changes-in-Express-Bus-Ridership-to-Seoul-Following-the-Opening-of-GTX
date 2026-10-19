"""⑩-6 연계버스(광역 24개 외 시내·마을버스) 개통 전후 변화와 광역버스 변화의 관계 (예비: 수집 완료 노선만).

자료: s10_4로 받은 교통카드 합성데이터 (경기, users_type_cd=01, ride_ctpv_cd=41), 2024-10-17 / 2025-11-06, 건수 기준
격자별 지표 (연도별 격자-정류장 매핑, data/processed/16)
  연계버스 승차      = 격자 정류장에서 탄 연계버스 승차 건수
  역행 연계버스 승차 = 그중 격자의 최근접 GTX역 반경 R 안에서 내린 건수 (승차는 반경 밖), R = 1km(주) / 500m
  광역 GTX방향 승차  = 광역 24개 노선 승차 중 연신내·서울역 1km 안 하차 (기존 정의)
점검: 2024 역행 연계버스 승차를 전국 원자료(TCN, H1a)와 비교해 수집 비율 확인
관계 (격자 126개, OLS HC3)
  ② Δ광역 GTX방향 승차(건) ~ Δ역행 연계버스 승차(건) + log(1+개통전 광역 건수)
     (광역 GTX방향이 많이 준 격자일수록 역행 연계버스가 늘었나: 기대 부호 −)
  ① 광역 변화율(%, 가중) ~ log(1+보행시간) × log(1+2025 역행 연계버스 승차) + log(1+개통전 건수)
     (멀어도 역행 연계버스가 많으면 광역이 더 줄었나: 상호작용 기대 부호 −)
  ④ 권역별 요약

실행 (k 폴더에서): python 건수기준_재분석/code/s10_6_feeder_analysis.py <STTN_20241017.parquet>
"""

# UROP 통합: 실행 위치에 독립적인 경로 설정 (계산 방식은 원본 유지)
import sys as _fb_sys
from pathlib import Path as _FBPath
_fb_dir = _FBPath(__file__).resolve().parent
if str(_fb_dir) not in _fb_sys.path:
    _fb_sys.path.insert(0, str(_fb_dir))
from feeder_paths import transfer_path as _fb_path, station_path as _fb_station_path, korean_font as _fb_font

import glob
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
TAB = Path(_fb_path('건수기준_재분석/10_가설검증/tables'))
CARD = Path(_fb_path('gtx_a_seoul_bus_outputs/transport_card'))
STATIONS = ["운정중앙", "킨텍스", "대곡"]
COMP = {"연신내": (37.619001, 126.921008), "서울역": (37.5547, 126.9707)}


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371008.8 * 2 * np.arcsin(np.sqrt(a))


def norm(s):
    return s.astype(str).str.replace(".0", "", regex=False).str.lstrip("0")


info = pd.read_csv(_fb_path('건수기준_재분석/04_상대유지/tables/F0_분석용_데이터_126격자.csv'), encoding="utf-8-sig").set_index("셀 ID")
acc = pd.read_csv(glob.glob(_fb_path('data/processed/17_*'))[0]).set_index("셀 ID")
cells = pd.read_csv(glob.glob(_fb_path('data/processed/16_*'))[0], dtype=str)
cells["셀 ID"] = cells["셀 ID"].astype(int)
s2c = {y: {sid.lstrip("0"): cid for cid, s in zip(cells["셀 ID"], cells[f"{y}년 정류장 ID"])
           for sid in re.findall(r"\d{6,}", s)} for y in ("2024", "2025")}
st = pq.read_table(STTN, columns=["정류장 ID", "정류장 X 좌표", "정류장 Y 좌표"]).to_pandas()
st["정류장 ID"] = norm(st["정류장 ID"])
st = st.drop_duplicates("정류장 ID").set_index("정류장 ID")
lat, lon = st["정류장 X 좌표"].astype(float), st["정류장 Y 좌표"].astype(float)
# 2024 정류장 목록에 없는 2025 신설 정류장(운정중앙역·킨텍스역5번출구 등) 좌표 보완:
# 국토교통부_버스정류장 API(15142032)로 이름·ARS 번호 확인 → GBIS에서 ARS로 좌표 조회 (10_가설검증/stops)
new = pd.read_csv(_fb_path('건수기준_재분석/10_가설검증/stops/2025_신설정류장_좌표_GBIS.csv'), dtype={"정류장 ID": str})
new["정류장 ID"] = new["정류장 ID"].str.lstrip("0")
new = new[~new["정류장 ID"].isin(lat.index)].set_index("정류장 ID")
lat, lon = pd.concat([lat, new["위도"]]), pd.concat([lon, new["경도"]])

files = sorted(RAW.glob("경기_*.csv"))
F = pd.concat([pd.read_csv(f, dtype=str, usecols=["opr_ymd", "rte_id", "ride_sttn_id", "goff_sttn_id"]) for f in files
               if f.stat().st_size > 10], ignore_index=True)
F["year"] = F.opr_ymd.str[:4]
F["on"], F["off"] = norm(F.ride_sttn_id), norm(F.goff_sttn_id)
F["cell"] = [s2c[y].get(o) for y, o in zip(F.year, F.on)]
F = F[F.cell.notna()].copy()
F["cell"] = F.cell.astype(int)
slat, slon = F.cell.map(acc["GTX-A역 위도"]).values, F.cell.map(acc["GTX-A역 경도"]).values
F["d_off"] = hav(F.off.map(lat).values, F.off.map(lon).values, slat, slon)
F["d_on"] = hav(F.on.map(lat).values, F.on.map(lon).values, slat, slon)
coord_miss = F.groupby("year").d_off.apply(lambda s: s.isna().mean()).round(4).to_dict()
done = {f.stem.split("_")[2] for f in files}

g = pd.DataFrame(index=info.index)
for y in ("2024", "2025"):
    Fy = F[F.year == y]
    g[f"연계버스 승차 {y}"] = Fy.groupby("cell").size()
    for r in (500, 1000):
        g[f"역행 연계버스 승차 {r}m {y}"] = Fy[(Fy.d_off <= r) & (Fy.d_on > r)].groupby("cell").size()
g = g.fillna(0)

# 광역 GTX방향 승차 (기존 정의)
def metro(path, y):
    s2 = {sid: cid for cid, s in zip(cells["셀 ID"], cells[f"{y}년 정류장 ID"]) for sid in re.findall(r"\d{6,}", s)}
    c = pd.read_csv(path, dtype=str, usecols=["승차정류장ID", "하차위도", "하차경도"])
    c["cell"] = c["승차정류장ID"].map(s2)
    c = c[c.cell.notna()].copy()
    la, lo = c["하차위도"].astype(float).values, c["하차경도"].astype(float).values
    comp = np.zeros(len(c), bool)
    for a, b in COMP.values():
        comp |= hav(la, lo, a, b) <= 1000
    return c.assign(cell=c.cell.astype(int), comp=comp)


m24 = metro(CARD / "gtx_a_transport_card_20241017_raw_with_coords_api_filled.csv", "2024")
m25 = metro(CARD / "gtx_a_transport_card_20251106_raw_with_coords.csv", "2025")
g["광역 GTX방향 2024"] = m24[m24.comp].groupby("cell").size().reindex(g.index).fillna(0)
g["광역 GTX방향 2025"] = m25[m25.comp].groupby("cell").size().reindex(g.index).fillna(0)
g["광역 2024"], g["광역 2025"] = info.board_2024, info.board_2025
g["권역"] = info.nearest_station
g["보행시간"] = info.walk_min
tcn = pd.read_csv(TAB / "H1a_격자별_역행연계버스.csv", encoding="utf-8-sig").set_index("셀 ID")
g["TCN 역행 연계버스 승차 1000m 2024"] = tcn["역행 연계버스 승차(1000m)"]
g.round(3).reset_index().to_csv(TAB / "H6a_격자별_연계버스_전후.csv", index=False, encoding="utf-8-sig")

# 점검: 수집 비율
cov = {"수집 노선 파일 수": len(files), "하차 좌표 없음 비율(연도별)": coord_miss,
       "2024 역행(1km) 합: API": int(g["역행 연계버스 승차 1000m 2024"].sum()),
       "2024 역행(1km) 합: TCN": int(g["TCN 역행 연계버스 승차 1000m 2024"].sum()),
       "격자별 API-TCN 순위상관(2024, 1km)": stats.spearmanr(g["역행 연계버스 승차 1000m 2024"],
                                                       g["TCN 역행 연계버스 승차 1000m 2024"])[0]}

# ④ 권역 요약
S = g.groupby("권역")[[c for c in g.columns if ("승차" in c or "광역" in c) and "TCN" not in c]].sum().reindex(STATIONS)
for r in (500, 1000):
    S[f"역행 연계버스 {r}m 변화율(%)"] = 100 * (S[f"역행 연계버스 승차 {r}m 2025"] / S[f"역행 연계버스 승차 {r}m 2024"] - 1)
S["연계버스 전체 변화율(%)"] = 100 * (S["연계버스 승차 2025"] / S["연계버스 승차 2024"] - 1)
S["광역 GTX방향 변화율(%)"] = 100 * (S["광역 GTX방향 2025"] / S["광역 GTX방향 2024"] - 1)
S["광역 전체 변화율(%)"] = 100 * (S["광역 2025"] / S["광역 2024"] - 1)
S.round(2).reset_index().to_csv(TAB / "H6b_권역별_연계버스_전후.csv", index=False, encoding="utf-8-sig")

# 관계
rows = []
g["Δ광역 GTX방향"] = g["광역 GTX방향 2025"] - g["광역 GTX방향 2024"]
g["Δ광역 전체"] = g["광역 2025"] - g["광역 2024"]
g["광역 변화율"] = 100 * (g["광역 2025"] / g["광역 2024"] - 1)
g["log(1+개통전 광역)"] = np.log1p(g["광역 2024"])
g["log(1+보행시간)c"] = np.log1p(g["보행시간"]) - np.log1p(g["보행시간"]).mean()
for r in (1000, 500):
    g["Δ역행 연계"] = g[f"역행 연계버스 승차 {r}m 2025"] - g[f"역행 연계버스 승차 {r}m 2024"]
    g["log(1+역행 연계 2025)c"] = np.log1p(g[f"역행 연계버스 승차 {r}m 2025"])
    g["log(1+역행 연계 2025)c"] -= g["log(1+역행 연계 2025)c"].mean()
    g["보행×역행연계"] = g["log(1+보행시간)c"] * g["log(1+역행 연계 2025)c"]
    for yv, xs, w in [("Δ광역 GTX방향", ["Δ역행 연계", "log(1+개통전 광역)"], None),
                      ("Δ광역 전체", ["Δ역행 연계", "log(1+개통전 광역)"], None),
                      ("광역 변화율", ["log(1+보행시간)c", "log(1+역행 연계 2025)c", "보행×역행연계", "log(1+개통전 광역)"],
                       g["광역 2024"])]:
        m = sm.WLS(g[yv], sm.add_constant(g[xs]), weights=w if w is not None else 1.0).fit(cov_type="HC3")
        for v in xs:
            rows.append({"반경": r, "종속변수": yv, "변수": v, "계수": m.params[v], "p": m.pvalues[v],
                         "R²": m.rsquared, "n": int(m.nobs)})
    rows.append({"반경": r, "종속변수": "순위상관 Δ광역 GTX방향 vs Δ역행 연계", "변수": "",
                 "계수": stats.spearmanr(g["Δ광역 GTX방향"], g["Δ역행 연계"])[0],
                 "p": stats.spearmanr(g["Δ광역 GTX방향"], g["Δ역행 연계"])[1]})
R = pd.DataFrame(rows)
R.round(4).to_csv(TAB / "H6c_연계버스_관계.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(cov)
print(S.round(1).T.to_string())
print(R.round(3).to_string(index=False))
