"""⑩-3 GTX역 연계 후보 노선 목록 (2024-10-17 전국 TCN). 수집 요청용.

GTX-A 3개 역(운정중앙·킨텍스·대곡) 반경 1km 안 정류장에서 승차 또는 하차한 버스 구간을 노선별로 센다.
광역버스 24개 노선은 표시만 하고 따로 둔다. 노선 ID는 수집 자료의 rte_id와 같은 체계(41xxxxxx 등).
산출: 노선 ID, 역별 반경 1km/500m 승하차 건수, 126개 격자 정류장 승차 건수, 광역 24개 여부

실행 (k 폴더에서): python 건수기준_재분석/code/s10_3_feeder_route_list.py <TCN_20241017.parquet> <STTN_20241017.parquet>
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

TCN, STTN = sys.argv[1], sys.argv[2]
OUT = Path(_fb_path('건수기준_재분석/10_가설검증/tables'))
STATIONS = {"운정중앙": (37.71667, 126.72833), "킨텍스": None, "대곡": (37.631626, 126.811024)}
acc = pd.read_csv(glob.glob(_fb_path('data/processed/17_*'))[0])
for s in STATIONS:
    r = acc[acc["최근접 GTX-A역"] == s].iloc[0]
    STATIONS[s] = (r["GTX-A역 위도"], r["GTX-A역 경도"])


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371008.8 * 2 * np.arcsin(np.sqrt(a))


st = pq.read_table(STTN, columns=["정류장 ID", "정류장 명칭", "정류장 X 좌표", "정류장 Y 좌표", "시군구명"]).to_pandas()
st["정류장 ID"] = st["정류장 ID"].astype(str).str.replace(".0", "", regex=False)
st = st.drop_duplicates("정류장 ID").set_index("정류장 ID")
lat, lon = st["정류장 X 좌표"].astype(float), st["정류장 Y 좌표"].astype(float)
near = {}
for s, (la, lo) in STATIONS.items():
    dist = pd.Series(haversine(lat.values, lon.values, la, lo), index=st.index)
    near[s] = {r: set(dist[dist <= r].index) for r in (500, 1000)}
all_near = set().union(*[near[s][1000] for s in near])

cells = pd.read_csv(glob.glob(_fb_path('data/processed/16_*'))[0], dtype=str)
grid_stops = {sid for s in cells["2024년 정류장 ID"] for sid in re.findall(r"\d{6,}", s)}
card = pd.read_csv(_fb_path('gtx_a_seoul_bus_outputs/transport_card/gtx_a_transport_card_20241017_raw_with_coords_api_filled.csv'),
                   dtype=str, usecols=["rte_id", "query_route_no"])
METRO = dict(card.dropna().drop_duplicates("rte_id").values[:, ::-1])  # rte_id -> 번호

f = pq.ParquetFile(TCN)
legs = [[f"승차 역 {k} ID", f"하차 역 {k} ID", f"교통수단 구분 {k} 코드", f"노선 {k} ID"] for k in range(1, 11)]
parts = []
for batch in f.iter_batches(batch_size=1_000_000, columns=[c for l in legs for c in l]):
    b = batch.to_pandas()
    for on, off, mode, route in legs:
        m = b[mode] == "B"
        if not m.any():
            continue
        x = pd.DataFrame({"on": b.loc[m, on].astype(str).str.replace(".0", "", regex=False),
                          "off": b.loc[m, off].astype(str).str.replace(".0", "", regex=False),
                          "route": b.loc[m, route].astype(str).str.replace(".0", "", regex=False)})
        x = x[x.on.isin(all_near) | x.off.isin(all_near) | x.on.isin(grid_stops)]
        parts.append(x)
L = pd.concat(parts, ignore_index=True)

rows = []
for r, g in L.groupby("route"):
    row = {"노선 ID": r, "광역 24개 여부": r in METRO, "광역 노선번호": METRO.get(r, ""),
           "126격자 정류장 승차": int(g.on.isin(grid_stops).sum())}
    for s in STATIONS:
        for rad in (500, 1000):
            row[f"{s} {rad}m 승차"] = int(g.on.isin(near[s][rad]).sum())
            row[f"{s} {rad}m 하차"] = int(g.off.isin(near[s][rad]).sum())
    row["역 1km 승하차 합"] = sum(row[f"{s} 1000m 승차"] + row[f"{s} 1000m 하차"] for s in STATIONS)
    stops = pd.concat([g.on, g.off])
    row["주요 경유 시군구"] = "/".join(stops.map(st["시군구명"]).dropna().value_counts().head(3).index)
    rows.append(row)
R = pd.DataFrame(rows)
R = R[(R["역 1km 승하차 합"] > 0)].sort_values("역 1km 승하차 합", ascending=False)
R.to_csv(OUT / "H3a_GTX역연계_후보노선_TCN2024.csv", index=False, encoding="utf-8-sig")
print("후보 노선 수(역 1km 승하차 1건 이상):", len(R), "| 광역 24개 제외:", int((~R["광역 24개 여부"]).sum()))
for s in STATIONS:
    k = R[(~R["광역 24개 여부"]) & ((R[f"{s} 500m 승차"] + R[f"{s} 500m 하차"]) > 0)]
    print(s, "역 500m 경유 비광역 노선 수:", len(k))
print(R.head(30).to_string(index=False))
