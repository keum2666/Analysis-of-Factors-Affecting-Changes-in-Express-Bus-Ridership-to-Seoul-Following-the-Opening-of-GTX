"""⑩-1 가설 1 변수: 격자별 'GTX역으로 가는 연계버스(시내·마을버스)' 연결 (개통 전 2024-10-17 전국 교통카드 TCN).

정의
  연계버스 승차 = TCN 통행의 각 구간(1~10) 중
     교통수단 구분 = 'B'(버스), 노선이 GTX-A 관련 광역버스 24개 노선이 아님,
     승차 정류장이 126개 격자 정류장(2024 매핑), 하차 정류장이 그 격자의 최근접 GTX역에서 R 이내
  R = 500m (주), 300m·1000m (점검)  — GTX역 좌표는 data/processed/17 자료
  TCN은 마을버스·시내버스를 따로 구분하는 코드가 없어 '광역 24개 노선 외 버스'를 연계버스로 본다.
격자별 산출: 역행 연계버스 승차 건수, 역행 연계버스 노선 수, 격자 전체 연계버스(24개 외) 승차 건수

실행 (k 폴더에서): python 건수기준_재분석/code/s10_1_feeder_extract.py <TCN_20241017.parquet> <STTN_20241017.parquet>
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
OUT.mkdir(parents=True, exist_ok=True)
RADII = [300, 500, 1000]


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371008.8 * 2 * np.arcsin(np.sqrt(a))


acc = pd.read_csv(glob.glob(_fb_path('data/processed/17_*'))[0])
acc = acc.set_index("셀 ID")[["최근접 GTX-A역", "GTX-A역 위도", "GTX-A역 경도"]]
cells = pd.read_csv(glob.glob(_fb_path('data/processed/16_*'))[0], dtype=str)
cells["셀 ID"] = cells["셀 ID"].astype(int)
stop2cell = {sid: cid for cid, s in zip(cells["셀 ID"], cells["2024년 정류장 ID"]) for sid in re.findall(r"\d{6,}", s)}

card = pd.read_csv(_fb_path('gtx_a_seoul_bus_outputs/transport_card/gtx_a_transport_card_20241017_raw_with_coords_api_filled.csv'),
                   dtype=str, usecols=["rte_id"])
METRO = set(card.rte_id.dropna().unique())

st = pq.read_table(STTN, columns=["정류장 ID", "정류장 X 좌표", "정류장 Y 좌표"]).to_pandas()
st["정류장 ID"] = st["정류장 ID"].astype(str).str.replace(".0", "", regex=False)
st = st.drop_duplicates("정류장 ID").set_index("정류장 ID")
st_lat, st_lon = st["정류장 X 좌표"].astype(float), st["정류장 Y 좌표"].astype(float)   # STTN: X=위도, Y=경도

f = pq.ParquetFile(TCN)
legs = []
for k in range(1, 11):
    legs.append([f"승차 역 {k} ID", f"하차 역 {k} ID", f"교통수단 구분 {k} 코드", f"노선 {k} ID"])
cols = [c for l in legs for c in l]
rows = []
for batch in f.iter_batches(batch_size=1_000_000, columns=cols):
    b = batch.to_pandas()
    for on, off, mode, route in legs:
        s = b[on].astype(str).str.replace(".0", "", regex=False)
        m = s.isin(stop2cell.keys()) & (b[mode] == "B")
        if not m.any():
            continue
        x = pd.DataFrame({"on": s[m], "off": b.loc[m, off].astype(str).str.replace(".0", "", regex=False),
                          "route": b.loc[m, route].astype(str).str.replace(".0", "", regex=False)})
        rows.append(x)
L = pd.concat(rows, ignore_index=True)
L["cell"] = L.on.map(stop2cell).astype(int)
L["metro"] = L.route.isin(METRO)
L["off_lat"], L["off_lon"] = L.off.map(st_lat), L.off.map(st_lon)
L["d_station"] = haversine(L.off_lat.values, L.off_lon.values, L.cell.map(acc["GTX-A역 위도"]).values,
                           L.cell.map(acc["GTX-A역 경도"]).values)
L["on_d_station"] = haversine(L.on.map(st_lat).values, L.on.map(st_lon).values,
                              L.cell.map(acc["GTX-A역 위도"]).values, L.cell.map(acc["GTX-A역 경도"]).values)

check = {"버스 구간 수(126격자 승차)": len(L), "그중 광역 24개 노선": int(L.metro.sum()),
         "하차 좌표 없음(연계버스)": int(L[~L.metro].off_lat.isna().sum())}
F = L[~L.metro]
out = pd.DataFrame(index=sorted(set(cells["셀 ID"])))
out.index.name = "셀 ID"
out["연계버스 승차(전체)"] = F.groupby("cell").size()
out["연계버스 노선 수(전체)"] = F.groupby("cell").route.nunique()
for r in RADII:
    near = F[(F.d_station <= r) & (F.on_d_station > r)]   # 역 반경 밖에서 타서 역 반경 안에서 내림
    out[f"역행 연계버스 승차({r}m)"] = near.groupby("cell").size()
    out[f"역행 연계버스 노선 수({r}m)"] = near.groupby("cell").route.nunique()
out = out.fillna(0)
base = pd.read_csv(_fb_path('건수기준_재분석/tables/S1_126격자_건수기준_기초표.csv'), encoding="utf-8-sig")
out = out.reindex(base["셀 ID"]).fillna(0)
out.reset_index().to_csv(OUT / "H1a_격자별_역행연계버스.csv", index=False, encoding="utf-8-sig")
pd.Series(check).to_csv(OUT / "H1b_추출점검.csv", encoding="utf-8-sig")
print(check)
print(out.describe().round(1).T.to_string())
print(out.assign(st=base.set_index("셀 ID").nearest_station).groupby("st")[
    ["역행 연계버스 노선 수(500m)", "역행 연계버스 승차(500m)"]].agg(["mean", lambda s: (s > 0).mean()]).round(2))
