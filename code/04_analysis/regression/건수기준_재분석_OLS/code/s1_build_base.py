"""①단계: 126개 격자 기초표를 승차 건수(행 수) 기준으로 다시 만든다.

- 교통카드에서 나온 변수만 다시 계산: 개통 전·후 승차, 변화율, 권역 변화율,
  경쟁 목적지 비율, 평균 버스통행시간.
- 먼저 기존 이용인원(utztn_nope) 가중 방식으로 기존 값을 재현해 계산 로직을 검증한 뒤,
  같은 로직에서 가중치만 1(건수)로 바꾼다.
- 교통카드와 무관한 변수(보행시간, 인구, 연면적, 주거지역 비율, IC)는 기존 값 그대로 사용.
- 기존 파일은 읽기만 한다.

실행 (k 폴더에서): python 건수기준_재분석/code/s1_build_base.py
"""
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

CARD = Path("gtx_a_seoul_bus_outputs/transport_card")
OUT = Path("건수기준_재분석/tables")
OUT.mkdir(parents=True, exist_ok=True)

# 경쟁 목적지: 연신내역·서울역 하차 반경 1km (기존 정의)
COMPETING = {"연신내역": (37.619001, 126.921008), "서울역": (37.5547, 126.9707)}  # 기존 126개 값을 모두 재현하는 좌표
RADIUS_KM = 1.0

base = pd.read_csv("GTX_A_126_EDA_필요데이터/GTX_A_126_분석데이터.csv", encoding="utf-8-sig")
cells = pd.read_csv(glob.glob("data/processed/16_*")[0], dtype=str)
cells["셀 ID"] = cells["셀 ID"].astype(int)
stops = {y: dict(zip(cells["셀 ID"], [re.findall(r"\d{6,}", s) for s in cells[f"{y}년 정류장 ID"]]))
         for y in ("2024", "2025")}

c24 = pd.read_csv(CARD / "gtx_a_transport_card_20241017_raw_with_coords_api_filled.csv", dtype=str)
c25 = pd.read_csv(CARD / "gtx_a_transport_card_20251106_raw_with_coords.csv", dtype=str)
for d in (c24, c25):
    d["nope"] = d.utztn_nope.astype(int)
    d["cnt"] = 1


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


lat = pd.to_numeric(c24["하차위도"], errors="coerce")
lon = pd.to_numeric(c24["하차경도"], errors="coerce")
c24["dest_known"] = lat.notna() & lon.notna()
c24["competing"] = False
for la, lo in COMPETING.values():
    c24["competing"] |= haversine(lat, lon, la, lo) <= RADIUS_KM
c24["bus_min"] = c24.brdg_hr.astype(float) / 60
c24["bus_valid"] = (c24.bus_min > 0) & (c24.bus_min <= 240)


def cell_vars(w):
    rows = []
    for cid in base["셀 ID"]:
        s24 = c24[c24["승차정류장ID"].isin(stops["2024"][cid])]
        s25 = c25[c25["승차정류장ID"].isin(stops["2025"][cid])]
        known = s24[s24.dest_known]
        valid = s24[s24.bus_valid]
        den = known[w].sum()
        rows.append({
            "셀 ID": cid,
            "board_2024": s24[w].sum(),
            "board_2025": s25[w].sum(),
            "competing_share": 100 * known.loc[known.competing, w].sum() / den if den > 0 else 0.0,
            "bus_time_mean": np.average(valid.bus_min, weights=valid[w]) if len(valid) else np.nan,
        })
    return pd.DataFrame(rows)


# --- 검증: 이용인원 가중으로 기존 값 재현
old = cell_vars("nope")
chk = base[["셀 ID", "board_2024_all", "board_2025_all", "competing_share", "bus_time_mean"]].merge(
    old, on="셀 ID", suffixes=("_기존", "_재계산"))
check = pd.DataFrame([{
    "변수": v,
    "최대절대차이": (chk[a] - chk[b]).abs().max(),
    "일치격자수(|차|<1e-6)": int(((chk[a] - chk[b]).abs() < 1e-6).sum()),
} for v, a, b in [
    ("개통전 승차", "board_2024_all", "board_2024"),
    ("개통후 승차", "board_2025_all", "board_2025"),
    ("경쟁 목적지 비율", "competing_share_기존", "competing_share_재계산"),
    ("평균 버스통행시간", "bus_time_mean_기존", "bus_time_mean_재계산"),
]])
check.to_csv(OUT / "S1a_기존값_재현검증.csv", index=False, encoding="utf-8-sig")
print(check.to_string())

# --- 건수 기준 기초표
new = cell_vars("cnt")
keep = ["셀 ID", "nearest_station", "중심점 위도", "중심점 경도", "walk_min", "pop_20_50",
        "2024년 건축물 연면적(천㎡)", "주거지역 비율(%)", "ic_access"]
t = base[keep].merge(new, on="셀 ID")
t["change"] = t.board_2025 - t.board_2024
t["change_rate_pct"] = 100 * t.change / t.board_2024
agg = t.groupby("nearest_station")[["board_2024", "board_2025"]].transform("sum")
t["station_change_rate_pct"] = 100 * (agg.board_2025 / agg.board_2024 - 1)
t["relative_change_pp"] = t.change_rate_pct - t.station_change_rate_pct
# 비교용: 기존 이용인원 기준 값 나란히
ref = base[["셀 ID", "board_2024_all", "board_2025_all", "change_rate_pct", "competing_share", "bus_time_mean"]]
ref.columns = ["셀 ID"] + [f"[이용인원]{c}" for c in ref.columns[1:]]
t = t.merge(ref, on="셀 ID")
assert len(t) == 126 and t[["board_2024", "board_2025", "change_rate_pct"]].notna().all().all()
t.to_csv(OUT / "S1_126격자_건수기준_기초표.csv", index=False, encoding="utf-8-sig")

pd.DataFrame([
    ["셀 ID ~ ic_access", "", "기존 GTX_A_126_분석데이터.csv 값 그대로 (교통카드와 무관)"],
    ["board_2024", "건", "2024-10-17 격자 정류장 승차 건수(교통카드 행 수)"],
    ["board_2025", "건", "2025-11-06 격자 정류장 승차 건수(교통카드 행 수)"],
    ["change", "건", "board_2025 - board_2024"],
    ["change_rate_pct", "%", "100*change/board_2024"],
    ["station_change_rate_pct", "%", "권역 합계 건수로 계산한 변화율"],
    ["relative_change_pp", "%p", "격자 변화율 - 권역 변화율"],
    ["competing_share", "%", "2024 하차좌표가 있는 승차 건수 중 연신내역·서울역 반경 1km 하차 건수 비율"],
    ["bus_time_mean", "분", "2024 유효 통행시간(0초과 240분 이하)의 건수 기준 평균"],
    ["[이용인원]…", "", "비교용: 기존 utztn_nope 합계/가중 기준 값"],
], columns=["변수", "단위", "설명"]).to_csv(OUT / "S1_변수설명.csv", index=False, encoding="utf-8-sig")

s = t.groupby("nearest_station")[["board_2024", "board_2025", "[이용인원]board_2024_all",
                                   "[이용인원]board_2025_all"]].sum()
s.loc["전체"] = s.sum()
s["건수_변화율(%)"] = 100 * (s.board_2025 / s.board_2024 - 1)
s["이용인원_변화율(%)"] = 100 * (s["[이용인원]board_2025_all"] / s["[이용인원]board_2024_all"] - 1)
s.round(2).reset_index().to_csv(OUT / "S1b_권역별_합계비교.csv", index=False, encoding="utf-8-sig")
print(s.round(1).to_string())
print(t[["change_rate_pct", "[이용인원]change_rate_pct", "competing_share", "[이용인원]competing_share",
         "bus_time_mean", "[이용인원]bus_time_mean"]].describe().round(2).to_string())
