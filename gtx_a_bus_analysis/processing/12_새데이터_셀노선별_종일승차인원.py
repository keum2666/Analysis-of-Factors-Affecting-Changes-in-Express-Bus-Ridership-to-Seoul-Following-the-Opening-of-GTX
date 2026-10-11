"""2024-10-17→2025-11-06 새 안정격자의 셀×노선 종일 집계자료를 만든다."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
PROCESSED = PROJECT / "data" / "processed"
CARD = ROOT / "gtx_a_seoul_bus_outputs" / "transport_card"

STABLE_PATH = PROCESSED / "16_새데이터_공급구조유지_250m격자.csv"
F24 = CARD / "gtx_a_transport_card_20241017_raw_with_coords_api_filled.csv"
F25 = CARD / "gtx_a_transport_card_20251106_raw_with_coords.csv"
OUTPUT = PROCESSED / "20_새데이터_셀노선별_종일승차인원.csv"


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def split_ids(value):
    return [
        item.strip()
        for item in str(value).split("|")
        if item.strip() and item.strip().lower() != "nan"
    ]


def station_map(stable, column):
    mapping = {}
    for _, row in stable.iterrows():
        for station_id in split_ids(row[column]):
            previous = mapping.get(station_id)
            if previous is not None and previous != int(row["셀 ID"]):
                raise ValueError(f"정류장 {station_id}가 셀 {previous}, {row['셀 ID']}에 중복 배정됨")
            mapping[station_id] = int(row["셀 ID"])
    return mapping


def prepare(path, mapping):
    frame = read_csv(path)
    frame["정류장키"] = frame["ride_sttn_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    frame["셀 ID"] = frame["정류장키"].map(mapping)
    frame["노선"] = frame["query_route_no"].astype(str).str.strip()
    frame["승차인원"] = pd.to_numeric(frame["utztn_nope"], errors="coerce").fillna(0)
    return frame[frame["셀 ID"].notna()].copy()


def haversine(lat1, lon1, lat2, lon2):
    radius = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    return 2 * radius * np.arcsin(
        np.sqrt(np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2)
    )


def main():
    stable = read_csv(STABLE_PATH)
    stable["셀 ID"] = pd.to_numeric(stable["셀 ID"], errors="raise").astype(int)
    if len(stable) != 126:
        raise ValueError(f"안정격자가 126개가 아님: {len(stable)}")

    d24 = prepare(F24, station_map(stable, "2024년 정류장 ID"))
    d25 = prepare(F25, station_map(stable, "2025년 정류장 ID"))
    keys = ["셀 ID", "노선"]

    y24 = d24.groupby(keys, as_index=False)["승차인원"].sum().rename(columns={"승차인원": "y2024"})
    y25 = d25.groupby(keys, as_index=False)["승차인원"].sum().rename(columns={"승차인원": "y2025"})
    pairs = y24.merge(y25, on=keys, how="outer").fillna({"y2024": 0, "y2025": 0})

    # 2024년 셀×노선별 GTX 경쟁 목적지 비율
    alight_lat = pd.to_numeric(d24["하차위도"], errors="coerce")
    alight_lon = pd.to_numeric(d24["하차경도"], errors="coerce")
    valid = alight_lat.notna() & alight_lon.notna()
    dist_yeonsinnae = haversine(alight_lat, alight_lon, 37.618855, 126.920859)
    dist_seoul = haversine(alight_lat, alight_lon, 37.555980, 126.972091)
    competitive = valid & ((dist_yeonsinnae <= 1) | (dist_seoul <= 1))
    d24["분류가능인원"] = np.where(valid, d24["승차인원"], 0)
    d24["경쟁목적인원"] = np.where(competitive, d24["승차인원"], 0)
    comp = d24.groupby(keys, as_index=False)[["분류가능인원", "경쟁목적인원"]].sum()
    comp["competing_share"] = np.where(
        comp["분류가능인원"] > 0,
        comp["경쟁목적인원"] / comp["분류가능인원"] * 100,
        0,
    )

    pairs = pairs.merge(comp[keys + ["competing_share"]], on=keys, how="left")
    pairs["competing_share"] = pairs["competing_share"].fillna(0)
    pairs = pairs.sort_values(keys).reset_index(drop=True)
    pairs.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print("저장:", OUTPUT)
    print("셀×노선:", len(pairs))
    print("셀:", pairs["셀 ID"].nunique())
    print("노선:", pairs["노선"].nunique())
    print("2024 승차>0:", int((pairs["y2024"] > 0).sum()))
    print("2025 승차=0:", int((pairs["y2025"] == 0).sum()))


if __name__ == "__main__":
    main()
