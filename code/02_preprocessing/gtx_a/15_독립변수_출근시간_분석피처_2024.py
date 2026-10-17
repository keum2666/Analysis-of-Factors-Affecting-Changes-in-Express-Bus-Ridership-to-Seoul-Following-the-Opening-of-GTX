"""2024년 출근시간(06:00~09:59) 셀 단위 회귀 피처를 구축한다.

산출 변수
- GTX 경쟁 목적지 비율
- 승객가중 평균·중앙 통행시간
- 승객가중 평균·중앙 이동거리

분석대상은 새 교통카드 자료로 공급구조가 유지된 126개 격자이며,
서울에서 하차한 분석대상 광역버스 이용만 사용한다.
"""
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir())
ROOT = PROJECT
PROCESSED = PROJECT / "data" / "processed" / "gtx_a"
CARD_DIR = ROOT / "data" / "raw" / "gtx_a" / "transport_card"

STABLE_PATH = PROCESSED / "16_새데이터_공급구조유지_250m격자.csv"
PAIR_PATH = PROCESSED / "20_새데이터_셀노선별_종일승차인원.csv"
CARD_PATH = CARD_DIR / "gtx_a_transport_card_20241017_raw_with_coords_api_filled.csv"
OUTPUT_PATH = PROCESSED / "23_출근시간_분석피처_2024_250m격자.csv"

GTX_DESTINATIONS = [(37.618855, 126.920859), (37.555980, 126.972091)]
EARTH_RADIUS_M = 6_371_008.8


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def split_ids(value):
    return [
        item.strip().removesuffix(".0")
        for item in str(value).split("|")
        if item.strip() and item.strip().lower() != "nan"
    ]


def weighted_median(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    cutoff = weights[order].sum() / 2
    return float(values[order][np.searchsorted(np.cumsum(weights[order]), cutoff)])


def haversine_m(lat, lon, station_lat, station_lon):
    lat1, lon1 = np.radians(lat), np.radians(lon)
    lat2, lon2 = np.radians(station_lat), np.radians(station_lon)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def main():
    stable = read_csv(STABLE_PATH)
    stable["셀 ID"] = pd.to_numeric(stable["셀 ID"], errors="raise").astype(int)
    if stable["셀 ID"].nunique() != 126:
        raise ValueError(f"공급구조 유지 격자가 126개가 아님: {stable['셀 ID'].nunique()}")

    stop_to_cell = {}
    for _, row in stable.iterrows():
        for stop_id in split_ids(row["2024년 정류장 ID"]):
            stop_to_cell[stop_id] = int(row["셀 ID"])

    pairs = read_csv(PAIR_PATH)[["셀 ID", "노선"]].drop_duplicates()
    pairs["셀 ID"] = pd.to_numeric(pairs["셀 ID"], errors="coerce").astype("Int64")
    pairs["노선"] = pairs["노선"].astype(str).str.removesuffix(".0")

    raw = read_csv(CARD_PATH)
    ride = pd.to_datetime(raw["ride_dt"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
    alight = pd.to_datetime(raw["goff_dt"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
    frame = pd.DataFrame({
        "셀 ID": raw["ride_sttn_id"].astype(str).str.removesuffix(".0").map(stop_to_cell),
        "노선": raw["query_route_no"].astype(str).str.removesuffix(".0"),
        "승차시": ride.dt.hour,
        "서울하차": pd.to_numeric(raw["goff_ctpv_cd"], errors="coerce").eq(11),
        "승차인원": pd.to_numeric(raw["utztn_nope"], errors="coerce"),
        "통행시간(분)": (alight - ride).dt.total_seconds() / 60,
        "이동거리(m)": pd.to_numeric(raw["utztn_dstnc"], errors="coerce"),
        "하차위도": pd.to_numeric(raw["하차위도"], errors="coerce"),
        "하차경도": pd.to_numeric(raw["하차경도"], errors="coerce"),
    })
    frame = frame[
        frame["셀 ID"].notna()
        & frame["서울하차"]
        & frame["승차시"].between(6, 9, inclusive="both")
        & frame["승차인원"].gt(0)
    ].copy()
    frame["셀 ID"] = frame["셀 ID"].astype("Int64")
    frame = frame.merge(pairs, on=["셀 ID", "노선"], how="inner", validate="many_to_one")

    frame["목적지분류가능"] = frame[["하차위도", "하차경도"]].notna().all(axis=1)
    distances = np.vstack([
        haversine_m(frame["하차위도"], frame["하차경도"], lat, lon)
        for lat, lon in GTX_DESTINATIONS
    ])
    frame["경쟁목적지"] = frame["목적지분류가능"] & (np.nanmin(distances, axis=0) <= 1000)

    rows = []
    for cell_id, group in frame.groupby("셀 ID", sort=True):
        classified = group[group["목적지분류가능"]]
        denominator = classified["승차인원"].sum()
        competing = classified.loc[classified["경쟁목적지"], "승차인원"].sum()

        valid_time = group[group["통행시간(분)"].gt(0) & group["통행시간(분)"].le(240)]
        valid_distance = group[group["이동거리(m)"].gt(0) & group["이동거리(m)"].le(200_000)]
        row = {
            "셀 ID": int(cell_id),
            "2024년 출근시간 목적지 분류 가능 승차인원": float(denominator),
            "2024년 출근시간 GTX 경쟁 목적지 승차인원": float(competing),
            "2024년 출근시간 GTX 경쟁 목적지 비율(%)": float(competing / denominator * 100) if denominator > 0 else np.nan,
            "2024년 출근시간 승객가중 평균 통행시간(분)": np.nan,
            "2024년 출근시간 승객가중 중앙 통행시간(분)": np.nan,
            "2024년 출근시간 승객가중 평균 이동거리(km)": np.nan,
            "2024년 출근시간 승객가중 중앙 이동거리(km)": np.nan,
        }
        if not valid_time.empty:
            row["2024년 출근시간 승객가중 평균 통행시간(분)"] = float(np.average(valid_time["통행시간(분)"], weights=valid_time["승차인원"]))
            row["2024년 출근시간 승객가중 중앙 통행시간(분)"] = weighted_median(valid_time["통행시간(분)"], valid_time["승차인원"])
        if not valid_distance.empty:
            row["2024년 출근시간 승객가중 평균 이동거리(km)"] = float(np.average(valid_distance["이동거리(m)"], weights=valid_distance["승차인원"]) / 1000)
            row["2024년 출근시간 승객가중 중앙 이동거리(km)"] = weighted_median(valid_distance["이동거리(m)"], valid_distance["승차인원"]) / 1000
        rows.append(row)

    result = stable[["셀 ID"]].merge(pd.DataFrame(rows), on="셀 ID", how="left", validate="one_to_one")
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print("저장:", OUTPUT_PATH)
    print("격자 수:", len(result))
    print("변수별 결측 셀 수")
    print(result.isna().sum().to_string())
    missing = result[result.drop(columns="셀 ID").isna().any(axis=1)]["셀 ID"].tolist()
    print("결측 포함 셀 ID:", missing)


if __name__ == "__main__":
    main()
