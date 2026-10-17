"""2025-11-06 교통카드 자료에 정류장명·순서·위경도를 붙인다.

기준 자료는 이미 정류장 좌표 매칭이 완료된 2025-10-16 파일이다.
우선 `노선 ID + 정류장 ID`로 매칭하고, 미매칭 정류장은 정류장 ID의
유일한 이름·좌표로 보완한다.
"""

from pathlib import Path

import pandas as pd


ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir())
DATA_DIR = ROOT / "data" / "raw" / "gtx_a" / "transport_card"
RAW_PATH = DATA_DIR / "gtx_a_transport_card_20251106_raw.csv"
REFERENCE_PATHS = [
    DATA_DIR / "gtx_a_transport_card_20251016_raw_with_coords.csv",
    DATA_DIR / "gtx_a_transport_card_20241017_raw_with_coords_api_filled.csv",
]
OUTPUT_PATH = DATA_DIR / "gtx_a_transport_card_20251106_raw_with_coords.csv"
UNMATCHED_PATH = DATA_DIR / "gtx_a_transport_card_20251106_unmatched_stops.csv"

# STCIS 정류장 API 등으로 개별 확인한 수동 보완값
MANUAL_STOPS = {
    "4111073": {
        "정류장명": "일산복음병원.일산하이병원",
        "위도": 37.6821833,
        "경도": 126.78025,
    },
    "4124968": {
        "정류장명": "걸포북변역(걸포초교방면)",
        "위도": 37.6313833,
        "경도": 126.7045167,
    },
    "4111082": {
        "정류장명": "동국대병원사거리",
        "위도": 37.6731333,
        "경도": 126.8061667,
    },
    "4115898": {
        "정류장명": "농협하나로클럽",
        "위도": 37.7316833,
        "경도": 126.83495,
    },
    "4111224": {
        "정류장명": "행주동행정복지센터.능곡전화국",
        "위도": 37.6180167,
        "경도": 126.824366,
    },
    "4116350": {
        "정류장명": "산호아파트",
        "위도": 37.6291833,
        "경도": 126.7037,
    },
}

FRONT_COLUMNS = [
    "query_date", "query_route_no", "query_route_id",
    "승차정류장ID", "승차정류장명", "승차정류장순서", "승차위도", "승차경도",
    "하차정류장ID", "하차정류장명", "하차정류장순서", "하차위도", "하차경도",
]


def clean_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def route_stop_map(reference: pd.DataFrame, side: str) -> pd.DataFrame:
    prefix = "승차" if side == "ride" else "하차"
    stop_col = "ride_sttn_id" if side == "ride" else "goff_sttn_id"
    cols = {
        "query_route_id": "_route_id",
        stop_col: "_stop_id",
        f"{prefix}정류장명": f"{prefix}정류장명",
        f"{prefix}정류장순서": f"{prefix}정류장순서",
        f"{prefix}위도": f"{prefix}위도",
        f"{prefix}경도": f"{prefix}경도",
    }
    mapping = reference[list(cols)].rename(columns=cols).copy()
    mapping["_route_id"] = clean_id(mapping["_route_id"])
    mapping["_stop_id"] = clean_id(mapping["_stop_id"])
    return mapping.drop_duplicates(["_route_id", "_stop_id"])


def unique_stop_fallback(reference: pd.DataFrame, side: str) -> pd.DataFrame:
    prefix = "승차" if side == "ride" else "하차"
    stop_col = "ride_sttn_id" if side == "ride" else "goff_sttn_id"
    value_cols = [f"{prefix}정류장명", f"{prefix}위도", f"{prefix}경도"]
    fallback = reference[[stop_col] + value_cols].copy()
    fallback["_stop_id"] = clean_id(fallback[stop_col])
    fallback = fallback.drop(columns=stop_col).drop_duplicates()
    counts = fallback.groupby("_stop_id").size()
    unique_ids = counts[counts == 1].index
    fallback = fallback[fallback["_stop_id"].isin(unique_ids)].copy()
    return fallback.rename(columns={column: f"_fallback_{column}" for column in value_cols})


def attach_side(raw: pd.DataFrame, reference: pd.DataFrame, side: str) -> pd.DataFrame:
    prefix = "승차" if side == "ride" else "하차"
    stop_col = "ride_sttn_id" if side == "ride" else "goff_sttn_id"
    raw["_route_id"] = clean_id(raw["query_route_id"])
    raw["_stop_id"] = clean_id(raw[stop_col])
    raw = raw.merge(route_stop_map(reference, side), on=["_route_id", "_stop_id"], how="left")
    raw = raw.merge(unique_stop_fallback(reference, side), on="_stop_id", how="left")
    for suffix in ["정류장명", "위도", "경도"]:
        column = f"{prefix}{suffix}"
        raw[column] = raw[column].fillna(raw[f"_fallback_{column}"])
        raw = raw.drop(columns=f"_fallback_{column}")
    raw[f"{prefix}정류장ID"] = raw[stop_col]
    return raw.drop(columns=["_route_id", "_stop_id"])


def main() -> None:
    raw = pd.read_csv(RAW_PATH, dtype=str, encoding="utf-8-sig", low_memory=False)
    reference = pd.concat(
        [
            pd.read_csv(path, dtype=str, encoding="utf-8-sig", low_memory=False)
            for path in REFERENCE_PATHS
        ],
        ignore_index=True,
    )

    mapped = attach_side(raw.copy(), reference, "ride")
    mapped = attach_side(mapped, reference, "goff")

    for stop_id, values in MANUAL_STOPS.items():
        for prefix, id_column in [("승차", "ride_sttn_id"), ("하차", "goff_sttn_id")]:
            mask = clean_id(mapped[id_column]).eq(stop_id)
            mapped.loc[mask, f"{prefix}정류장명"] = mapped.loc[
                mask, f"{prefix}정류장명"
            ].fillna(values["정류장명"])
            mapped.loc[mask, f"{prefix}위도"] = mapped.loc[mask, f"{prefix}위도"].fillna(values["위도"])
            mapped.loc[mask, f"{prefix}경도"] = mapped.loc[mask, f"{prefix}경도"].fillna(values["경도"])

    remaining = [column for column in mapped.columns if column not in FRONT_COLUMNS]
    mapped = mapped[FRONT_COLUMNS + remaining]
    mapped.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    unmatched_frames = []
    for side, prefix, stop_col in [
        ("승차", "승차", "ride_sttn_id"),
        ("하차", "하차", "goff_sttn_id"),
    ]:
        missing = mapped[mapped[f"{prefix}위도"].isna()][
            ["query_route_no", "query_route_id", stop_col]
        ].drop_duplicates()
        missing.insert(0, "구분", side)
        missing = missing.rename(columns={stop_col: "정류장ID"})
        unmatched_frames.append(missing)
    unmatched = pd.concat(unmatched_frames, ignore_index=True)
    unmatched.to_csv(UNMATCHED_PATH, index=False, encoding="utf-8-sig")

    print(f"저장: {OUTPUT_PATH}")
    print(f"행 수: {len(mapped):,}")
    print(f"승차 좌표 미매칭 행: {mapped['승차위도'].isna().sum():,}")
    print(f"하차 좌표 미매칭 행: {mapped['하차위도'].isna().sum():,}")
    print(f"미매칭 정류장 목록: {UNMATCHED_PATH} ({len(unmatched):,}개)")


if __name__ == "__main__":
    main()
