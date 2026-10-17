"""2024년 종일 GTX 경쟁 목적지 비율을 셀×노선 단위로 구축한다.

분자: 해당 셀×노선에서 승차하여 연신내역·서울역 1km 이내에 하차한 승객
분모: 해당 셀×노선에서 승차하여 서울에서 하차했고 하차 좌표가 확인되는 승객
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir())
sys.path.insert(0, str(PROJECT_DIR))

from config import (  # noqa: E402
    CELL_ROUTE_ALL_DAY_PATH,
    COMPETING_CELL_ROUTE_PATH,
    RAW_PATHS,
    SEOUL,
    is_competing_destination,
    read_csv,
    stop_to_cell_map,
)


def resolve_raw_2024() -> Path:
    """프로젝트 raw 폴더가 비어 있으면 UROP 원자료 위치를 사용한다."""
    configured = RAW_PATHS[2024]
    if configured.exists():
        return configured
    fallback = (
        PROJECT_DIR
        / "data" / "raw" / "gtx_a" / "transport_card"
        / "gtx_a_transport_card_20241017_raw_with_coords_api_filled.csv"
    )
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        "2024년 교통카드 원자료가 없습니다. data/raw/transport_card에 원자료를 넣어주세요."
    )


def main() -> None:
    pairs = read_csv(CELL_ROUTE_ALL_DAY_PATH)
    pairs["셀 ID"] = pd.to_numeric(pairs["셀 ID"], errors="coerce").astype("Int64")
    pairs["노선"] = pairs["노선"].astype(str).str.removesuffix(".0")
    pairs["y2024"] = pd.to_numeric(pairs["y2024"], errors="coerce").fillna(0)
    pairs = pairs.loc[pairs["y2024"] > 0, ["셀 ID", "노선", "y2024"]].copy()

    raw = read_csv(resolve_raw_2024())
    mapping = stop_to_cell_map()
    stop = raw["승차정류장ID"].astype(str).str.removesuffix(".0")
    trips = pd.DataFrame(
        {
            "셀 ID": stop.map(mapping),
            "노선": raw["query_route_no"].astype(str).str.removesuffix(".0"),
            "서울 하차": pd.to_numeric(raw["goff_ctpv_cd"], errors="coerce").eq(SEOUL),
            "하차위도": pd.to_numeric(raw["하차위도"], errors="coerce"),
            "하차경도": pd.to_numeric(raw["하차경도"], errors="coerce"),
            "승차인원": pd.to_numeric(raw["utztn_nope"], errors="coerce").fillna(0),
        }
    ).dropna(subset=["셀 ID"])
    trips["셀 ID"] = trips["셀 ID"].astype("Int64")
    trips = trips.merge(pairs[["셀 ID", "노선"]], on=["셀 ID", "노선"], how="inner")
    trips = trips.loc[trips["서울 하차"]].copy()
    trips["목적지 분류 가능"] = trips[["하차위도", "하차경도"]].notna().all(axis=1)
    trips["GTX 경쟁 목적지"] = False
    classified = trips["목적지 분류 가능"]
    trips.loc[classified, "GTX 경쟁 목적지"] = is_competing_destination(
        trips.loc[classified, "하차위도"], trips.loc[classified, "하차경도"]
    )

    keys = ["셀 ID", "노선"]
    total = trips.groupby(keys)["승차인원"].sum().rename("2024년 종일 전체 서울행 승차인원")
    classifiable = (
        trips.loc[trips["목적지 분류 가능"]]
        .groupby(keys)["승차인원"]
        .sum()
        .rename("2024년 종일 목적지 분류 가능 승차인원")
    )
    competing = (
        trips.loc[trips["GTX 경쟁 목적지"]]
        .groupby(keys)["승차인원"]
        .sum()
        .rename("2024년 종일 GTX 경쟁 목적지 승차인원")
    )

    result = pairs.merge(total, on=keys, how="left").merge(classifiable, on=keys, how="left").merge(competing, on=keys, how="left")
    count_cols = [
        "2024년 종일 전체 서울행 승차인원",
        "2024년 종일 목적지 분류 가능 승차인원",
        "2024년 종일 GTX 경쟁 목적지 승차인원",
    ]
    result[count_cols] = result[count_cols].fillna(0)
    denominator = result["2024년 종일 목적지 분류 가능 승차인원"]
    result["GTX 경쟁 목적지 승차인원 비율(%)"] = np.where(
        denominator > 0,
        result["2024년 종일 GTX 경쟁 목적지 승차인원"] / denominator * 100,
        np.nan,
    )
    result["GTX 경쟁 목적지 기준"] = "연신내역·서울역 하차 반경 1km 이내"
    result = result.sort_values(keys).reset_index(drop=True)
    result.to_csv(COMPETING_CELL_ROUTE_PATH, index=False, encoding="utf-8-sig")

    print("저장:", COMPETING_CELL_ROUTE_PATH)
    print("셀×노선:", len(result), "| 셀:", result["셀 ID"].nunique(), "| 노선:", result["노선"].nunique())
    print("비율 결측:", result["GTX 경쟁 목적지 승차인원 비율(%)"].isna().sum())
    print(result["GTX 경쟁 목적지 승차인원 비율(%)"].describe().round(2))


if __name__ == "__main__":
    main()
