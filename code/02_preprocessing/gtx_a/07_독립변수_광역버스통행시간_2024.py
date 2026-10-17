"""2024년 종일 서울행 광역버스 통행시간을 셀×노선 단위로 구축한다.

산출 변수
- 승객가중 평균 통행시간(분)
- 승객가중 중앙값 통행시간(분): 회귀분석 주 변수
- 유효 승객 수와 원자료 행 수

승차·하차시각 차이가 0분 초과 240분 이하인 기록만 사용한다.
한 행에 여러 명이 기록될 수 있으므로 utztn_nope를 가중치로 사용한다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir())
sys.path.insert(0, str(PROJECT_DIR))
from config import (  # noqa: E402
    CELL_ROUTE_ALL_DAY_PATH,
    GYEONGGI,
    RAW_PATHS,
    TRAVEL_TIME_2024_PATH,
    TRAVEL_TIME_2024_CELL_PATH,
    read_csv,
    stop_to_cell_map,
)


def source_path() -> Path:
    """저장소 raw를 우선 사용하고, 기존 UROP 원자료 폴더를 대체 경로로 찾는다."""
    configured = RAW_PATHS[2024]
    if configured.exists():
        return configured
    fallback = PROJECT_DIR / "data" / "raw" / "gtx_a" / "transport_card" / configured.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"2024년 교통카드 원자료를 찾을 수 없습니다: {configured}")


def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    order = np.argsort(values.to_numpy())
    value = values.to_numpy(dtype=float)[order]
    weight = weights.to_numpy(dtype=float)[order]
    cutoff = weight.sum() / 2
    return float(value[np.searchsorted(np.cumsum(weight), cutoff, side="left")])


def main() -> None:
    raw = read_csv(source_path())
    required = {"query_route_no", "ride_sttn_id", "ride_dt", "goff_dt", "goff_ctpv_cd", "utztn_nope"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"원자료에 필요한 열이 없습니다: {sorted(missing)}")

    stop_map = stop_to_cell_map()
    route_pairs = read_csv(CELL_ROUTE_ALL_DAY_PATH)[["셀 ID", "노선"]].drop_duplicates()
    route_pairs["셀 ID"] = pd.to_numeric(route_pairs["셀 ID"], errors="coerce").astype("Int64")
    route_pairs["노선"] = route_pairs["노선"].astype(str).str.removesuffix(".0")

    ride = pd.to_datetime(raw["ride_dt"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
    alight = pd.to_datetime(raw["goff_dt"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
    trips = pd.DataFrame({
        "셀 ID": raw["ride_sttn_id"].astype(str).str.removesuffix(".0").map(stop_map),
        "노선": raw["query_route_no"].astype(str).str.removesuffix(".0"),
        "서울하차": pd.to_numeric(raw["goff_ctpv_cd"], errors="coerce").eq(11),
        "통행시간(분)": (alight - ride).dt.total_seconds() / 60,
        "승객 수": pd.to_numeric(raw["utztn_nope"], errors="coerce"),
    })
    trips = trips[
        trips["서울하차"]
        & trips["셀 ID"].notna()
        & trips["통행시간(분)"].gt(0)
        & trips["통행시간(분)"].le(240)
        & trips["승객 수"].gt(0)
    ].copy()
    trips["셀 ID"] = trips["셀 ID"].astype("Int64")
    trips = trips.merge(route_pairs, on=["셀 ID", "노선"], how="inner", validate="many_to_one")

    rows = []
    for (cell_id, route), group in trips.groupby(["셀 ID", "노선"], sort=True):
        weights = group["승객 수"]
        duration = group["통행시간(분)"]
        rows.append({
            "셀 ID": int(cell_id),
            "노선": route,
            "2024년 승객가중 평균 통행시간(분)": float(np.average(duration, weights=weights)),
            "2024년 승객가중 중앙값 통행시간(분)": weighted_median(duration, weights),
            "2024년 유효 통행 승객 수": float(weights.sum()),
            "2024년 유효 통행 원자료 행 수": int(len(group)),
        })

    result = pd.DataFrame(rows)
    result["2024년 중앙값 통행시간(10분 단위)"] = result["2024년 승객가중 중앙값 통행시간(분)"] / 10
    result.to_csv(TRAVEL_TIME_2024_PATH, index=False, encoding="utf-8-sig")

    # 셀 단위 분석용: 노선별 중앙값을 다시 평균하지 않고 원자료 전체에서 직접 계산한다.
    cell_rows = []
    for cell_id, group in trips.groupby("셀 ID", sort=True):
        weights = group["승객 수"]
        duration = group["통행시간(분)"]
        cell_rows.append({
            "셀 ID": int(cell_id),
            "2024년 셀 승객가중 평균 통행시간(분)": float(np.average(duration, weights=weights)),
            "2024년 셀 승객가중 중앙값 통행시간(분)": weighted_median(duration, weights),
            "2024년 셀 유효 통행 승객 수": float(weights.sum()),
            "2024년 셀 유효 통행 원자료 행 수": int(len(group)),
        })
    cell_result = pd.DataFrame(cell_rows)
    cell_result["2024년 셀 평균 통행시간(10분 단위)"] = (
        cell_result["2024년 셀 승객가중 평균 통행시간(분)"] / 10
    )
    cell_result["2024년 셀 중앙값 통행시간(10분 단위)"] = (
        cell_result["2024년 셀 승객가중 중앙값 통행시간(분)"] / 10
    )
    cell_result.to_csv(TRAVEL_TIME_2024_CELL_PATH, index=False, encoding="utf-8-sig")
    print(f"저장: {TRAVEL_TIME_2024_PATH}")
    print(f"셀×노선: {len(result)}개 / 셀: {result['셀 ID'].nunique()}개 / 노선: {result['노선'].nunique()}개")
    print(f"셀 단위 저장: {TRAVEL_TIME_2024_CELL_PATH} / {len(cell_result)}개")
    print(result.describe(include="all").to_string())


if __name__ == "__main__":
    main()
