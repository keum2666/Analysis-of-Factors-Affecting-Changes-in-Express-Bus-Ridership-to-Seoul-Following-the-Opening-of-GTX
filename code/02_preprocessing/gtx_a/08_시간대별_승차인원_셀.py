"""2024·2025년 서울행 광역버스 승차인원을 셀×시간대 단위로 집계한다."""
from pathlib import Path
import sys

import pandas as pd

PROJECT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir())
sys.path.insert(0, str(PROJECT_DIR))

from config import CELL_ROUTE_ALL_DAY_PATH, RAW_PATHS, read_csv, stop_to_cell_map

OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "gtx_a" / "14_시간대별_승차인원_셀.csv"
TIME_LABELS = ["새벽(00~05)", "출근(06~09)", "낮(10~15)", "저녁(16~23)"]


def resolve_raw(year):
    if RAW_PATHS[year].exists():
        return RAW_PATHS[year]
    filename = RAW_PATHS[year].name
    fallback = PROJECT_DIR / "data" / "raw" / "gtx_a" / "transport_card" / filename
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"{year}년 교통카드 원자료를 찾지 못했습니다: {filename}")


def time_band(hour):
    if hour <= 5:
        return TIME_LABELS[0]
    if hour <= 9:
        return TIME_LABELS[1]
    if hour <= 15:
        return TIME_LABELS[2]
    return TIME_LABELS[3]


def main():
    mapping = stop_to_cell_map()
    valid_pairs = read_csv(CELL_ROUTE_ALL_DAY_PATH)[["셀 ID", "노선"]].drop_duplicates()
    valid_pairs["셀 ID"] = pd.to_numeric(valid_pairs["셀 ID"], errors="coerce").astype("Int64")
    valid_pairs["노선"] = valid_pairs["노선"].astype(str).str.removesuffix(".0")

    frames = []
    for year in (2024, 2025):
        raw = read_csv(resolve_raw(year))
        ride_time = pd.to_datetime(
            raw["ride_dt"].astype(str), format="%Y%m%d%H%M%S", errors="coerce"
        )
        frame = pd.DataFrame({
            "셀 ID": raw["승차정류장ID"].astype(str).str.removesuffix(".0").map(mapping),
            "노선": raw["query_route_no"].astype(str).str.removesuffix(".0"),
            "연도": year,
            "시": ride_time.dt.hour,
            "승차인원": pd.to_numeric(raw["utztn_nope"], errors="coerce").fillna(0),
            "서울 하차": pd.to_numeric(raw["goff_ctpv_cd"], errors="coerce").eq(11),
        }).dropna(subset=["셀 ID", "시"])
        frame["셀 ID"] = frame["셀 ID"].astype("Int64")
        frame = frame.merge(valid_pairs, on=["셀 ID", "노선"], how="inner")
        frame = frame[frame["서울 하차"]].copy()
        frame["시간대"] = frame["시"].astype(int).map(time_band)
        frames.append(frame[["셀 ID", "연도", "시간대", "승차인원"]])

    trips = pd.concat(frames, ignore_index=True)
    result = (
        trips.groupby(["셀 ID", "시간대", "연도"], as_index=False)["승차인원"].sum()
        .pivot_table(index=["셀 ID", "시간대"], columns="연도", values="승차인원", fill_value=0)
        .rename(columns={2024: "2024년 승차인원", 2025: "2025년 승차인원"})
    )
    ids = sorted(valid_pairs["셀 ID"].dropna().unique())
    full_index = pd.MultiIndex.from_product([ids, TIME_LABELS], names=["셀 ID", "시간대"])
    result = result.reindex(full_index, fill_value=0).reset_index()
    result["승차인원 변화량"] = result["2025년 승차인원"] - result["2024년 승차인원"]
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print("저장:", OUTPUT_PATH)
    print("행 수:", len(result), "| 셀 수:", result["셀 ID"].nunique())
    print(result.groupby("시간대")[["2024년 승차인원", "2025년 승차인원"]].sum())


if __name__ == "__main__":
    main()
