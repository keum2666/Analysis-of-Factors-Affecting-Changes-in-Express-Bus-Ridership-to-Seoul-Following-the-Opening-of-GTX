"""원시 교통카드 두 파일을 최종모형용 셀×노선 종일 집계자료로 변환한다.

사용 예:
python scripts/build_cell_route_all_day.py --raw-2024 <CSV> --raw-2025 <CSV>

출력에는 셀 ID, 노선, 2024년 승차인원, 2025년 승차인원만 남긴다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir())
sys.path.insert(0, str(PROJECT_DIR))
from config import CELL_ROUTE_ALL_DAY_PATH, STABLE_PATH, read_csv  # noqa: E402


def stop_to_cell_map() -> dict[str, int]:
    stable = read_csv(STABLE_PATH)
    mapping: dict[str, int] = {}
    for _, row in stable.iterrows():
        for column in ("2024년 정류장 ID", "2025년 정류장 ID"):
            for stop_id in str(row[column]).split("|"):
                stop_id = stop_id.strip().removesuffix(".0")
                if stop_id and stop_id.lower() != "nan":
                    mapping[stop_id] = int(row["셀 ID"])
    return mapping


def aggregate(path: Path, year: int, mapping: dict[str, int]) -> pd.DataFrame:
    raw = read_csv(path)
    stop = raw["승차정류장ID"].astype(str).str.removesuffix(".0")
    frame = pd.DataFrame(
        {
            "셀 ID": stop.map(mapping),
            "노선": raw["query_route_no"].astype(str),
            "승차인원": pd.to_numeric(raw["utztn_nope"], errors="coerce").fillna(0),
        }
    ).dropna(subset=["셀 ID"])
    frame["셀 ID"] = frame["셀 ID"].astype(int)
    return (
        frame.groupby(["셀 ID", "노선"], as_index=False)["승차인원"]
        .sum()
        .rename(columns={"승차인원": f"y{year}"})
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-2024", type=Path, required=True)
    parser.add_argument("--raw-2025", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=CELL_ROUTE_ALL_DAY_PATH)
    args = parser.parse_args()

    mapping = stop_to_cell_map()
    y2024 = aggregate(args.raw_2024, 2024, mapping)
    y2025 = aggregate(args.raw_2025, 2025, mapping)
    result = y2024.merge(y2025, on=["셀 ID", "노선"], how="outer").fillna(0)
    result = result.sort_values(["셀 ID", "노선"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"저장: {args.output}")
    print(f"전체 셀×노선: {len(result)}, 2024년 승차가 있는 쌍: {(result.y2024 > 0).sum()}")


if __name__ == "__main__":
    main()
