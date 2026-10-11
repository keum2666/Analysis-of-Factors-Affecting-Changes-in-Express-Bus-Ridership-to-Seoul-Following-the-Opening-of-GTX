"""2024년 종일 교통카드 자료로 250m 격자별 출근시간 승차 비율을 구축한다."""

from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
PROCESSED = PROJECT / "data" / "processed"
CARD_DIR = ROOT / "gtx_a_seoul_bus_outputs" / "transport_card"

STABLE_PATH = PROCESSED / "16_새데이터_공급구조유지_250m격자.csv"
CARD_PATH = CARD_DIR / "gtx_a_transport_card_20241017_raw_with_coords_api_filled.csv"
OUTPUT_PATH = PROCESSED / "22_출근시간승차비율_2024_종일_250m격자.csv"


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def split_ids(value):
    return [
        item.strip()
        for item in str(value).split("|")
        if item.strip() and item.strip().lower() != "nan"
    ]


def main():
    stable = read_csv(STABLE_PATH)
    stable["셀 ID"] = pd.to_numeric(stable["셀 ID"], errors="raise").astype(int)
    if len(stable) != 126:
        raise ValueError(f"공급구조 유지 격자가 126개가 아님: {len(stable)}")

    stop_to_cell = {}
    for _, row in stable.iterrows():
        for stop_id in split_ids(row["2024년 정류장 ID"]):
            previous = stop_to_cell.get(stop_id)
            if previous is not None and previous != int(row["셀 ID"]):
                raise ValueError(f"정류장 {stop_id}가 두 셀에 중복 배정됨")
            stop_to_cell[stop_id] = int(row["셀 ID"])

    card = read_csv(CARD_PATH)
    card["정류장키"] = card["ride_sttn_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    card["셀 ID"] = card["정류장키"].map(stop_to_cell)
    card["이용인원"] = pd.to_numeric(card["utztn_nope"], errors="coerce").fillna(0)
    card["승차시각"] = pd.to_datetime(
        card["ride_dt"].astype(str).str.replace(r"\.0$", "", regex=True),
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )
    card["승차시간"] = card["승차시각"].dt.hour
    card = card[card["셀 ID"].notna()].copy()

    total = card.groupby("셀 ID")["이용인원"].sum().rename("2024년 종일 승차인원")
    morning = (
        card[card["승차시간"].between(6, 9)]
        .groupby("셀 ID")["이용인원"]
        .sum()
        .rename("2024년 출근시간 승차인원(06~09시)")
    )
    result = stable[["셀 ID"]].merge(total, on="셀 ID", how="left").merge(morning, on="셀 ID", how="left")
    result[["2024년 종일 승차인원", "2024년 출근시간 승차인원(06~09시)"]] = result[
        ["2024년 종일 승차인원", "2024년 출근시간 승차인원(06~09시)"]
    ].fillna(0)
    result["2024년 출근시간 승차 비율(%)"] = (
        result["2024년 출근시간 승차인원(06~09시)"]
        / result["2024년 종일 승차인원"].where(result["2024년 종일 승차인원"] > 0)
        * 100
    )
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("저장:", OUTPUT_PATH)
    print("격자 수:", len(result))
    print("결측 셀:", int(result["2024년 출근시간 승차 비율(%)"].isna().sum()))
    print(result["2024년 출근시간 승차 비율(%)"].describe().round(3))


if __name__ == "__main__":
    main()
