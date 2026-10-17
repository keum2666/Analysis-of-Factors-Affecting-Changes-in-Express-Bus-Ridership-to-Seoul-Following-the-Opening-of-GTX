"""2024년 종일 교통카드 자료로 250m 격자별 광역버스 이용거리를 구축한다.

이용거리 원자료 열: utztn_dstnc
원자료 값은 미터 단위로 보고 km로 환산한다. 회귀용 대표값은 이용인원으로
가중한 평균 이동거리이며, 이상치에 덜 민감한 가중 중앙값도 함께 저장한다.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir())
ROOT = PROJECT
PROCESSED = PROJECT / "data" / "processed" / "gtx_a"
CARD_DIR = ROOT / "data" / "raw" / "gtx_a" / "transport_card"

STABLE_PATH = PROCESSED / "16_새데이터_공급구조유지_250m격자.csv"
CARD_PATH = CARD_DIR / "gtx_a_transport_card_20241017_raw_with_coords_api_filled.csv"
OUTPUT_PATH = PROCESSED / "21_광역버스이동거리_2024_종일_250m격자.csv"


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def split_ids(value):
    return [
        item.strip()
        for item in str(value).split("|")
        if item.strip() and item.strip().lower() != "nan"
    ]


def weighted_median(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = weights.sum() / 2
    return float(values[np.searchsorted(np.cumsum(weights), cutoff)])


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
    card["이용인원"] = pd.to_numeric(card["utztn_nope"], errors="coerce")
    card["이용거리_m"] = pd.to_numeric(card["utztn_dstnc"], errors="coerce")

    # 0보다 크고 200km 이하인 관측만 사용한다. 현재 자료의 최댓값은 약 55km다.
    valid = card[
        card["셀 ID"].notna()
        & card["이용인원"].gt(0)
        & card["이용거리_m"].gt(0)
        & card["이용거리_m"].le(200_000)
    ].copy()

    rows = []
    for cell_id, group in valid.groupby("셀 ID", sort=True):
        values = group["이용거리_m"].to_numpy(dtype=float)
        weights = group["이용인원"].to_numpy(dtype=float)
        rows.append({
            "셀 ID": int(cell_id),
            "2024년 승객가중 평균 이동거리(km)": np.average(values, weights=weights) / 1000,
            "2024년 승객가중 중앙 이동거리(km)": weighted_median(values, weights) / 1000,
            "이동거리 산정 이용인원": weights.sum(),
            "이동거리 원자료 행 수": len(group),
        })

    result = stable[["셀 ID"]].merge(pd.DataFrame(rows), on="셀 ID", how="left", validate="one_to_one")
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("저장:", OUTPUT_PATH)
    print("격자 수:", len(result))
    print("평균 이동거리 결측 셀:", int(result["2024년 승객가중 평균 이동거리(km)"].isna().sum()))
    print(result[["2024년 승객가중 평균 이동거리(km)", "2024년 승객가중 중앙 이동거리(km)"]].describe().round(3))


if __name__ == "__main__":
    main()
