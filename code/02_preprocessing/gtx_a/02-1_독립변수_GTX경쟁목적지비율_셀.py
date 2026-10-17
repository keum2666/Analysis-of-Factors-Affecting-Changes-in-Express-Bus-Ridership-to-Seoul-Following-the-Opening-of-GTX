"""셀×노선 경쟁 목적지 집계자료를 셀 단위 비율로 재집계한다.

단순한 노선별 비율 평균이 아니라, 각 노선의 경쟁 목적지 승객 수와
목적지 분류 가능 승객 수를 먼저 합산한 뒤 비율을 계산한다.
"""
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir())
INPUT_PATH = PROJECT_DIR / "data" / "processed" / "gtx_a" / "05_GTX_경쟁목적지비율_종일_셀노선.csv"
OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "gtx_a" / "06_GTX_경쟁목적지비율_종일_셀.csv"

ID = "셀 ID"
TOTAL = "2024년 종일 전체 서울행 승차인원"
CLASSIFIED = "2024년 종일 목적지 분류 가능 승차인원"
COMPETING = "2024년 종일 GTX 경쟁 목적지 승차인원"
SHARE = "GTX 경쟁 목적지 승차인원 비율(%)"


def main():
    source = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    source[ID] = pd.to_numeric(source[ID], errors="coerce").astype("Int64")
    for column in (TOTAL, CLASSIFIED, COMPETING):
        source[column] = pd.to_numeric(source[column], errors="coerce").fillna(0)

    result = (
        source.groupby(ID, as_index=False)[[TOTAL, CLASSIFIED, COMPETING]]
        .sum()
        .sort_values(ID)
        .reset_index(drop=True)
    )
    result[SHARE] = np.where(
        result[CLASSIFIED] > 0,
        result[COMPETING] / result[CLASSIFIED] * 100,
        np.nan,
    )
    result["GTX 경쟁 목적지 기준"] = "연신내역·서울역 하차 반경 1km 이내"
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("저장:", OUTPUT_PATH)
    print("셀 수:", result[ID].nunique())
    print("비율 결측:", result[SHARE].isna().sum())
    print(result[SHARE].describe().round(2))


if __name__ == "__main__":
    main()
