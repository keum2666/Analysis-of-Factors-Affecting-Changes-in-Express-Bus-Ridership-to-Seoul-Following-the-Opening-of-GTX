"""SGIS 2024년 250m 건축물 연면적을 126개 분석 격자에 결합한다."""
from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT = next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir())
RAW_DIR = PROJECT / "data" / "raw" / "gtx_a" / "sgis_2024_floor_area"
PROCESSED = PROJECT / "data" / "processed" / "gtx_a"
STABLE_PATH = PROCESSED / "16_새데이터_공급구조유지_250m격자.csv"
OUTPUT_PATH = PROCESSED / "24_건축물연면적_2024_250m격자.csv"


def main():
    stable = pd.read_csv(STABLE_PATH, encoding="utf-8-sig")
    stable["셀 ID"] = pd.to_numeric(stable["셀 ID"], errors="raise").astype(int)
    points = gpd.GeoDataFrame(
        stable[["셀 ID"]].copy(),
        geometry=gpd.points_from_xy(stable["중심점 경도"], stable["중심점 위도"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:5179")

    frames = []
    for shp in sorted(RAW_DIR.rglob("*.shp")):
        frame = gpd.read_file(shp)[["val", "geometry"]].to_crs("EPSG:5179")
        frame["val"] = pd.to_numeric(frame["val"], errors="coerce")
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"연면적 shp가 없습니다: {RAW_DIR}")
    floor_area = pd.concat(frames, ignore_index=True)
    floor_area = gpd.GeoDataFrame(floor_area, geometry="geometry", crs="EPSG:5179")

    joined = gpd.sjoin(points, floor_area, how="left", predicate="within")
    # 시 경계에서 동일 격자가 중복 제공될 가능성에 대비해 셀별 최댓값 하나를 사용한다.
    values = joined.groupby("셀 ID", as_index=False)["val"].max()
    result = stable[["셀 ID"]].merge(values, on="셀 ID", how="left", validate="one_to_one")
    result = result.rename(columns={"val": "2024년 건축물 연면적(㎡)"})
    # SGIS는 건축물이 없는 격자를 행으로 제공하지 않으므로 미매칭은 연면적 0으로 처리한다.
    result["2024년 건축물 연면적(㎡)"] = result["2024년 건축물 연면적(㎡)"].fillna(0)
    result["2024년 건축물 연면적(천㎡)"] = result["2024년 건축물 연면적(㎡)"] / 1000
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("저장:", OUTPUT_PATH)
    print("격자 수:", len(result))
    print("결측 셀:", int(result["2024년 건축물 연면적(㎡)"].isna().sum()))
    print(result[["2024년 건축물 연면적(㎡)", "2024년 건축물 연면적(천㎡)"]].describe().round(3))


if __name__ == "__main__":
    main()
