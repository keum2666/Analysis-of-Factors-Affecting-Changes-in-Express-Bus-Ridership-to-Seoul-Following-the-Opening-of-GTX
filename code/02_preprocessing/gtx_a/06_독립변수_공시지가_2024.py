"""현재 132개 분석 격자에 2024년 공시지가와 로그값을 결합한다.

원자료는 국토통계 250m 공시지가 격자의 val(원/㎡)이다.
고양·파주 경계에서 동일 셀이 두 지역 자료에 모두 포함되면 두 값을 평균한다.
원자료에 값이 없는 셀은 0으로 대체하지 않고 결측으로 유지한다.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir())
UROP_ROOT = PROJECT_DIR
PROCESSED_DIR = PROJECT_DIR / "data" / "processed" / "gtx_a"
STABLE_PATH = PROCESSED_DIR / "01_공급구조유지_250m격자.csv"
OUTPUT_PATH = PROCESSED_DIR / "12_공시지가_2024_250m격자.csv"

LAND_ROOT_CANDIDATES = [
    PROJECT_DIR / "data" / "raw" / "gtx_a" / "land_price",
    UROP_ROOT / "data" / "raw" / "gtx_a" / "land_price",
]
LAND_ROOT = next((p for p in LAND_ROOT_CANDIDATES if p.exists()), None)
if LAND_ROOT is None:
    raise FileNotFoundError(
        "공시지가 원자료 폴더를 찾지 못했습니다. "
        "data/raw/gtx_a/land_price를 확인하세요."
    )

land_files = sorted(LAND_ROOT.glob("*202407/vl_blk.shp"))
if len(land_files) != 2:
    raise FileNotFoundError(f"2024년 고양·파주 공시지가 shp 2개가 필요합니다: {land_files}")

stable = pd.read_csv(STABLE_PATH, encoding="utf-8-sig")
stable["셀 ID"] = pd.to_numeric(stable["셀 ID"], errors="coerce").astype("Int64")
points = gpd.GeoDataFrame(
    stable[["셀 ID"]].copy(),
    geometry=gpd.points_from_xy(stable["중심점 경도"], stable["중심점 위도"]),
    crs="EPSG:4326",
).to_crs("EPSG:5179")

layers = []
for path in land_files:
    layer = gpd.read_file(path)[["gid", "val", "geometry"]].to_crs("EPSG:5179")
    layer["공시지가"] = pd.to_numeric(layer["val"], errors="coerce")
    layer["원자료 지역"] = "고양시" if "고양시" in str(path) else "파주시"
    layers.append(layer[["gid", "공시지가", "원자료 지역", "geometry"]])

land = gpd.GeoDataFrame(pd.concat(layers, ignore_index=True), crs="EPSG:5179")
joined = gpd.sjoin(points, land, how="left", predicate="within")

# 경계 셀은 고양·파주 두 값의 산술평균을 사용한다.
summary = joined.groupby("셀 ID", as_index=False).agg(
    **{
        "2024년 공시지가(원/㎡)": ("공시지가", "mean"),
        "매칭 원자료 수": ("공시지가", "count"),
    }
)
summary["log_2024년 공시지가"] = np.where(
    summary["2024년 공시지가(원/㎡)"] > 0,
    np.log(summary["2024년 공시지가(원/㎡)"]),
    np.nan,
)
summary["공시지가 결측"] = summary["2024년 공시지가(원/㎡)"].isna().astype(int)
summary.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("저장:", OUTPUT_PATH)
print("전체 격자:", len(summary))
print("공시지가 매칭:", int(summary["2024년 공시지가(원/㎡)"].notna().sum()))
print("결측 셀:", summary.loc[summary["공시지가 결측"] == 1, "셀 ID"].tolist())
print("경계 중복 셀:", summary.loc[summary["매칭 원자료 수"] > 1, "셀 ID"].tolist())
print(summary[["2024년 공시지가(원/㎡)", "log_2024년 공시지가"]].describe())
