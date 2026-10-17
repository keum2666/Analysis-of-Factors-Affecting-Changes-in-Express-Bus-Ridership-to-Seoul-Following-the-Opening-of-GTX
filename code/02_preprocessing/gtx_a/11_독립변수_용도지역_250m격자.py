"""VWorld 도시지역(LT_C_UQ111)을 이용한 250m 격자별 용도지역 면적비율 구축.

실행하면 VWorld API 키와 키 발급 시 등록한 도메인을 입력받는다.
키는 코드나 결과 파일에 저장하지 않는다.
"""

from __future__ import annotations

import getpass
import os
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from pyproj import Transformer
from shapely.geometry import Polygon, box, shape
from shapely.ops import unary_union


ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir())
INPUT_PATH = ROOT / "data" / "processed" / "gtx_a" / "16_새데이터_공급구조유지_250m격자.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "gtx_a" / "19_용도지역_면적비율_250m격자.csv"
CACHE_DIR = ROOT / "data" / "raw" / "gtx_a" / "vworld_용도지역_캐시"

API_URL = "https://api.vworld.kr/req/data"
DATASET = "LT_C_UQ111"
PROJECTED_CRS = "EPSG:5179"
GEOGRAPHIC_CRS = "EPSG:4326"
CELL_HALF_SIZE_M = 125.0


def classify_zone(name: str) -> str:
    """세부 용도지역명을 주거·상업·공업·녹지·기타도시지역으로 묶는다."""
    name = str(name or "").strip()
    if "주거지역" in name:
        return "주거지역"
    if "상업지역" in name:
        return "상업지역"
    if "공업지역" in name:
        return "공업지역"
    if "녹지지역" in name:
        return "녹지지역"
    return "기타도시지역"


def request_cell_features(
    session: requests.Session,
    api_key: str,
    domain: str,
    bounds_4326: tuple[float, float, float, float],
) -> list[dict]:
    """한 격자 BOX와 교차하는 모든 피처를 페이지 끝까지 조회한다."""
    minx, miny, maxx, maxy = bounds_4326
    page = 1
    features: list[dict] = []

    while True:
        params = {
            "service": "data",
            "version": "2.0",
            "request": "GetFeature",
            "key": api_key,
            "format": "json",
            "errorFormat": "json",
            "size": 1000,
            "page": page,
            "data": DATASET,
            "geomFilter": f"BOX({minx},{miny},{maxx},{maxy})",
            "columns": "uname,sido_name,sigg_name,dyear,dnum",
            "geometry": "true",
            "attribute": "true",
            "crs": "EPSG:4326",
        }
        if domain:
            params["domain"] = domain

        response = session.get(API_URL, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()

        if payload.get("response", {}).get("status") == "ERROR":
            error = payload["response"].get("error", {})
            raise RuntimeError(f"VWorld 오류: {error.get('code')} / {error.get('text')}")

        response_root = payload.get("response", {})
        result = response_root.get("result") or {}
        collection = result.get("featureCollection") or result
        current_features = collection.get("features", []) if isinstance(collection, dict) else []
        features.extend(current_features)

        page_info = response_root.get("page") or {}
        total_pages = int(page_info.get("total", 1) or 1)
        if page >= total_pages or not current_features:
            break
        page += 1

    return features


def main() -> None:
    api_key = os.getenv("VWORLD_API_KEY") or getpass.getpass("VWorld API 키: ").strip()
    if not api_key:
        raise ValueError("VWorld API 키가 필요합니다.")
    domain = os.getenv("VWORLD_DOMAIN") or input(
        "키 발급 시 등록한 도메인(없으면 Enter): "
    ).strip()

    cells = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    cells = cells[cells["공급구조 유지 여부"].eq(1)].copy()
    cells["셀 ID"] = pd.to_numeric(cells["셀 ID"], errors="raise").astype(int)

    to_projected = Transformer.from_crs(GEOGRAPHIC_CRS, PROJECTED_CRS, always_xy=True)
    to_geographic = Transformer.from_crs(PROJECTED_CRS, GEOGRAPHIC_CRS, always_xy=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    rows = []

    for sequence, (_, row) in enumerate(cells.iterrows(), start=1):
        cell_id = int(row["셀 ID"])
        lon = float(row["중심점 경도"])
        lat = float(row["중심점 위도"])
        cx, cy = to_projected.transform(lon, lat)
        cell_polygon = box(
            cx - CELL_HALF_SIZE_M,
            cy - CELL_HALF_SIZE_M,
            cx + CELL_HALF_SIZE_M,
            cy + CELL_HALF_SIZE_M,
        )

        corners = [to_geographic.transform(x, y) for x, y in cell_polygon.exterior.coords]
        bbox_4326 = Polygon(corners).bounds
        features = request_cell_features(session, api_key, domain, bbox_4326)

        zone_geometries: dict[str, list] = {
            "주거지역": [], "상업지역": [], "공업지역": [],
            "녹지지역": [], "기타도시지역": [],
        }
        zone_names = set()
        years = set()

        for feature in features:
            geometry = feature.get("geometry")
            if not geometry:
                continue
            props = feature.get("properties") or {}
            zone_name = str(props.get("uname", "")).strip()
            zone_class = classify_zone(zone_name)
            projected_geom = gpd.GeoSeries(
                [shape(geometry)], crs=GEOGRAPHIC_CRS
            ).to_crs(PROJECTED_CRS).iloc[0]
            clipped = projected_geom.intersection(cell_polygon)
            if not clipped.is_empty:
                zone_geometries[zone_class].append(clipped)
                if zone_name:
                    zone_names.add(zone_name)
                if props.get("dyear"):
                    years.add(str(props["dyear"]))

        areas = {}
        for zone_class, geometries in zone_geometries.items():
            areas[zone_class] = unary_union(geometries).area if geometries else 0.0

        # 하위 용도지역 경계가 겹치는 경우에도 도시지역 전체 면적은 중복 계산하지 않는다.
        all_geometries = [geom for values in zone_geometries.values() for geom in values]
        urban_area = unary_union(all_geometries).area if all_geometries else 0.0
        cell_area = cell_polygon.area

        result = {
            "셀 ID": cell_id,
            "주거지역 비율(%)": areas["주거지역"] / cell_area * 100,
            "상업지역 비율(%)": areas["상업지역"] / cell_area * 100,
            "공업지역 비율(%)": areas["공업지역"] / cell_area * 100,
            "녹지지역 비율(%)": areas["녹지지역"] / cell_area * 100,
            "기타도시지역 비율(%)": areas["기타도시지역"] / cell_area * 100,
            "도시지역 비율(%)": urban_area / cell_area * 100,
            "비도시지역 비율(%)": max(0.0, 100 - urban_area / cell_area * 100),
            "세부 용도지역": " | ".join(sorted(zone_names)),
            "지정연도": " | ".join(sorted(years)),
        }
        rows.append(result)
        print(f"[{sequence:>3}/{len(cells)}] 셀 {cell_id}: 피처 {len(features)}개")
        time.sleep(0.05)

    output = pd.DataFrame(rows).sort_values("셀 ID")
    ratio_columns = [column for column in output.columns if "비율(%)" in column]
    output[ratio_columns] = output[ratio_columns].clip(lower=0, upper=100).round(4)
    output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n저장:", OUTPUT_PATH)
    print("격자 수:", len(output))
    print(output[ratio_columns].describe().round(2))


if __name__ == "__main__":
    main()
