"""VWorld 건축물정보(LT_C_BLDGINFO)를 126개 250m 격자별로 집계한다.

API 키와 등록 도메인은 실행 중 입력하며 파일에 저장하지 않는다.
공동주택은 건물용도 코드 02xxx 또는 건물명에 아파트·연립·다세대가 포함된
건물로 분류한다. 경계에 걸친 건물의 연면적은 건물 바닥 도형의 격자 내
포함면적 비율로 배분한다.
"""
from __future__ import annotations

import getpass
import json
import os
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from shapely.geometry import Polygon, box, shape


PROJECT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT / "data" / "processed" / "16_새데이터_공급구조유지_250m격자.csv"
OUTPUT_PATH = PROJECT / "data" / "processed" / "25_VWorld건축물정보_현재_250m격자.csv"
CACHE_DIR = PROJECT / "data" / "raw" / "vworld_건축물정보_캐시"

API_URL = "https://api.vworld.kr/req/data"
DATASET = "LT_C_BLDGINFO"
CELL_HALF_SIZE_M = 125.0
PROJECTED_CRS = "EPSG:5179"
GEOGRAPHIC_CRS = "EPSG:4326"
APARTMENT_WORDS = ("아파트", "연립", "다세대", "공동주택")


def as_number(value):
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def is_apartment(props):
    usage = str(props.get("usability") or "").strip().replace(".0", "")
    name = f"{props.get('bld_nm') or ''} {props.get('dong_nm') or ''}"
    return usage.startswith("02") or any(word in name for word in APARTMENT_WORDS)


def request_features(session, api_key, domain, bounds_4326, cache_path):
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    minx, miny, maxx, maxy = bounds_4326
    page = 1
    features = []
    while True:
        params = {
            "service": "data", "version": "2.0", "request": "GetFeature",
            "key": api_key, "format": "json", "errorFormat": "json",
            "size": 1000, "page": page, "data": DATASET,
            "geomFilter": f"BOX({minx},{miny},{maxx},{maxy})",
            "columns": "bld_nm,dong_nm,usability,strct_cd,grnd_flr,ugrnd_flr,archarea,height,vl_rat,bc_rat,totalarea,platarea,useapr_day",
            "geometry": "true", "attribute": "true", "crs": GEOGRAPHIC_CRS,
        }
        if domain:
            params["domain"] = domain
        response = session.get(API_URL, params=params, timeout=90)
        response.raise_for_status()
        payload = response.json()
        root = payload.get("response", {})
        if root.get("status") == "ERROR":
            error = root.get("error", {})
            raise RuntimeError(f"VWorld 오류: {error.get('code')} / {error.get('text')}")
        result = root.get("result") or {}
        collection = result.get("featureCollection") or result
        current = collection.get("features", []) if isinstance(collection, dict) else []
        features.extend(current)
        total_pages = int((root.get("page") or {}).get("total", 1) or 1)
        if page >= total_pages or not current:
            break
        page += 1
    cache_path.write_text(json.dumps(features, ensure_ascii=False), encoding="utf-8")
    return features


def main():
    api_key = os.getenv("VWORLD_API_KEY") or getpass.getpass("VWorld API 키: ").strip()
    if not api_key:
        raise ValueError("VWorld API 키가 필요합니다.")
    domain_env = os.getenv("VWORLD_DOMAIN")
    domain = input("키 발급 시 등록한 도메인(없으면 Enter): ").strip() if domain_env is None else domain_env.strip()
    if domain == "-":
        domain = ""

    cells = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    cells["셀 ID"] = pd.to_numeric(cells["셀 ID"], errors="raise").astype(int)
    if cells["셀 ID"].nunique() != 126:
        raise ValueError(f"분석 격자가 126개가 아님: {cells['셀 ID'].nunique()}")

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
        cell = box(cx - CELL_HALF_SIZE_M, cy - CELL_HALF_SIZE_M, cx + CELL_HALF_SIZE_M, cy + CELL_HALF_SIZE_M)
        corners = [to_geographic.transform(x, y) for x, y in cell.exterior.coords]
        bounds = Polygon(corners).bounds
        features = request_features(session, api_key, domain, bounds, CACHE_DIR / f"cell_{cell_id}.json")

        building_count = apartment_count = highrise_count = 0
        total_floor_area = apartment_floor_area = 0.0
        floors = []
        heights = []
        apartment_names = set()
        seen = set()

        for feature in features:
            geometry = feature.get("geometry")
            if not geometry:
                continue
            props = feature.get("properties") or {}
            geom = gpd.GeoSeries([shape(geometry)], crs=GEOGRAPHIC_CRS).to_crs(PROJECTED_CRS).iloc[0]
            if geom.is_empty or geom.area <= 0:
                continue
            clipped = geom.intersection(cell)
            if clipped.is_empty or clipped.area <= 0:
                continue
            unique_key = (geom.wkb_hex, str(props.get("bld_nm")), str(props.get("dong_nm")))
            if unique_key in seen:
                continue
            seen.add(unique_key)

            overlap_share = min(1.0, clipped.area / geom.area)
            totalarea = as_number(props.get("totalarea"))
            grnd_flr = as_number(props.get("grnd_flr"))
            height = as_number(props.get("height"))
            apartment = is_apartment(props)

            # 건물 수·층수·높이는 건물 중심점이 해당 격자 안에 있을 때만 한 번 센다.
            if cell.covers(geom.representative_point()):
                building_count += 1
                if pd.notna(grnd_flr):
                    floors.append(float(grnd_flr))
                    if grnd_flr >= 10:
                        highrise_count += 1
                if pd.notna(height):
                    heights.append(float(height))
                if apartment:
                    apartment_count += 1
                    name = str(props.get("bld_nm") or "").strip()
                    if name:
                        apartment_names.add(name)
            if pd.notna(totalarea) and totalarea >= 0:
                allocated = float(totalarea) * overlap_share
                total_floor_area += allocated
                if apartment:
                    apartment_floor_area += allocated

        rows.append({
            "셀 ID": cell_id,
            "VWorld 전체 건축물 수": building_count,
            "VWorld 전체 연면적(㎡)": total_floor_area,
            "VWorld 공동주택 건축물 수": apartment_count,
            "VWorld 공동주택 연면적(㎡)": apartment_floor_area,
            "VWorld 공동주택 연면적 비율(%)": apartment_floor_area / total_floor_area * 100 if total_floor_area > 0 else 0.0,
            "VWorld 평균 지상층수": np.mean(floors) if floors else 0.0,
            "VWorld 최대 지상층수": max(floors) if floors else 0.0,
            "VWorld 10층 이상 건축물 수": highrise_count,
            "VWorld 평균 건물높이(m)": np.mean(heights) if heights else 0.0,
            "VWorld 최대 건물높이(m)": max(heights) if heights else 0.0,
            "VWorld 공동주택 명칭": " | ".join(sorted(apartment_names)),
            "API 반환 피처 수": len(features),
        })
        print(f"[{sequence:>3}/{len(cells)}] 셀 {cell_id}: API {len(features)}개 / 공동주택 {apartment_count}개")
        time.sleep(0.05)

    output = pd.DataFrame(rows).sort_values("셀 ID")
    output["VWorld 전체 연면적(천㎡)"] = output["VWorld 전체 연면적(㎡)"] / 1000
    output["VWorld 공동주택 연면적(천㎡)"] = output["VWorld 공동주택 연면적(㎡)"] / 1000
    output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print("\n저장:", OUTPUT_PATH)
    print("격자 수:", len(output))
    print("공동주택이 있는 격자:", int(output["VWorld 공동주택 건축물 수"].gt(0).sum()))
    print(output.drop(columns=["VWorld 공동주택 명칭"]).describe().round(2).to_string())


if __name__ == "__main__":
    main()
