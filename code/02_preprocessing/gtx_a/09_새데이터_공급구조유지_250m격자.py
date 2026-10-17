"""2024-10-17과 2025-11-06의 공급구조 유지 250m 격자를 재선정한다.

공급구조 유지 기준
1. 격자 내 노선 집합이 동일
2. 노선별 정류장 구성이 동일
3. 정류장 ID가 바뀌어도 정류장명과 좌표(100m 이내)가 일치하면 동일 정류장으로 판정
4. 두 기준일 모두 승차인원이 0보다 큼

기존 132개 격자 파일은 덮어쓰지 않는다.
"""

from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir())
ROOT = PROJECT_DIR
PROCESSED_DIR = PROJECT_DIR / "data" / "processed" / "gtx_a"
GRID_PATH = PROJECT_DIR / "data" / "raw" / "common" / "grid" / "빈격자(250m)" / "빈격자(250m).shp"
ADMIN_PATH = PROJECT_DIR / "data" / "raw" / "common" / "api_cache_gtx_a_routes" / "HangJeongDong_ver20260701.geojson"
if not GRID_PATH.exists():
    GRID_PATH = ROOT / "data" / "raw" / "common" / "grid" / "빈격자(250m)" / "빈격자(250m).shp"
if not ADMIN_PATH.exists():
    ADMIN_PATH = ROOT / "data" / "raw" / "common" / "api_cache_gtx_a_routes" / "HangJeongDong_ver20260701.geojson"
CARD_DIR = ROOT / "data" / "raw" / "gtx_a" / "transport_card"
FILE_2024 = CARD_DIR / "gtx_a_transport_card_20241017_raw_with_coords_api_filled.csv"
FILE_2025 = CARD_DIR / "gtx_a_transport_card_20251106_raw_with_coords.csv"

OUTPUT_STABLE = PROCESSED_DIR / "16_새데이터_공급구조유지_250m격자.csv"
OUTPUT_ALL = PROCESSED_DIR / "16-1_새데이터_공급구조판정_전체격자.csv"
OUTPUT_SUMMARY = PROCESSED_DIR / "16-2_새데이터_공급구조유지_요약.csv"


def normalize_name(value):
    return re.sub(r"[ .·ㆍ,()/\-]+", "", str(value).strip().lower())


def load_card(path, year, date):
    raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame = pd.DataFrame({
        "연도": year,
        "기준일": date,
        "노선번호": raw["query_route_no"].astype(str).str.strip(),
        "정류장ID": raw["ride_sttn_id"].astype(str).str.replace(r"\.0$", "", regex=True),
        "정류장명": raw["승차정류장명"].astype("string"),
        "위도": pd.to_numeric(raw["승차위도"], errors="coerce"),
        "경도": pd.to_numeric(raw["승차경도"], errors="coerce"),
        "승차인원": pd.to_numeric(raw["utztn_nope"], errors="coerce").fillna(0),
    })
    missing = int(frame[["정류장명", "위도", "경도"]].isna().any(axis=1).sum())
    if missing:
        print(f"[경고] {date}: 정류장명/좌표 미매칭 {missing:,}행은 공간 배정에서 제외됩니다.")
    return frame.dropna(subset=["정류장명", "위도", "경도"])


def same_stop_configuration(old, new, tolerance=100):
    if old.empty and new.empty:
        return True
    if old.empty or new.empty or len(old) != len(new):
        return False

    old = old.copy()
    new = new.copy()
    for frame in (old, new):
        frame["이름키"] = frame["정류장명"].map(normalize_name)
        frame["x"] = frame.geometry.x
        frame["y"] = frame.geometry.y

    used = set()
    for _, stop in old.iterrows():
        candidates = new[(new["노선번호"] == stop["노선번호"]) & (~new.index.isin(used))]
        if candidates.empty:
            return False
        exact = candidates[candidates["이름키"] == stop["이름키"]]
        pool = exact if not exact.empty else candidates
        distance = np.sqrt((pool["x"] - stop["x"]) ** 2 + (pool["y"] - stop["y"]) ** 2)
        best = distance.idxmin()
        if float(distance.loc[best]) > tolerance:
            return False
        used.add(best)
    return len(used) == len(new)


def join_text(values):
    return " | ".join(sorted(set(values.astype(str))))


def route_stop_text(group):
    values = {
        f"{route} : {name} [{stop_id}]"
        for route, name, stop_id in zip(group["노선번호"], group["정류장명"], group["정류장ID"])
    }
    return " | ".join(sorted(values))


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    card = pd.concat([
        load_card(FILE_2024, 2024, "2024-10-17"),
        load_card(FILE_2025, 2025, "2025-11-06"),
    ], ignore_index=True)

    routes_by_year = card.groupby("연도")["노선번호"].apply(lambda x: sorted(set(x)))
    print("2024 노선 수:", len(routes_by_year[2024]))
    print("2025 노선 수:", len(routes_by_year[2025]))
    if routes_by_year[2024] != routes_by_year[2025]:
        raise ValueError("두 기준일의 전체 노선 목록이 다릅니다.")

    grid = gpd.read_file(GRID_PATH)
    admin = gpd.read_file(ADMIN_PATH)
    admin["sgg_code"] = admin["sgg"].astype(str).str.extract(r"([0-9]+)")[0].str.zfill(5)
    city = admin[admin["sgg_code"].str.startswith(("4128", "4148"))].to_crs(grid.crs)
    city_union = city.geometry.union_all() if hasattr(city.geometry, "union_all") else city.geometry.unary_union
    city_grid = grid[grid.geometry.intersects(city_union)].copy().reset_index(drop=True)
    city_grid["_geometry_key"] = city_grid.geometry.apply(lambda geom: geom.wkb if geom is not None else None)
    city_grid = city_grid.drop_duplicates("_geometry_key").drop(columns="_geometry_key").reset_index(drop=True)
    city_grid["셀 ID"] = np.arange(1, len(city_grid) + 1)

    points = gpd.GeoDataFrame(
        card,
        geometry=gpd.points_from_xy(card["경도"], card["위도"]),
        crs="EPSG:4326",
    ).to_crs(grid.crs)
    joined = gpd.sjoin(points, city_grid[["셀 ID", "geometry"]], how="inner", predicate="within").drop(columns="index_right")

    stops = joined[["연도", "셀 ID", "노선번호", "정류장ID", "정류장명", "geometry"]].drop_duplicates(
        ["연도", "셀 ID", "노선번호", "정류장ID", "정류장명"]
    )
    route_lookup = stops.groupby(["셀 ID", "연도"])["노선번호"].apply(lambda x: set(x.astype(str))).to_dict()
    stop_lookup = {
        key: group.drop_duplicates(["노선번호", "정류장명"])
        for key, group in stops.groupby(["셀 ID", "연도"])
    }
    demand = joined.groupby(["셀 ID", "연도"])["승차인원"].sum().unstack(fill_value=0)

    rows = []
    for cell_id in sorted(stops["셀 ID"].unique()):
        routes24 = route_lookup.get((cell_id, 2024), set())
        routes25 = route_lookup.get((cell_id, 2025), set())
        stops24 = stop_lookup.get((cell_id, 2024), stops.iloc[0:0])
        stops25 = stop_lookup.get((cell_id, 2025), stops.iloc[0:0])
        board24 = float(demand.get(2024, pd.Series(dtype=float)).get(cell_id, 0))
        board25 = float(demand.get(2025, pd.Series(dtype=float)).get(cell_id, 0))
        route_same = routes24 == routes25
        stop_same = same_stop_configuration(stops24, stops25)
        stable = route_same and stop_same and board24 > 0 and board25 > 0
        rows.append({
            "셀 ID": int(cell_id),
            "공급구조 유지 여부": int(stable),
            "노선 구성 동일": int(route_same),
            "정류장 구성 동일": int(stop_same),
            "2024년 승차인원": board24,
            "2025년 승차인원": board25,
            "승차인원 증감": board25 - board24,
            "2024년 노선 목록": join_text(stops24["노선번호"]) if not stops24.empty else "",
            "2025년 노선 목록": join_text(stops25["노선번호"]) if not stops25.empty else "",
            "2024년 정류장 ID": join_text(stops24["정류장ID"]) if not stops24.empty else "",
            "2025년 정류장 ID": join_text(stops25["정류장ID"]) if not stops25.empty else "",
            "2024년 노선·정류장 상세": route_stop_text(stops24) if not stops24.empty else "",
            "2025년 노선·정류장 상세": route_stop_text(stops25) if not stops25.empty else "",
        })

    judged = pd.DataFrame(rows)
    centers = city_grid.copy()
    center_wgs = gpd.GeoSeries(centers.geometry.centroid, crs=grid.crs).to_crs("EPSG:4326")
    centers["중심점 위도"] = center_wgs.y.values
    centers["중심점 경도"] = center_wgs.x.values
    judged = judged.merge(centers[["셀 ID", "중심점 위도", "중심점 경도"]], on="셀 ID", how="left")
    judged.to_csv(OUTPUT_ALL, index=False, encoding="utf-8-sig")

    stable = judged[judged["공급구조 유지 여부"] == 1].copy().sort_values("셀 ID")
    stable.to_csv(OUTPUT_STABLE, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([{
        "2024 기준일": "2024-10-17",
        "2025 기준일": "2025-11-06",
        "분석대상 노선 수": len(routes_by_year[2024]),
        "교통카드 정류장 관측 격자 수": len(judged),
        "공급구조 유지 격자 수": len(stable),
        "2024년 유지격자 승차인원": stable["2024년 승차인원"].sum(),
        "2025년 유지격자 승차인원": stable["2025년 승차인원"].sum(),
    }])
    summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")

    print("\n=== 새 데이터 공급구조 판정 결과 ===")
    print(summary.to_string(index=False))
    print("\n저장:", OUTPUT_STABLE)
    print("전체 판정표:", OUTPUT_ALL)


if __name__ == "__main__":
    main()
