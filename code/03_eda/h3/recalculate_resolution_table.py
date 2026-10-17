"""2024-10-17과 2025-11-06 자료로 격자 해상도별 공급구조 유지 현황을 계산한다."""
from pathlib import Path
import re
import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir())
A2 = ROOT / "data" / "processed" / "gtx_a" / "workspace" / "Analisis2"
GRID_DIR = ROOT / "data" / "raw" / "common" / "grid"
TRANSPORT = ROOT / "data" / "raw" / "gtx_a" / "transport_card"
ADMIN_PATH = ROOT / "data" / "raw" / "common" / "api_cache_gtx_a_routes" / "HangJeongDong_ver20260701.geojson"
FILE_2024 = TRANSPORT / "gtx_a_transport_card_20241017_raw_with_coords_api_filled.csv"
FILE_2025 = TRANSPORT / "gtx_a_transport_card_20251106_raw_with_coords.csv"
OUTPUT = A2 / "grid_resolution_supply_comparison_20241017_20251106.csv"

def load_card(path, year):
    raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    frame = pd.DataFrame({
        "연도": year,
        "노선번호": raw["query_route_no"].astype(str).str.strip(),
        "정류장ID": raw["ride_sttn_id"].astype(str).str.replace(r"\.0$", "", regex=True),
        "정류장명": raw["승차정류장명"].astype("string"),
        "위도": pd.to_numeric(raw["승차위도"], errors="coerce"),
        "경도": pd.to_numeric(raw["승차경도"], errors="coerce"),
        "승차인원": pd.to_numeric(raw["utztn_nope"], errors="coerce").fillna(0),
    })
    return frame.dropna(subset=["정류장명", "위도", "경도"])

def normalize_name(value):
    return re.sub(r"[ .·ㆍ,()/\-]+", "", str(value).strip().lower())

def same_stop_configuration(old, new, tolerance=100):
    if old.empty and new.empty:
        return True
    if old.empty or new.empty or len(old) != len(new):
        return False
    old, new = old.copy(), new.copy()
    for frame in (old, new):
        frame["이름키"] = frame["정류장명"].map(normalize_name)
        frame["x"], frame["y"] = frame.geometry.x, frame.geometry.y
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

def grid_path(size):
    label = "1km" if size == 1000 else f"{size}m"
    path = GRID_DIR / f"빈격자({label})" / f"빈격자({label}).shp"
    if not path.exists():
        raise FileNotFoundError(path)
    return path

def calculate(size, card, admin):
    path = grid_path(size)
    crs = gpd.read_file(path, rows=1).crs
    city = admin[admin["sgg_code"].str.startswith(("4128", "4148"))].to_crs(crs)
    city_union = city.geometry.union_all() if hasattr(city.geometry, "union_all") else city.geometry.unary_union
    # 전국 100m 격자는 매우 크므로 고양·파주 경계 주변만 읽는다.
    grid = gpd.read_file(path, bbox=city_union.bounds)
    city_grid = grid[grid.geometry.intersects(city_union)].copy()
    city_grid["_geometry_key"] = city_grid.geometry.map(lambda geom: geom.wkb if geom is not None else None)
    city_grid = city_grid.drop_duplicates("_geometry_key").drop(columns="_geometry_key").reset_index(drop=True)
    city_grid["셀 ID"] = np.arange(1, len(city_grid) + 1)

    points = gpd.GeoDataFrame(card, geometry=gpd.points_from_xy(card["경도"], card["위도"]), crs="EPSG:4326").to_crs(crs)
    joined = gpd.sjoin(points, city_grid[["셀 ID", "geometry"]], how="inner", predicate="within").drop(columns="index_right")
    stops = joined[["연도", "셀 ID", "노선번호", "정류장ID", "정류장명", "geometry"]].drop_duplicates(
        ["연도", "셀 ID", "노선번호", "정류장ID", "정류장명"]
    )
    route_lookup = stops.groupby(["셀 ID", "연도"])["노선번호"].apply(lambda x: set(x.astype(str))).to_dict()
    stop_lookup = {key: group.drop_duplicates(["노선번호", "정류장명"]) for key, group in stops.groupby(["셀 ID", "연도"])}
    demand = joined.groupby(["셀 ID", "연도"])["승차인원"].sum().unstack(fill_value=0)

    observed_cells = sorted(stops["셀 ID"].unique())
    stable_stop_counts = []
    for cell_id in observed_cells:
        routes24, routes25 = route_lookup.get((cell_id, 2024), set()), route_lookup.get((cell_id, 2025), set())
        stops24 = stop_lookup.get((cell_id, 2024), stops.iloc[0:0])
        stops25 = stop_lookup.get((cell_id, 2025), stops.iloc[0:0])
        board24 = float(demand.get(2024, pd.Series(dtype=float)).get(cell_id, 0))
        board25 = float(demand.get(2025, pd.Series(dtype=float)).get(cell_id, 0))
        stable = routes24 == routes25 and same_stop_configuration(stops24, stops25) and board24 > 0 and board25 > 0
        if stable:
            stable_stop_counts.append((len(stops24) + len(stops25)) / 2)

    stable_n, observed_n = len(stable_stop_counts), len(observed_cells)
    return {
        "격자 크기": "1km" if size == 1000 else f"{size}m",
        "공급구조 유지 격자 수": stable_n,
        "유지 비율(%)": stable_n / observed_n * 100 if observed_n else np.nan,
        "유지 격자당 평균 정류장 수": np.mean(stable_stop_counts) if stable_stop_counts else np.nan,
    }

def main():
    admin = gpd.read_file(ADMIN_PATH)
    admin["sgg_code"] = admin["sgg"].astype(str).str.extract(r"([0-9]+)")[0].str.zfill(5)
    card = pd.concat([load_card(FILE_2024, 2024), load_card(FILE_2025, 2025)], ignore_index=True)
    table = pd.DataFrame([calculate(size, card, admin) for size in [100, 250, 500, 1000]])
    table[["유지 비율(%)", "유지 격자당 평균 정류장 수"]] = table[["유지 비율(%)", "유지 격자당 평균 정류장 수"]].round(2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("저장:", OUTPUT)

if __name__ == "__main__":
    main()
