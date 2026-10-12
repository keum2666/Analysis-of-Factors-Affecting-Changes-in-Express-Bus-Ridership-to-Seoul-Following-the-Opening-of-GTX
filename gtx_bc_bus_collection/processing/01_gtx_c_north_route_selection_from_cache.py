"""기존 TAGO 캐시에서 GTX-C 북부권 서울방향 광역버스를 추출한다.

대상 권역
- 덕정역: 양주시
- 의정부역: 의정부시

판정 기준은 GTX-A 노선 추출과 동일하다.
1. 광역급행·직행좌석·광역버스 계열일 것
2. 대상 시의 정류장을 경유할 것
3. 노선 정류장 순서상 대상 시 정류장 뒤에 서울특별시 정류장이 있을 것

공공데이터포털 API를 다시 호출하지 않고 기존 JSON 캐시만 읽는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
REPO = PROJECT_DIR.parent
CACHE_DIR = REPO / "Desktop" / "Gachon" / "3-2" / "UROP" / "api_cache_gtx_a_routes"
BOUNDARY_FILE = CACHE_DIR / "HangJeongDong_ver20260701.geojson"
OUTPUT_DIR = PROJECT_DIR / "processed"

TARGETS = [
    {"GTX_C_역": "덕정역", "대상지역": "양주시", "행정구역": "경기도 양주시"},
    {"GTX_C_역": "의정부역", "대상지역": "의정부시", "행정구역": "경기도 의정부시"},
]


def as_items(payload: dict) -> list[dict]:
    """TAGO JSON 응답에서 item 목록만 꺼낸다."""
    response = payload.get("response", payload)
    body = response.get("body", {}) if isinstance(response, dict) else {}
    items = body.get("items", {}) if isinstance(body, dict) else {}
    items = items.get("item", []) if isinstance(items, dict) else items
    if isinstance(items, dict):
        return [items]
    return items if isinstance(items, list) else []


def text(value) -> str:
    return "" if value is None else str(value).strip()


def is_metropolitan(route: dict) -> bool:
    value = f"{text(route.get('routeno'))} {text(route.get('routetp'))}".upper()
    return any(keyword in value for keyword in ["광역버스", "광역급행", "직행좌석", "M버스"])


def load_cached_routes() -> tuple[dict[str, dict], dict[str, list[dict]]]:
    route_info: dict[str, dict] = {}
    route_stops: dict[str, list[dict]] = {}

    for path in CACHE_DIR.glob("getRouteInfoIem_*.json"):
        for row in as_items(json.loads(path.read_text(encoding="utf-8"))):
            route_id = text(row.get("routeid"))
            if route_id:
                route_info[route_id] = row

    for path in CACHE_DIR.glob("getRouteAcctoThrghSttnList_*.json"):
        rows = as_items(json.loads(path.read_text(encoding="utf-8")))
        if not rows:
            continue
        route_id = text(rows[0].get("routeid"))
        if route_id:
            route_stops[route_id] = rows

    return route_info, route_stops


def build_stop_table(route_stops: dict[str, list[dict]]) -> gpd.GeoDataFrame:
    records = []
    for route_id, rows in route_stops.items():
        for row in rows:
            if "미정차" in text(row.get("nodenm")):
                continue
            try:
                lat = float(row.get("gpslati"))
                lon = float(row.get("gpslong"))
                order = int(row.get("nodeord"))
            except (TypeError, ValueError):
                continue
            records.append(
                {
                    "노선ID": route_id,
                    "정류장순서": order,
                    "정류장ID": text(row.get("nodeid")),
                    "정류장번호": text(row.get("nodeno")),
                    "정류장명": text(row.get("nodenm")),
                    "경도": lon,
                    "위도": lat,
                }
            )

    stops = pd.DataFrame(records)
    points = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops["경도"], stops["위도"]),
        crs="EPSG:4326",
    )
    boundaries = gpd.read_file(BOUNDARY_FILE).to_crs("EPSG:4326")
    joined = gpd.sjoin(
        points,
        boundaries[["adm_cd2", "adm_nm", "geometry"]],
        how="left",
        predicate="within",
    )
    return joined.drop(columns=["index_right", "geometry"], errors="ignore")


def main() -> None:
    if not CACHE_DIR.exists():
        raise FileNotFoundError(f"기존 API 캐시가 없습니다: {CACHE_DIR}")
    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(f"행정동 경계가 없습니다: {BOUNDARY_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    route_info, route_stops = load_cached_routes()
    stops = build_stop_table(route_stops)

    candidates = []
    for route_id, info in route_info.items():
        if not is_metropolitan(info) or route_id not in route_stops:
            continue
        route_stop_df = stops[stops["노선ID"] == route_id].copy()
        if route_stop_df.empty:
            continue
        seoul = route_stop_df[route_stop_df["adm_nm"].fillna("").str.startswith("서울특별시")]
        if seoul.empty:
            continue

        for target in TARGETS:
            local = route_stop_df[
                route_stop_df["adm_nm"].fillna("").str.startswith(target["행정구역"])
            ]
            if local.empty:
                continue

            # 한 노선 안에 왕복 정류장 순서가 함께 있을 수 있으므로 실제로
            # 대상지역 정류장보다 뒤에 나타나는 서울 정류장이 있는지 확인한다.
            valid_pairs = [
                (int(local_order), int(seoul_order))
                for local_order in local["정류장순서"]
                for seoul_order in seoul["정류장순서"]
                if int(local_order) < int(seoul_order)
            ]
            if not valid_pairs:
                continue

            first_local, first_seoul = min(valid_pairs, key=lambda pair: pair[1] - pair[0])
            candidates.append(
                {
                    "GTX_C_역": target["GTX_C_역"],
                    "대상지역": target["대상지역"],
                    "버스번호": text(info.get("routeno")),
                    "노선ID": route_id,
                    "노선유형": text(info.get("routetp")),
                    "기점": text(info.get("startnodenm")),
                    "종점": text(info.get("endnodenm")),
                    "대상지역 정류장 수": len(local),
                    "서울 정류장 수": len(seoul),
                    "확인된 대상지역 순서": first_local,
                    "확인된 서울진입 순서": first_seoul,
                    "서울진입여부": "Y",
                }
            )

    candidate_df = pd.DataFrame(candidates).drop_duplicates(["GTX_C_역", "노선ID"])
    # 심야(N), 출근형(P), 예약형 노선은 일일 일반노선 분석에서 제외한다.
    excluded = (
        candidate_df["버스번호"].str.endswith("N")
        | candidate_df["버스번호"].str.contains(r"\((?:출근|예약)\)", regex=True)
    )
    candidate_df = candidate_df.loc[~excluded].copy()
    candidate_df = candidate_df.sort_values(["GTX_C_역", "버스번호", "노선ID"]).reset_index(drop=True)

    summary = (
        candidate_df.groupby(["GTX_C_역", "대상지역"], as_index=False)
        .agg(
            후보노선수=("노선ID", "nunique"),
            서울방향_광역버스_후보=("버스번호", lambda values: ", ".join(sorted(set(map(str, values))))),
        )
    )

    candidate_path = OUTPUT_DIR / "gtx_c_uijeongbu_yangju_metropolitan_bus_candidates.csv"
    candidate_df.to_csv(candidate_path, index=False, encoding="utf-8-sig")

    print(f"캐시 노선정보: {len(route_info):,}개")
    print(f"캐시 정류장 노선: {len(route_stops):,}개")
    print("\n===== GTX-C 북부 서울방향 광역버스 후보 =====")
    print(candidate_df.to_string(index=False))
    print("\n===== 역별 요약 =====")
    print(summary.to_string(index=False))
    print("\n저장:")
    print(candidate_path)


if __name__ == "__main__":
    main()
