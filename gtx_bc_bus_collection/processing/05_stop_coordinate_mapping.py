"""STCIS 정류장에 TAGO 캐시 좌표를 붙이고 교통카드 OD에 결합한다."""

from __future__ import annotations

from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
REPO = PROJECT_DIR.parent
CACHE_DIR = REPO / "Desktop" / "Gachon" / "3-2" / "UROP" / "api_cache_gtx_a_routes"
NATIONAL_STOP_FILE = REPO.parent / "국토교통부_전국 버스정류장 위치정보_20251031.csv"
ROUTE_FILE = PROJECT_DIR / "processed" / "gtx_c_uijeongbu_yangju_stcis_routes.csv"
STCIS_STOP_FILE = PROJECT_DIR / "raw" / "route_stops" / "gtx_c_route_stops.csv"
RAW_STOP_DIR = PROJECT_DIR / "raw" / "route_stops"
PROCESSED_CARD_DIR = PROJECT_DIR / "processed" / "transport_card"
DATES = ["20241017", "20251106"]


def api_items(payload: dict) -> list[dict]:
    response = payload.get("response", payload)
    body = response.get("body", {}) if isinstance(response, dict) else {}
    items = body.get("items", {}) if isinstance(body, dict) else {}
    items = items.get("item", []) if isinstance(items, dict) else items
    if isinstance(items, dict):
        return [items]
    return items if isinstance(items, list) else []


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFC", str(value))
    value = re.sub(r"\((?:중|미정차|경유)\)", "", value)
    return re.sub(r"[^0-9A-Za-z가-힣]", "", value).lower()


def load_tago_stops(routes: pd.DataFrame) -> pd.DataFrame:
    cached_routes: dict[str, list[dict]] = {}
    for path in CACHE_DIR.glob("getRouteAcctoThrghSttnList_*.json"):
        rows = api_items(json.loads(path.read_text(encoding="utf-8")))
        if rows:
            cached_routes[str(rows[0].get("routeid", "")).strip()] = rows

    records: list[dict] = []
    for route in routes.itertuples(index=False):
        for stop in cached_routes.get(route.TAGO_노선ID, []):
            try:
                latitude = float(stop.get("gpslati"))
                longitude = float(stop.get("gpslong"))
                sequence = int(stop.get("nodeord"))
            except (TypeError, ValueError):
                continue
            records.append(
                {
                    "query_route_id": route.STCIS_노선ID,
                    "query_route_no": route.버스번호,
                    "TAGO_노선ID": route.TAGO_노선ID,
                    "TAGO_정류장ID": str(stop.get("nodeid", "")).strip(),
                    "TAGO_정류장명": str(stop.get("nodenm", "")).strip(),
                    "TAGO_정류장순서": sequence,
                    "정류장명키": normalize_name(stop.get("nodenm", "")),
                    "위도": latitude,
                    "경도": longitude,
                }
            )
    result = pd.DataFrame(records).sort_values(["query_route_id", "TAGO_정류장순서"])
    result["동일명_등장순서"] = result.groupby(["query_route_id", "정류장명키"]).cumcount()
    return result


def map_stcis_to_tago(stcis: pd.DataFrame, tago: pd.DataFrame) -> pd.DataFrame:
    result = stcis.copy()
    result = result[result["query_route_id"].isin(set(tago["query_route_id"]))].copy()
    result["STCIS_정류장순서"] = pd.to_numeric(result["sttnSeq"], errors="coerce")
    result["정류장명키"] = result["sttnNm"].map(normalize_name)
    result = result.sort_values(["query_route_id", "STCIS_정류장순서"])
    result["동일명_등장순서"] = result.groupby(["query_route_id", "정류장명키"]).cumcount()

    tago_columns = [
        "query_route_id", "정류장명키", "동일명_등장순서", "TAGO_노선ID",
        "TAGO_정류장ID", "TAGO_정류장명", "TAGO_정류장순서", "위도", "경도",
    ]
    result = result.merge(
        tago[tago_columns],
        on=["query_route_id", "정류장명키", "동일명_등장순서"],
        how="left",
    )
    result["좌표매칭방법"] = result["위도"].notna().map({True: "노선·정류장명·등장순서", False: ""})

    # 명칭이 조금 달라 정확히 연결되지 않은 정류장은 같은 노선의 인접 순서 중
    # 이름 유사도가 가장 높은 정류장으로 보완한다.
    for index in result.index[result["위도"].isna()]:
        route_id = result.at[index, "query_route_id"]
        stop_name = result.at[index, "sttnNm"]
        stop_sequence = result.at[index, "STCIS_정류장순서"]
        candidates = tago[tago["query_route_id"].eq(route_id)].copy()
        candidates["순서차이"] = (candidates["TAGO_정류장순서"] - stop_sequence).abs()
        candidates = candidates[candidates["순서차이"].le(3)].copy()
        if candidates.empty:
            continue
        candidates["이름유사도"] = candidates["TAGO_정류장명"].map(
            lambda name: SequenceMatcher(None, normalize_name(stop_name), normalize_name(name)).ratio()
        )
        best = candidates.sort_values(["이름유사도", "순서차이"], ascending=[False, True]).iloc[0]
        if best["이름유사도"] < 0.6:
            continue
        for column in [
            "TAGO_노선ID", "TAGO_정류장ID", "TAGO_정류장명",
            "TAGO_정류장순서", "위도", "경도",
        ]:
            result.at[index, column] = best[column]
        result.at[index, "좌표매칭방법"] = "노선·인접순서·이름유사도"

    # TAGO 정류장 ID는 국토부 자료의 정류장번호와 동일하다. 기준일에 가까운
    # 국토부 2025-10-31 좌표를 최종 좌표로 사용하고 ARS 번호도 함께 보존한다.
    if NATIONAL_STOP_FILE.exists():
        national = pd.read_csv(NATIONAL_STOP_FILE, dtype=str, encoding="cp949").fillna("")
        national = national.rename(
            columns={
                "정류장번호": "TAGO_정류장ID",
                "정류장명": "국토부_정류장명",
                "위도": "국토부_위도",
                "경도": "국토부_경도",
                "정보수집일": "국토부_정보수집일",
                "모바일단축번호": "ARS번호",
                "도시명": "국토부_도시명",
            }
        )
        official_columns = [
            "TAGO_정류장ID", "국토부_정류장명", "국토부_위도", "국토부_경도",
            "국토부_정보수집일", "ARS번호", "국토부_도시명",
        ]
        national = national[official_columns].drop_duplicates("TAGO_정류장ID")
        result = result.merge(national, on="TAGO_정류장ID", how="left", validate="many_to_one")
        result["국토부_위도"] = pd.to_numeric(result["국토부_위도"], errors="coerce")
        result["국토부_경도"] = pd.to_numeric(result["국토부_경도"], errors="coerce")
        official_match = result["국토부_위도"].notna() & result["국토부_경도"].notna()
        result.loc[official_match, "위도"] = result.loc[official_match, "국토부_위도"]
        result.loc[official_match, "경도"] = result.loc[official_match, "국토부_경도"]
        result.loc[official_match, "좌표매칭방법"] = "국토부 정류장번호"
        result = result.drop(columns=["국토부_위도", "국토부_경도"])
    return result


def attach_coordinates(od: pd.DataFrame, stop_map: pd.DataFrame) -> pd.DataFrame:
    reference = stop_map[
        ["query_route_id", "sttnId", "위도", "경도", "좌표매칭방법"]
    ].drop_duplicates(["query_route_id", "sttnId"])
    ride = reference.rename(
        columns={
            "sttnId": "승차정류장ID",
            "위도": "승차위도",
            "경도": "승차경도",
            "좌표매칭방법": "승차좌표매칭방법",
        }
    )
    goff = reference.rename(
        columns={
            "sttnId": "하차정류장ID",
            "위도": "하차위도",
            "경도": "하차경도",
            "좌표매칭방법": "하차좌표매칭방법",
        }
    )
    result = od.merge(ride, on=["query_route_id", "승차정류장ID"], how="left")
    return result.merge(goff, on=["query_route_id", "하차정류장ID"], how="left")


def main() -> None:
    routes = pd.read_csv(ROUTE_FILE, dtype=str, encoding="utf-8-sig").fillna("")
    stcis = pd.read_csv(STCIS_STOP_FILE, dtype=str, encoding="utf-8-sig").fillna("")
    tago = load_tago_stops(routes)
    stop_map = map_stcis_to_tago(stcis, tago)

    stop_map_path = RAW_STOP_DIR / "gtx_c_route_stops_with_coordinates.csv"
    stop_map.to_csv(stop_map_path, index=False, encoding="utf-8-sig")
    print(
        f"정류장 좌표: {stop_map['위도'].notna().sum():,}/{len(stop_map):,} "
        f"({stop_map['위도'].notna().mean():.1%})"
    )

    summaries: list[dict] = []
    for date in DATES:
        source = PROCESSED_CARD_DIR / f"gtx_c_transport_card_{date}_seoul_od_with_stop_names.csv"
        od = pd.read_csv(source, dtype=str, encoding="utf-8-sig").fillna("")
        result = attach_coordinates(od, stop_map)
        output = PROCESSED_CARD_DIR / f"gtx_c_transport_card_{date}_seoul_od_with_coordinates.csv"
        result.to_csv(output, index=False, encoding="utf-8-sig")
        ride_rate = result["승차위도"].notna().mean() if len(result) else 0
        goff_rate = result["하차위도"].notna().mean() if len(result) else 0
        summaries.append(
            {"기준일": date, "OD행수": len(result), "승차좌표매칭률": ride_rate, "하차좌표매칭률": goff_rate}
        )
        print(f"{date}: 승차 {ride_rate:.1%}, 하차 {goff_rate:.1%} / {output}")

    summary = pd.DataFrame(summaries)
    summary_path = PROCESSED_CARD_DIR / "gtx_c_stop_coordinate_mapping_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
