"""GTX-C 북부 후보 버스번호의 STCIS 노선 ID 후보를 조회한다.

01 단계에서 만든 전체 후보 노선 CSV를 읽고, 교통카드 합성데이터 API에
필요한 STCIS 노선 ID를 조회한다. 같은 번호에 여러 노선이 검색될 수 있으므로
기점·종점을 포함한 모든 후보를 저장하며 임의로 하나를 확정하지 않는다.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from getpass import getpass
import os
from pathlib import Path
import re
import time

import pandas as pd
import requests


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
PROCESSED_DIR = PROJECT_DIR / "processed"
INPUT_FILE = PROCESSED_DIR / "gtx_c_uijeongbu_yangju_metropolitan_bus_candidates.csv"
OUTPUT_FILE = PROCESSED_DIR / "gtx_c_uijeongbu_yangju_stcis_routes.csv"

STCIS_URL = "https://stcis.go.kr/openapi/busroute.json"
SD_CD = "41"  # 경기도
FUTURE_ROUTES = {"1108", "1407"}  # 2026년 개통: 2025-11-06 기준 분석에서 제외


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GTX-C 후보 노선의 STCIS 노선 ID 조회")
    parser.add_argument(
        "--api-key",
        default=os.getenv("STCIS_API_KEY", ""),
        help="STCIS 인증키. 생략하면 STCIS_API_KEY 환경변수 또는 보안 입력을 사용합니다.",
    )
    parser.add_argument("--delay", type=float, default=0.2, help="노선별 요청 간격(초)")
    return parser.parse_args()


def walk_dicts(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_dicts(child)


def first_value(item: dict, keys: list[str]) -> str:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def find_route_records(payload: dict) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in walk_dicts(payload):
        route_id = first_value(item, ["routeId", "rteId", "rte_id", "routeID"])
        if not route_id:
            continue
        record = {
            "STCIS_노선ID": route_id,
            "STCIS_노선번호": first_value(item, ["routeNo", "route_no"]),
            "기점_STCIS": first_value(item, ["stgSttnNma", "stgSttnNm", "startStation"]),
            "종점_STCIS": first_value(item, ["arrSttnNma", "arrSttnNm", "endStation"]),
        }
        key = tuple(record.values())
        if key not in seen:
            seen.add(key)
            records.append(record)
    return records


def query_route(api_key: str, route_no: str) -> list[dict[str, str]]:
    response = requests.get(
        STCIS_URL,
        params={"apikey": api_key, "sdCd": SD_CD, "routeNo": route_no},
        timeout=30,
    )
    response.raise_for_status()
    return find_route_records(response.json())


def query_number(bus_no: str) -> str:
    """P9601(출근)처럼 설명이 붙은 번호는 실제 노선번호만 조회한다."""
    return bus_no.split("(", 1)[0].strip()


def normalize_terminal(value: str) -> str:
    value = re.sub(r"\([^)]*\)", "", str(value))
    return re.sub(r"[^0-9A-Za-z가-힣]", "", value)


def terminal_match_score(tago_terminal: str, stcis_terminal: str) -> float:
    """표기 부호와 '(미정차)'를 제거한 기점명 유사도를 계산한다."""
    tago = normalize_terminal(tago_terminal)
    stcis = normalize_terminal(stcis_terminal)
    if not tago or not stcis:
        return 0.0
    return SequenceMatcher(None, tago, stcis).ratio()


def main() -> None:
    args = parse_args()
    api_key = args.api_key.strip() or getpass("STCIS 인증키를 입력하세요: ").strip()
    if not api_key:
        raise ValueError("STCIS 인증키가 입력되지 않았습니다.")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"01 단계 후보 노선 파일이 없습니다: {INPUT_FILE}")

    routes = pd.read_csv(INPUT_FILE, dtype=str, encoding="utf-8-sig").fillna("")
    routes["버스번호"] = routes["버스번호"].str.strip()
    excluded = (
        routes["버스번호"].str.endswith("N")
        | routes["버스번호"].str.contains(r"\((?:출근|예약)\)", regex=True)
        | routes["버스번호"].isin(FUTURE_ROUTES)
    )
    routes = routes.loc[~excluded].copy()
    routes["STCIS_조회번호"] = routes["버스번호"].map(query_number)

    # 같은 버스가 두 GTX 역 후보에 포함되어도 API는 한 번만 호출한다.
    unique_routes = (
        routes.groupby(["버스번호", "STCIS_조회번호"], as_index=False)
        .agg(
            GTX_C_역=("GTX_C_역", lambda x: ", ".join(sorted(set(x)))),
            대상지역=("대상지역", lambda x: ", ".join(sorted(set(x)))),
            TAGO_노선ID=("노선ID", lambda x: ", ".join(sorted(set(x)))),
            기점_TAGO=("기점", "first"),
            종점_TAGO=("종점", "first"),
        )
        .sort_values(["버스번호", "TAGO_노선ID"])
    )

    results: list[dict[str, str | int]] = []
    total = len(unique_routes)
    for index, route in enumerate(unique_routes.itertuples(index=False), start=1):
        base = {
            "GTX_C_역": route.GTX_C_역,
            "대상지역": route.대상지역,
            "버스번호": route.버스번호,
            "STCIS_조회번호": route.STCIS_조회번호,
            "TAGO_노선ID": route.TAGO_노선ID,
            "기점_TAGO": route.기점_TAGO,
            "종점_TAGO": route.종점_TAGO,
        }
        try:
            candidates = query_route(api_key, route.STCIS_조회번호)
            if candidates:
                for candidate_no, candidate in enumerate(candidates, start=1):
                    results.append(
                        base
                        | candidate
                        | {"STCIS_후보번호": candidate_no, "조회상태": "후보 확인", "오류": ""}
                    )
                print(f"[{index}/{total}] {route.버스번호}: STCIS 후보 {len(candidates)}개")
            else:
                results.append(
                    base
                    | {
                        "STCIS_노선ID": "",
                        "STCIS_노선번호": "",
                        "기점_STCIS": "",
                        "종점_STCIS": "",
                        "STCIS_후보번호": 0,
                        "조회상태": "노선ID 미확인",
                        "오류": "",
                    }
                )
                print(f"[{index}/{total}] {route.버스번호}: 노선ID 미확인")
        except Exception as exc:
            results.append(
                base
                | {
                    "STCIS_노선ID": "",
                    "STCIS_노선번호": "",
                    "기점_STCIS": "",
                    "종점_STCIS": "",
                    "STCIS_후보번호": 0,
                    "조회상태": "조회 실패",
                    "오류": repr(exc),
                }
            )
            print(f"[{index}/{total}] {route.버스번호}: 조회 실패 - {exc}")
        time.sleep(max(args.delay, 0))

    result = pd.DataFrame(results)
    result["기점일치점수"] = result.apply(
        lambda row: terminal_match_score(row["기점_TAGO"], row["기점_STCIS"]), axis=1
    )
    # 같은 번호의 STCIS 후보 중 TAGO 기점과 가장 잘 일치하는 노선을 확정한다.
    selected_index = result.groupby("버스번호")["기점일치점수"].idxmax()
    result = result.loc[selected_index].copy()
    result["선정근거"] = result["기점일치점수"].map(
        lambda score: "TAGO·STCIS 기점 일치" if score == 1 else "기점 유사도 최고 후보"
    )
    result = result[
        [
            "GTX_C_역", "대상지역", "버스번호", "TAGO_노선ID", "STCIS_노선ID",
            "기점_TAGO", "종점_TAGO", "기점_STCIS", "종점_STCIS",
            "기점일치점수", "선정근거", "조회상태", "오류",
        ]
    ].sort_values(["버스번호", "STCIS_노선ID"]).reset_index(drop=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n===== 조회 결과 =====")
    print(f"고유 후보 노선: {total}개")
    print(f"STCIS ID 확인 노선: {result.loc[result['STCIS_노선ID'].ne(''), '버스번호'].nunique()}개")
    print(f"미확인·실패 노선: {result.loc[result['STCIS_노선ID'].eq(''), '버스번호'].nunique()}개")
    print(f"저장: {OUTPUT_FILE}")
    print("같은 번호의 후보 중 TAGO 기점과 가장 잘 일치하는 STCIS 노선을 선정했습니다.")


if __name__ == "__main__":
    main()
