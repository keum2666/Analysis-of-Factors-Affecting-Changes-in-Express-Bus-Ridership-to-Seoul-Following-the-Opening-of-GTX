"""GTX-C 북부 17개 일반노선의 교통카드 합성데이터를 수집한다.

수집일: 2024-10-17, 2025-11-06
범위: 경기도 승차(41) 후 서울특별시 하차(11), 일반 이용자(01)
"""

from __future__ import annotations

import argparse
from getpass import getpass
import os
from pathlib import Path
import socket
import time

import pandas as pd
import requests


HERE = Path(__file__).resolve().parent
PROJECT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir())
BC_RAW = PROJECT_DIR / "data" / "raw" / "gtx_bc"
BC_PROCESSED = PROJECT_DIR / "data" / "processed" / "gtx_bc"
WORKSPACE = PROJECT_DIR
ROUTE_FILE = BC_PROCESSED / "gtx_c_uijeongbu_yangju_stcis_routes.csv"
RAW_DIR = BC_RAW / "transport_card"
PROCESSED_DIR = BC_PROCESSED / "transport_card"
ENV_FILE = WORKSPACE / ".env"

BASE_URL = (
    "https://apis.data.go.kr/1613000/RegionalTransportationCardUsageSyntheticData/"
    "getGyeonggiTransportationCardUsageSyntheticData"
)
DATES = ["20241017", "20251106"]
PAGE_SIZE = 1000
RIDE_CTPV_CD = "41"
SEOUL_CTPV_CD = "11"
GOFF_CTPV_COLUMNS = ["goff_ctpv_cd", "alight_ctpv_cd", "arr_ctpv_cd", "arrive_ctpv_cd"]


def env_value(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value or not ENV_FILE.exists():
        return value
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, raw_value = stripped.split("=", 1)
            if key.strip() == name:
                return raw_value.strip().strip('"').strip("'")
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GTX-C 북부 교통카드 합성데이터 수집")
    parser.add_argument(
        "--api-key",
        default=env_value("DATA_GO_KR_SERVICE_KEY"),
        help="공공데이터포털 인증키. 기본값은 환경변수 또는 UROP/.env입니다.",
    )
    parser.add_argument("--delay", type=float, default=0.2, help="노선별 호출 간격(초)")
    return parser.parse_args()


def parse_response(payload: dict) -> tuple[list[dict], dict]:
    response = payload.get("response", payload.get("Response", payload))
    header = response.get("header", {}) or {}
    body = response.get("body", {}) or {}
    items = body.get("items", {}) or {}
    items = items.get("item", []) if isinstance(items, dict) else items
    code = str(header.get("resultCode", header.get("resultcode", "")))
    if code not in ("", "00", "0", "200"):
        raise RuntimeError(f"API resultCode={code}: {header}")
    if isinstance(items, dict):
        items = [items]
    return items or [], body


def get_page(session: requests.Session, key: str, route_id: str, date: str, page: int):
    response = session.get(
        BASE_URL,
        params={
            "serviceKey": key,
            "pageNo": page,
            "numOfRows": PAGE_SIZE,
            "dataType": "JSON",
            "opr_ymd": date,
            "rte_id": route_id,
            "users_type_cd": "01",
            "ride_ctpv_cd": RIDE_CTPV_CD,
        },
        timeout=60,
    )
    response.raise_for_status()
    return parse_response(response.json())


def safe_error(exc: Exception, key: str) -> str:
    """오류 메시지나 조회 로그에 인증키가 기록되지 않도록 가린다."""
    return repr(exc).replace(key, "***") if key else repr(exc)


def collect_date(
    session: requests.Session, key: str, routes: pd.DataFrame, date: str, delay: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows: list[dict] = []
    logs: list[dict] = []
    total_routes = len(routes)

    for index, route in enumerate(routes.itertuples(index=False), start=1):
        route_id = str(route.STCIS_노선ID).strip()
        route_no = str(route.버스번호).strip()
        rows: list[dict] = []
        page = 1
        try:
            while True:
                items, body = get_page(session, key, route_id, date, page)
                rows.extend(items)
                total_count = int(body.get("totalCount", body.get("totalcount", 0)) or 0)
                if not items or (total_count and len(rows) >= total_count) or len(items) < PAGE_SIZE:
                    break
                page += 1

            for item in rows:
                item.update(
                    {
                        "query_date": date,
                        "query_route_id": route_id,
                        "query_route_no": route_no,
                        "GTX_C_역": route.GTX_C_역,
                        "대상지역": route.대상지역,
                    }
                )
            all_rows.extend(rows)
            logs.append(
                {
                    "query_date": date,
                    "route_no": route_no,
                    "route_id": route_id,
                    "status": "success",
                    "api_rows": len(rows),
                    "pages": page,
                    "error": "",
                }
            )
            print(f"[{date} {index}/{total_routes}] {route_no}: {len(rows):,}건")
        except Exception as exc:
            logs.append(
                {
                    "query_date": date,
                    "route_no": route_no,
                    "route_id": route_id,
                    "status": "failed",
                    "api_rows": 0,
                    "pages": page,
                    "error": safe_error(exc, key),
                }
            )
            print(f"[{date} {index}/{total_routes}] {route_no}: 실패 - {safe_error(exc, key)}")
        time.sleep(max(delay, 0))

    return pd.DataFrame(all_rows), pd.DataFrame(logs)


def filter_seoul_alighting(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw.copy()
    column = next((name for name in GOFF_CTPV_COLUMNS if name in raw.columns), None)
    if column is None:
        raise KeyError(f"서울 하차 시도코드 필드를 찾지 못했습니다: {raw.columns.tolist()}")
    result = raw.copy()
    result[column] = result[column].astype(str).str.strip().str.zfill(2)
    return result[result[column].eq(SEOUL_CTPV_CD)].copy()


def build_od(seoul_rows: pd.DataFrame) -> pd.DataFrame:
    if seoul_rows.empty:
        return pd.DataFrame()
    required = ["ride_sttn_id", "goff_sttn_id", "utztn_nope"]
    missing = [column for column in required if column not in seoul_rows.columns]
    if missing:
        raise KeyError(f"OD 생성 필수 필드가 없습니다: {missing}")
    od = seoul_rows.copy()
    od["utztn_nope"] = pd.to_numeric(od["utztn_nope"], errors="coerce").fillna(0)
    group_columns = [
        "query_date", "query_route_id", "query_route_no", "GTX_C_역", "대상지역",
        "ride_ctpv_cd", "goff_ctpv_cd", "ride_sttn_id", "goff_sttn_id",
    ]
    existing_group_columns = [column for column in group_columns if column in od.columns]
    od = (
        od.groupby(existing_group_columns, as_index=False, dropna=False)
        .agg(통행인원=("utztn_nope", "sum"), 원본행수=("utztn_nope", "size"))
        .rename(columns={"ride_sttn_id": "승차정류장ID", "goff_sttn_id": "하차정류장ID"})
    )
    return od


def main() -> None:
    args = parse_args()
    key = args.api_key.strip() or getpass("공공데이터포털 인증키를 입력하세요: ").strip()
    if not key:
        raise ValueError("공공데이터포털 인증키가 없습니다.")
    if not ROUTE_FILE.exists():
        raise FileNotFoundError(f"STCIS 노선 파일이 없습니다: {ROUTE_FILE}")
    try:
        socket.gethostbyname("apis.data.go.kr")
    except OSError as exc:
        raise RuntimeError(
            "apis.data.go.kr 주소를 확인할 수 없습니다. 인터넷 연결이 가능한 Terminal에서 다시 실행하세요."
        ) from exc

    routes = pd.read_csv(ROUTE_FILE, dtype=str, encoding="utf-8-sig").fillna("")
    routes = routes[routes["STCIS_노선ID"].str.strip().ne("")].copy()
    routes = routes.drop_duplicates("STCIS_노선ID").reset_index(drop=True)
    if len(routes) != 15:
        print(f"주의: 예상한 15개와 달리 조회 대상은 {len(routes)}개입니다.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    summaries: list[dict] = []

    for date in DATES:
        print(f"\n===== {date} 수집 =====")
        raw, log = collect_date(session, key, routes, date, args.delay)
        raw_path = RAW_DIR / f"gtx_c_transport_card_{date}_raw.csv"
        log_path = PROCESSED_DIR / f"gtx_c_transport_card_{date}_query_log.csv"
        raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

        seoul = filter_seoul_alighting(raw)
        if not log.empty:
            counts = seoul.groupby("query_route_id").size() if not seoul.empty else pd.Series(dtype=int)
            log["seoul_alighting_rows"] = log["route_id"].map(counts).fillna(0).astype(int)
        log.to_csv(log_path, index=False, encoding="utf-8-sig")

        od = build_od(seoul)
        od_path = PROCESSED_DIR / f"gtx_c_transport_card_{date}_seoul_od.csv"
        od.to_csv(od_path, index=False, encoding="utf-8-sig")

        summaries.append(
            {
                "기준일": date,
                "조회노선수": len(routes),
                "성공노선수": int(log["status"].eq("success").sum()),
                "실패노선수": int(log["status"].eq("failed").sum()),
                "API원본행수": len(raw),
                "서울하차행수": len(seoul),
                "서울OD행수": len(od),
                "서울하차통행인원": float(od["통행인원"].sum()) if not od.empty else 0,
            }
        )
        print(f"원본: {raw_path} ({len(raw):,}행)")
        print(f"서울 하차 OD: {od_path} ({len(od):,}행)")

    summary = pd.DataFrame(summaries)
    summary_path = PROCESSED_DIR / "gtx_c_transport_card_collection_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\n===== 수집 요약 =====")
    print(summary.to_string(index=False))
    print(f"요약 저장: {summary_path}")


if __name__ == "__main__":
    main()
