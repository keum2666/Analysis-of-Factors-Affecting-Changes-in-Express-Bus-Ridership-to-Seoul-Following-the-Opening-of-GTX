"""GTX-C 교통카드 승·하차 정류장 ID에 한글 정류장명과 순서를 매칭한다."""

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
ROUTE_FILE = BC_PROCESSED / "gtx_c_uijeongbu_yangju_stcis_routes.csv"
RAW_CARD_DIR = BC_RAW / "transport_card"
RAW_STOP_DIR = BC_RAW / "route_stops"
PROCESSED_CARD_DIR = BC_PROCESSED / "transport_card"

API_URL = "https://stcis.go.kr/openapi/busroutesttn.json"
REGION_CODES = ["41", "11"]  # 경기도, 서울특별시
DATES = ["20241017", "20251106"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GTX-C 교통카드 정류장명 매칭")
    parser.add_argument(
        "--api-key",
        default=os.getenv("STCIS_API_KEY", ""),
        help="STCIS 인증키. 생략하면 보안 입력창에서 받습니다.",
    )
    parser.add_argument("--delay", type=float, default=0.2, help="API 호출 간격(초)")
    return parser.parse_args()


def parse_response(payload: dict) -> tuple[list[dict], str]:
    result = payload.get("result", [])
    if isinstance(result, dict):
        result = [result]
    return (result if isinstance(result, list) else []), str(payload.get("status", ""))


def safe_error(exc: Exception, key: str) -> str:
    return repr(exc).replace(key, "***") if key else repr(exc)


def collect_route_stops(routes: pd.DataFrame, key: str, delay: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []
    logs: list[dict] = []
    session = requests.Session()

    for index, route in enumerate(routes.itertuples(index=False), start=1):
        route_id = str(route.STCIS_노선ID).strip()
        route_no = str(route.버스번호).strip()
        collected: list[dict] = []
        statuses: list[str] = []
        errors: list[str] = []

        for region_code in REGION_CODES:
            try:
                response = session.get(
                    API_URL,
                    params={"apikey": key, "sdCd": region_code, "routeId": route_id},
                    timeout=60,
                )
                response.raise_for_status()
                items, status = parse_response(response.json())
                statuses.append(f"{region_code}:{status or 'OK'}")
                collected.extend(items)
            except Exception as exc:
                errors.append(f"{region_code}:{safe_error(exc, key)}")
            time.sleep(max(delay, 0))

        seen: set[tuple[str, str]] = set()
        kept = 0
        for item in collected:
            stop_id = str(item.get("sttnId", "")).strip()
            stop_sequence = str(item.get("sttnSeq", "")).strip()
            stop_key = (stop_id, stop_sequence)
            if not stop_id or stop_key in seen:
                continue
            seen.add(stop_key)
            row = dict(item)
            row.update({"query_route_id": route_id, "query_route_no": route_no})
            records.append(row)
            kept += 1

        logs.append(
            {
                "route_no": route_no,
                "route_id": route_id,
                "status": ";".join(statuses) if statuses else "ERROR",
                "stop_count": kept,
                "error": "; ".join(errors),
            }
        )
        print(f"[{index}/{len(routes)}] {route_no} ({route_id}): {kept}개 정류장")

    stops = pd.DataFrame(records)
    if not stops.empty:
        stops["_sequence"] = pd.to_numeric(stops.get("sttnSeq"), errors="coerce")
        stops = (
            stops.drop_duplicates(["query_route_id", "sttnId", "sttnSeq"])
            .sort_values(["query_route_no", "query_route_id", "_sequence"])
            .drop(columns="_sequence")
            .reset_index(drop=True)
        )
    return stops, pd.DataFrame(logs)


def stop_reference(stops: pd.DataFrame) -> pd.DataFrame:
    required = ["query_route_id", "sttnId", "sttnNm", "sttnSeq"]
    missing = [column for column in required if column not in stops.columns]
    if missing:
        raise KeyError(f"정류장 응답에 필수 컬럼이 없습니다: {missing}")
    ref = stops[required].copy()
    for column in ["query_route_id", "sttnId", "sttnNm", "sttnSeq"]:
        ref[column] = ref[column].astype(str).str.strip()
    return ref.drop_duplicates(["query_route_id", "sttnId"])


def map_stop_names(data: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    id_pairs = [("ride_sttn_id", "goff_sttn_id"), ("승차정류장ID", "하차정류장ID")]
    ride_id, goff_id = next(
        ((ride, goff) for ride, goff in id_pairs if ride in result.columns and goff in result.columns),
        (None, None),
    )
    if ride_id is None:
        raise KeyError("승·하차 정류장 ID 컬럼을 찾지 못했습니다.")
    for column in ["query_route_id", ride_id, goff_id]:
        result[column] = result[column].astype(str).str.strip()

    ride_ref = ref.rename(
        columns={
            "sttnId": ride_id,
            "sttnNm": "승차정류장명",
            "sttnSeq": "승차정류장순서",
        }
    )
    goff_ref = ref.rename(
        columns={
            "sttnId": goff_id,
            "sttnNm": "하차정류장명",
            "sttnSeq": "하차정류장순서",
        }
    )
    result = result.merge(ride_ref, on=["query_route_id", ride_id], how="left")
    result = result.merge(goff_ref, on=["query_route_id", goff_id], how="left")

    front = [
        "query_date", "query_route_no", "query_route_id",
        ride_id, "승차정류장명", "승차정류장순서",
        goff_id, "하차정류장명", "하차정류장순서",
    ]
    front = [column for column in front if column in result.columns]
    return result[front + [column for column in result.columns if column not in front]]


def main() -> None:
    args = parse_args()
    key = args.api_key.strip() or getpass("STCIS 인증키를 입력하세요: ").strip()
    if not key:
        raise ValueError("STCIS 인증키가 없습니다.")
    try:
        socket.gethostbyname("stcis.go.kr")
    except OSError as exc:
        raise RuntimeError("stcis.go.kr에 연결할 수 있는 Terminal에서 실행하세요.") from exc

    routes = pd.read_csv(ROUTE_FILE, dtype=str, encoding="utf-8-sig").fillna("")
    routes = routes.drop_duplicates("STCIS_노선ID").reset_index(drop=True)
    stops, logs = collect_route_stops(routes, key, args.delay)
    if stops.empty:
        raise RuntimeError("조회된 정류장이 없습니다. STCIS 인증키와 노선 ID를 확인하세요.")

    RAW_STOP_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_CARD_DIR.mkdir(parents=True, exist_ok=True)
    stop_path = RAW_STOP_DIR / "gtx_c_route_stops.csv"
    log_path = PROCESSED_CARD_DIR / "gtx_c_route_stops_query_log.csv"
    stops.to_csv(stop_path, index=False, encoding="utf-8-sig")
    logs.to_csv(log_path, index=False, encoding="utf-8-sig")
    ref = stop_reference(stops)

    summaries: list[dict] = []
    valid_ids = set(routes["STCIS_노선ID"].astype(str).str.strip())
    for date in DATES:
        raw_path = RAW_CARD_DIR / f"gtx_c_transport_card_{date}_raw.csv"
        od_path = PROCESSED_CARD_DIR / f"gtx_c_transport_card_{date}_seoul_od.csv"
        for source_path, label in [(raw_path, "raw"), (od_path, "seoul_od")]:
            data = pd.read_csv(source_path, dtype=str, encoding="utf-8-sig").fillna("")
            data = data[data["query_route_id"].astype(str).str.strip().isin(valid_ids)].copy()
            mapped = map_stop_names(data, ref)
            output_path = source_path.with_name(source_path.stem + "_with_stop_names.csv")
            mapped.to_csv(output_path, index=False, encoding="utf-8-sig")
            ride_rate = mapped["승차정류장명"].fillna("").ne("").mean() if len(mapped) else 0
            goff_rate = mapped["하차정류장명"].fillna("").ne("").mean() if len(mapped) else 0
            summaries.append(
                {
                    "기준일": date,
                    "자료": label,
                    "행수": len(mapped),
                    "승차정류장명_매칭률": ride_rate,
                    "하차정류장명_매칭률": goff_rate,
                    "출력파일": str(output_path),
                }
            )
            print(f"{date} {label}: 승차 {ride_rate:.1%}, 하차 {goff_rate:.1%}")

    summary = pd.DataFrame(summaries)
    summary_path = PROCESSED_CARD_DIR / "gtx_c_stop_name_mapping_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\n===== 정류장명 매칭 결과 =====")
    print(summary.drop(columns="출력파일").to_string(index=False))
    print(f"정류장 원자료: {stop_path}")
    print(f"매칭 요약: {summary_path}")


if __name__ == "__main__":
    main()
