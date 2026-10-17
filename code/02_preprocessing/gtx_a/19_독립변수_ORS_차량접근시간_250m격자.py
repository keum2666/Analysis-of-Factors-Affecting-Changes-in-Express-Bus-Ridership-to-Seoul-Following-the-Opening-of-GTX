"""ORS driving-car로 126개 분석격자의 중심점→배정 GTX-A역 차량시간을 조회한다.

실행: python3 processing/19_독립변수_ORS_차량접근시간_250m격자.py
API 키는 ORS_API_KEY 환경변수 또는 실행 중 숨김 입력으로 받는다.
--dry-run은 API를 호출하거나 결과 파일을 만들지 않고 입력만 확인한다.
"""

import argparse
import csv
import getpass
import json
import math
import os
import socket
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir())
PROCESSED = ROOT / "data" / "processed" / "gtx_a"
INPUT = next(PROCESSED.glob("18_새데이터_회귀분석_격자통합데이터.csv"))
OUTPUT = PROCESSED / "26_ORS_차량접근시간_250m격자.csv"
URL = "https://api.heigit.org/openrouteservice/v2/directions/driving-car"

# 기존 보행 접근성 구축에 사용한 GTX-A역 좌표와 동일하게 고정한다.
STATIONS = {
    "운정중앙": (37.716670, 126.728330),
    "킨텍스": (37.665000, 126.748060),
    "대곡": (37.631626, 126.811024),
}
FIELDS = [
    "셀 ID", "nearest_station", "중심점 위도", "중심점 경도",
    "역 위도", "역 경도", "drive_min", "drive_km", "조회상태",
]


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def save_csv(rows):
    temporary = OUTPUT.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(OUTPUT)


def get_route(api_key, start_lat, start_lon, end_lat, end_lon):
    query = urlencode({
        "start": f"{start_lon:.8f},{start_lat:.8f}",
        "end": f"{end_lon:.8f},{end_lat:.8f}",
    })
    request = Request(
        f"{URL}?{query}",
        headers={"Authorization": api_key, "Accept": "application/geo+json"},
    )
    for attempt in range(4):
        try:
            with urlopen(request, timeout=45) as response:
                data = json.load(response)
            summary = data["features"][0]["properties"]["summary"]
            return float(summary["duration"]) / 60, float(summary["distance"]) / 1000, "성공"
        except HTTPError as error:
            # 인증/좌표 오류는 재시도해도 해결되지 않는다. 응답 본문은 키가 포함될 수 있어 출력하지 않는다.
            if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                return "", "", f"HTTP {error.code}"
        except URLError as error:
            if isinstance(error.reason, socket.gaierror):
                return "", "", "DNS 연결 실패"
            if attempt == 3:
                return "", "", "네트워크 오류"
        except TimeoutError:
            if attempt == 3:
                return "", "", "네트워크 오류"
        except (KeyError, IndexError, ValueError, TypeError):
            return "", "", "응답 형식 오류"
        time.sleep(min(2 ** (attempt + 1), 16))
    return "", "", "조회 실패"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="입력 검증만 수행")
    parser.add_argument("--retry-failed", action="store_true", help="이전 실패 격자 재조회")
    args = parser.parse_args()

    source = read_csv(INPUT)
    if len(source) != 126 or len({row["셀 ID"] for row in source}) != 126:
        raise ValueError("분석자료가 고유한 126개 격자인지 확인하세요.")
    for row in source:
        if row["nearest_station"] not in STATIONS:
            raise ValueError(f"알 수 없는 배정역: {row['nearest_station']}")
        for field in ("중심점 위도", "중심점 경도"):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"좌표 오류: 셀 {row['셀 ID']}")
    print(f"입력: {INPUT}\n분석 격자: {len(source)}개")
    if args.dry_run:
        for station in STATIONS:
            print(f"  {station}: {sum(r['nearest_station'] == station for r in source)}개")
        print("API 호출·파일 저장 없이 종료합니다.")
        return

    api_key = os.getenv("ORS_API_KEY", "").strip() or getpass.getpass("ORS API 키: ").strip()
    if not api_key:
        raise SystemExit("ORS API 키가 없어 조회하지 않았습니다.")
    try:
        socket.getaddrinfo("api.heigit.org", 443)
    except socket.gaierror:
        raise SystemExit("ORS 서버 DNS 연결이 되지 않습니다. 인터넷 연결 환경에서 다시 실행하세요.")

    previous = {row["셀 ID"]: row for row in read_csv(OUTPUT)} if OUTPUT.exists() else {}
    results = dict(previous)
    for number, row in enumerate(source, 1):
        cell_id = row["셀 ID"]
        old = previous.get(cell_id)
        if old and (old["조회상태"] == "성공" or not args.retry_failed):
            continue
        station = row["nearest_station"]
        end_lat, end_lon = STATIONS[station]
        start_lat, start_lon = float(row["중심점 위도"]), float(row["중심점 경도"])
        minutes, km, status = get_route(api_key, start_lat, start_lon, end_lat, end_lon)
        results[cell_id] = {
            "셀 ID": cell_id, "nearest_station": station,
            "중심점 위도": start_lat, "중심점 경도": start_lon,
            "역 위도": end_lat, "역 경도": end_lon,
            "drive_min": round(minutes, 4) if status == "성공" else "",
            "drive_km": round(km, 4) if status == "성공" else "",
            "조회상태": status,
        }
        save_csv([results[r["셀 ID"]] for r in source if r["셀 ID"] in results])
        # 중단되어도 성공한 요청을 다시 보내지 않도록 체크포인트 저장
        print(f"[{number:3d}/126] 셀 {cell_id} → {station}: {status}", flush=True)
        if number < len(source):
            time.sleep(1)

    ordered_results = [results[r["셀 ID"]] for r in source]
    save_csv(ordered_results)
    successes = sum(row["조회상태"] == "성공" for row in ordered_results)
    print(f"완료: 성공 {successes}/126개, 결과 {OUTPUT}")
    if successes < 126:
        print("실패 격자는 빈 값으로 두었습니다. 원인 확인 후 --retry-failed로 재조회하세요.")
    print("ORS 차량시간은 도로망 기반 추정치이며 실제 혼잡시간을 반영한 관측 통행시간은 아닙니다.")


if __name__ == "__main__":
    main()
