"""⑩-4 연계버스(광역 24개 외 시내·마을버스) 교통카드 합성데이터 수집.

API: 국토교통부_지역별 교통카드이용 합성데이터 (data.go.kr 15154079)
  http://apis.data.go.kr/1613000/RegionalTransportationCardUsageSyntheticData/<operation>
  필수: serviceKey, pageNo, numOfRows, dataType, opr_ymd, rte_id, users_type_cd, ride_ctpv_cd
기존 광역버스 수집과 맞춤: users_type_cd=01, ride_ctpv_cd=41, 날짜 20241017·20251106
인증키: 환경변수 DATA_GO_KR_KEY 또는 k 폴더의 .env (DATA_GO_KR_KEY=...)  — 코드·결과에 키를 남기지 않음
하루 호출 한도(개발계정 1,000건)를 고려해 이미 받은 노선·날짜는 건너뛴다(이어받기).

실행 (k 폴더에서)
  시험: python 건수기준_재분석/code/s10_4_collect_feeder.py test <rte_id>
        → 두 오퍼레이션(경기/서울)으로 1페이지씩 호출해 건수와 하차 시도코드를 보여준다
  수집: python 건수기준_재분석/code/s10_4_collect_feeder.py run [operation] [최대 호출 수]
        노선 목록 = 10_가설검증/tables/H3a_GTX역연계_후보노선_TCN2024.csv 의 광역 외 노선
                  + 10_가설검증/tables/H3b_신설연계노선.csv (있으면, 열 '노선 ID')
"""

# UROP 통합: 실행 위치에 독립적인 경로 설정 (계산 방식은 원본 유지)
import sys as _fb_sys
from pathlib import Path as _FBPath
_fb_dir = _FBPath(__file__).resolve().parent
if str(_fb_dir) not in _fb_sys.path:
    _fb_sys.path.insert(0, str(_fb_dir))
from feeder_paths import transfer_path as _fb_path, station_path as _fb_station_path, korean_font as _fb_font

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "http://apis.data.go.kr/1613000/RegionalTransportationCardUsageSyntheticData/"
OPS = {"경기": "getGyeonggiTransportationCardUsageSyntheticData",
       "서울": "getSeoulTransportationCardUsageSyntheticData"}
DATES = ["20241017", "20251106"]
ROWS = 1000
RAW = Path(_fb_path('건수기준_재분석/10_가설검증/raw'))
TAB = Path(_fb_path('건수기준_재분석/10_가설검증/tables'))
RAW.mkdir(parents=True, exist_ok=True)


def key():
    k = os.environ.get("DATA_GO_KR_KEY")
    if not k and Path(_fb_path('.env')).exists():
        for line in Path(_fb_path('.env')).read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line.startswith("DATA_GO_KR_KEY="):
                k = line.split("=", 1)[1].strip().strip('"')
    if not k:
        sys.exit("인증키 없음: 환경변수 DATA_GO_KR_KEY 또는 .env 필요")
    return k


def call(op, rte_id, ymd, page, tries=4):
    """게이트웨이가 가끔 HTTP_ERROR·429를 돌려주므로 몇 번 쉬었다가 다시 시도한다."""
    for k in range(tries):
        try:
            return _call(op, rte_id, ymd, page)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(60 * (k + 1))


def _call(op, rte_id, ymd, page):
    q = {"pageNo": page, "numOfRows": ROWS, "dataType": "JSON", "opr_ymd": ymd, "rte_id": rte_id,
         "users_type_cd": "01", "ride_ctpv_cd": "41"}
    url = BASE + OPS[op] + "?serviceKey=" + key() + "&" + urllib.parse.urlencode(q)
    with urllib.request.urlopen(url, timeout=60) as r:
        txt = r.read().decode("utf-8")
    try:
        j = json.loads(txt)
    except json.JSONDecodeError:
        raise RuntimeError("JSON 아님(인증·한도 오류 가능): " + txt[:200])
    if "Error" in j:  # 예: {"Error":{"code":"50","message":"NO_DATA_FOUND"}} → 0건
        if j["Error"].get("message") == "NO_DATA_FOUND":
            return {"resultCode": "NO_DATA", "resultMsg": "NO_DATA_FOUND"}, 0, []
        raise RuntimeError("API 오류: " + str(j["Error"])[:200])
    root = j.get("Response") or j.get("response") or {}
    body, head = root.get("body", {}), root.get("header", {})
    if str(head.get("resultCode")) != "200":  # 한도 초과 등 다른 응답을 0건으로 오인하지 않게
        raise RuntimeError("비정상 응답: " + txt[:200])
    items = body.get("items") or {}
    items = items.get("item", []) if isinstance(items, dict) else items
    if isinstance(items, dict):
        items = [items]
    return head, int(body.get("totalCount") or 0), items


def test(rte_id):
    for op in OPS:
        for ymd in DATES:
            try:
                head, total, items = call(op, rte_id, ymd, 1)
                goff = pd.Series([str(i.get("goff_ctpv_cd") or i.get("GOFF_CTPV_CD")) for i in items]).value_counts()
                print(op, ymd, head.get("resultCode"), head.get("resultMsg"), "총", total, "하차시도", goff.to_dict())
            except Exception as e:  # 시험 단계에서는 오류 내용을 그대로 보여준다
                print(op, ymd, "오류:", e)


def routes():
    r = pd.read_csv(TAB / "H3a_GTX역연계_후보노선_TCN2024.csv", dtype={"노선 ID": str})
    ids = list(r.loc[~r["광역 24개 여부"], "노선 ID"])
    extra = TAB / "H3b_신설연계노선.csv"
    if extra.exists():
        ids += [i for i in pd.read_csv(extra, dtype={"노선 ID": str})["노선 ID"].dropna() if i not in ids]
    return ids


def run(op="경기", max_calls=900):
    log, n_calls = [], 0
    for rid in routes():
        for ymd in DATES:
            out = RAW / f"{op}_{ymd}_{rid}.csv"
            if out.exists():
                continue
            rows, page, total = [], 1, None
            try:
                while True:
                    if n_calls >= max_calls:
                        raise StopIteration
                    head, total, items = call(op, rid, ymd, page)
                    n_calls += 1
                    time.sleep(3)
                    rows += items
                    if page * ROWS >= total or not items:
                        break
                    page += 1
                    time.sleep(3)
            except StopIteration:
                print("호출 한도 도달:", n_calls)
                pd.DataFrame(log).to_csv(RAW / "_수집기록.csv", mode="a", index=False, encoding="utf-8-sig",
                                         header=not (RAW / "_수집기록.csv").exists())
                return
            except Exception as e:
                log.append({"노선 ID": rid, "날짜": ymd, "상태": "오류", "건수": 0, "내용": str(e)[:200]})
                continue
            pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
            log.append({"노선 ID": rid, "날짜": ymd, "상태": "성공", "건수": len(rows), "총건수": total})
    pd.DataFrame(log).to_csv(RAW / "_수집기록.csv", mode="a", index=False, encoding="utf-8-sig",
                             header=not (RAW / "_수집기록.csv").exists())
    print("완료, 호출 수:", n_calls)


if __name__ == "__main__":
    if sys.argv[1] == "test":
        test(sys.argv[2])
    else:
        run(sys.argv[2] if len(sys.argv) > 2 else "경기", int(sys.argv[3]) if len(sys.argv) > 3 else 900)
