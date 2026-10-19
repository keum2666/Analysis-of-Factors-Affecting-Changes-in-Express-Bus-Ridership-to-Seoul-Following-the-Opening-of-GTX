"""⑩-14 126격자 20~50대 인구 2025 (챕터 2 주모형: 종속 = 2025 교통카드라 시점을 맞춤).

원자료: 국토정보플랫폼(SGIS) 국토통계 인구정보 250M 격자, 고양시·파주시, 20대·30대·40대·50대 인구 수(전체)
  data/raw/sgis_2025/{고양인구,파주인구}/*_202510/vl_blk.shp (EPSG:5179)
방법 (2024 값과 같은 방법): 격자 중심점이 들어가는 SGIS 250m 격자의 값, 비공개(N/A)는 0, 고양·파주 경계 격자는 두 시 값을 더함
점검: 같은 방법으로 2024(202410)를 다시 만들어 기존 S1 pop_20_50과 126격자 모두 일치하는지 확인

실행 (k 폴더에서): python 건수기준_재분석/code/s10_14_pop2025.py
"""

# UROP 통합: 실행 위치에 독립적인 경로 설정 (계산 방식은 원본 유지)
import sys as _fb_sys
from pathlib import Path as _FBPath
_fb_dir = _FBPath(__file__).resolve().parent
if str(_fb_dir) not in _fb_sys.path:
    _fb_sys.path.insert(0, str(_fb_dir))
from feeder_paths import transfer_path as _fb_path, station_path as _fb_station_path, korean_font as _fb_font

import os
import re
import unicodedata

import geopandas as gpd
import pandas as pd

TAB = _fb_path('건수기준_재분석/10_가설검증/tables')
info = pd.read_csv(_fb_path('건수기준_재분석/tables/S1_126격자_건수기준_기초표.csv'), encoding="utf-8-sig").set_index("셀 ID")
pts = gpd.GeoDataFrame(index=info.index, geometry=gpd.points_from_xy(info["중심점 경도"], info["중심점 위도"]), crs=4326).to_crs(5179)


def build(root, ym):
    out = pd.DataFrame(index=info.index)
    for r, _, ff in os.walk(root):
        r_ = unicodedata.normalize("NFC", r)  # 폴더 이름이 자모 분리(NFD)로 저장돼 있음
        m = re.search(r"([2-5]0)대", r_)
        if "vl_blk.shp" not in ff or ym not in r_ or not m:
            continue
        g = gpd.read_file(os.path.join(r, "vl_blk.shp"))
        j = gpd.sjoin(pts, g, how="left", predicate="within")
        v = pd.to_numeric(j[~j.index.duplicated()]["val"], errors="coerce").fillna(0)  # N/A(비공개) = 0
        col = f"{m.group(1)}대"
        out[col] = out[col] + v if col in out else v  # 경계 격자: 고양·파주 값 합
    return out[["20대", "30대", "40대", "50대"]]


p24, p25 = build(_fb_path('data/raw/sgis_2024'), "202410"), build(_fb_path('data/raw/sgis_2025'), "202510")
check = (p24.sum(axis=1) == info.pop_20_50).mean()
assert check == 1, f"2024 재구축이 S1과 다름 ({check:.3f})"
out = p25.add_prefix("2025 ")
out["20~50대 인구 2025(명)"] = p25.sum(axis=1)
out["20~50대 인구 2024(명)"] = info.pop_20_50
out.insert(0, "권역", info.nearest_station)
out.reset_index().to_csv(f"{TAB}/H14_격자별_20~50대인구_2025.csv", index=False, encoding="utf-8-sig")
print("2024 재구축 일치율:", check)
print(out.groupby("권역")[["20~50대 인구 2024(명)", "20~50대 인구 2025(명)"]].sum())
