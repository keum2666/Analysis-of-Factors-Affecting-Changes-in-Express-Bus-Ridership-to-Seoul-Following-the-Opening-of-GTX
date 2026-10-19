"""⑩-10 출근 시간 맞춤 비교: 광역버스가 줄어든 시간대와 GTX역행 마을·시내버스가 늘어난 시간대가 같은가.

격자 126개, 건수 기준, 하루치 비교 (2024-10-17 / 2025-11-06). 출근 = 06:00~09:59 (챕터 1·2와 같음)
광역: 24개 노선 교통카드 (승차정류장 → 격자, 연도별 매핑)
GTX역행 마을·시내버스: 격자 정류장 승차 → 역명 정류장(운정중앙역·킨텍스역·대곡역) 하차 (s10_8과 같은 정의)
비교 기준: 같은 노선들의 격자 정류장 승차 전체 (마을·시내버스 전체)
 T1 권역 × 시간대(1시간) 승차: 광역 2024·2025·변화, GTX역행 2024·2025·변화
 T2 출근 비율: 광역 감소분 중 출근, GTX역행(2025) 중 출근, 마을·시내버스 전체(2025) 중 출근
 T3 격자 단위 상관 (출근 / 그 외): Δ광역 vs Δ GTX역행  — 개통 후 변화끼리라 연관만 본다

실행 (k 폴더에서): python 건수기준_재분석/code/s10_10_peak_matching.py <STTN_20241017.parquet>
"""

# UROP 통합: 실행 위치에 독립적인 경로 설정 (계산 방식은 원본 유지)
import sys as _fb_sys
from pathlib import Path as _FBPath
_fb_dir = _FBPath(__file__).resolve().parent
if str(_fb_dir) not in _fb_sys.path:
    _fb_sys.path.insert(0, str(_fb_dir))
from feeder_paths import transfer_path as _fb_path, station_path as _fb_station_path, korean_font as _fb_font

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

plt.rcParams["font.family"] = _fb_font()
plt.rcParams["axes.unicode_minus"] = False

STTN = _fb_station_path(sys.argv)
RAW = Path(_fb_path('건수기준_재분석/10_가설검증/raw'))
STOPS = Path(_fb_path('건수기준_재분석/10_가설검증/stops'))
TAB = Path(_fb_path('건수기준_재분석/10_가설검증/tables'))
FIG = Path(_fb_path('건수기준_재분석/10_가설검증/figures'))
FIG.mkdir(exist_ok=True)
CARD = Path(_fb_path('gtx_a_seoul_bus_outputs/transport_card'))
PAT = "운정중앙역|킨텍스역|대곡역"
STATIONS = ["운정중앙", "킨텍스", "대곡"]


def norm(s):
    return s.astype(str).str.replace(".0", "", regex=False).str.lstrip("0")


a = pq.read_table(STTN, columns=["정류장 ID", "정류장 명칭"]).to_pandas()
b = pd.read_csv(STOPS / "BusStop_20251106_파주고양서울일부.csv", dtype=str)
c = pd.read_csv(STOPS / "2025_신설정류장_좌표_GBIS.csv", dtype=str)
name = pd.concat([pd.Series(c["정류장명"].values, index=norm(c["정류장 ID"])),
                  pd.Series(b.sttn_nm.values, index=norm(b.sttn_id)),
                  pd.Series(a["정류장 명칭"].values, index=norm(a["정류장 ID"]))])
name = name[~name.index.duplicated()]
station_stops = set(name[name.str.contains(PAT, na=False)].index)

info = pd.read_csv(_fb_path('건수기준_재분석/tables/S1_126격자_건수기준_기초표.csv'), encoding="utf-8-sig").set_index("셀 ID")
cells = pd.read_csv(next(Path(_fb_path('data/processed')).glob("16_*")), dtype=str)
s2c = {y: {sid.lstrip("0"): int(cid) for cid, s in zip(cells["셀 ID"], cells[f"{y}년 정류장 ID"])
           for sid in re.findall(r"\d{6,}", s)} for y in ("2024", "2025")}


def prep(df, on_col):
    df["year"] = df.opr_ymd.str[:4]
    df["on"] = norm(df[on_col])
    df["cell"] = [s2c[y].get(o) for y, o in zip(df.year, df.on)]
    df = df[df.cell.notna()].copy()
    df["cell"] = df.cell.astype(int)
    df["hour"] = df.ride_dt.str[8:10].astype(int)
    df["출근"] = df.hour.between(6, 9)
    df["권역"] = df.cell.map(info.nearest_station)
    return df


M = prep(pd.concat([pd.read_csv(CARD / f, dtype=str, usecols=["opr_ymd", "승차정류장ID", "ride_dt"]) for f in
                    ["gtx_a_transport_card_20241017_raw_with_coords_api_filled.csv",
                     "gtx_a_transport_card_20251106_raw_with_coords.csv"]], ignore_index=True), "승차정류장ID")
chk = M.groupby(["year", "cell"]).size().unstack(0).reindex(info.index).fillna(0)
match = {"2024": bool((chk["2024"] == info.board_2024).all()), "2025": bool((chk["2025"] == info.board_2025).all())}

F = prep(pd.concat([pd.read_csv(f, dtype=str, usecols=["opr_ymd", "ride_sttn_id", "goff_sttn_id", "ride_dt"])
                    for f in sorted(RAW.glob("경기_*.csv")) if f.stat().st_size > 10], ignore_index=True), "ride_sttn_id")
F["off"] = norm(F.goff_sttn_id)
F["GTX역행"] = F.off.isin(station_stops) & ~F.on.isin(station_stops)
G = F[F.GTX역행]

# T1 권역 × 시간대
def hourly(df, tag):
    t = df.groupby(["권역", "hour", "year"]).size().unstack("year").fillna(0)
    t.columns = [f"{tag} {y}" for y in t.columns]
    return t


T1 = pd.concat([hourly(M, "광역"), hourly(G, "GTX역행")], axis=1).fillna(0)
for col in ("광역", "GTX역행"):
    T1[f"{col} 변화"] = T1[f"{col} 2025"] - T1[f"{col} 2024"]
T1 = T1.astype(int).reset_index()
T1.to_csv(TAB / "H10a_권역x시간대_광역vsGTX역행.csv", index=False, encoding="utf-8-sig")

# T2 출근 비율
rows = []
for st in STATIONS + ["전체"]:
    m = M if st == "전체" else M[M.권역 == st]
    g = G if st == "전체" else G[G.권역 == st]
    f = F if st == "전체" else F[F.권역 == st]
    dm = lambda x: (x.year == "2025").sum() - (x.year == "2024").sum()
    rows.append({"권역": st,
                 "광역 변화(건)": dm(m), "광역 출근 변화(건)": dm(m[m.출근]), "광역 그 외 변화(건)": dm(m[~m.출근]),
                 "광역 출근 비율 2024(%)": 100 * m[m.year == "2024"].출근.mean(),
                 "GTX역행 2025(건)": (g.year == "2025").sum(), "GTX역행 변화(건)": dm(g),
                 "GTX역행 출근 변화(건)": dm(g[g.출근]),
                 "GTX역행 2025 출근 비율(%)": 100 * g[g.year == "2025"].출근.mean(),
                 "마을·시내버스 전체 2025 출근 비율(%)": 100 * f[f.year == "2025"].출근.mean()})
T2 = pd.DataFrame(rows)
T2.round(1).to_csv(TAB / "H10b_출근비율_광역vsGTX역행.csv", index=False, encoding="utf-8-sig")

# T3 격자 단위 상관 (출근 / 그 외)
def grid_delta(df, mask):
    t = df[mask].groupby(["cell", "year"]).size().unstack("year").reindex(info.index).fillna(0)
    return t.get("2025", 0) - t.get("2024", 0)


rows = []
for lab, mm, gm in [("출근(06~09시)", M.출근, G.출근), ("그 외 시간", ~M.출근, ~G.출근)]:
    dM, dG = grid_delta(M, mm), grid_delta(G, gm)
    for st in STATIONS + ["전체"]:
        idx = info.index if st == "전체" else info.index[info.nearest_station == st]
        r = stats.spearmanr(dM[idx], dG[idx])
        rows.append({"시간대": lab, "권역": st, "격자 수": len(idx), "Δ광역 합": int(dM[idx].sum()),
                     "Δ GTX역행 합": int(dG[idx].sum()), "순위상관": r[0], "p값": r[1]})
T3 = pd.DataFrame(rows)
T3.round(4).to_csv(TAB / "H10c_격자상관_출근vs그외.csv", index=False, encoding="utf-8-sig")

# 그림: 권역별 시간대 승차 건수 — 광역버스 2024 · 광역버스 2025 · GTX역행 마을·시내버스 2025 (모두 승차 건수)
#   챕터 2 주모형 종속 = 2025 GTX역행 승차. 전체 시간대(0~23시) / 출근 전후(5~10시), 회색 = 출근(06~09시)
def hour_fig(hours, out, width, share=True):
    fig, axes = plt.subplots(1, 3, figsize=(width, 4.5), sharey=share)  # 기본: 세 권역 y축 통일 (크기 비교용)
    for ax, st in zip(axes, STATIONS):
        t = T1[T1.권역 == st].set_index("hour").reindex(hours).fillna(0)
        for k, (col, lab, colr) in enumerate([("광역 2024", "광역버스 2024", "#f4a6a6"), ("광역 2025", "광역버스 2025", "tab:red"),
                                              ("GTX역행 2025", "GTX역행 마을·시내버스 2025", "tab:blue")]):
            ax.bar(t.index + (k - 1) * 0.27, t[col], 0.27, label=lab, color=colr)
        ax.axvspan(5.5, 9.5, color="grey", alpha=0.12, label="출근 시간대 (06~09시, 분석 기준)")
        ax.set_xticks(list(hours), [f"{h}시" for h in hours], fontsize=8 if len(hours) > 12 else 10)
        ax.set_xlim(hours[0] - 0.6, hours[-1] + 0.6)
        ax.set_title(f"{st}: 시간대별 승차 건수")
        ax.set_xlabel("승차 시각")
        ax.tick_params(labelleft=True)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("승차 건수")
    fig.tight_layout()
    fig.savefig(FIG / out, dpi=150)
    return axes[0].get_legend_handles_labels()  # 범례는 그래프에 넣지 않고 따로 저장 (PPT에 따로 붙임)


def legend_fig(handles, labels, out):
    fig = plt.figure(figsize=(9, 0.6))
    fig.legend(handles, labels, loc="center", ncol=len(labels), frameon=False, fontsize=11)
    fig.savefig(FIG / out, dpi=150, bbox_inches="tight", transparent=True)


hour_fig(range(0, 24), "H10_시간대별_광역vsGTX역행.png", 20)
legend_fig(*hour_fig(range(5, 11), "H10_출근시간대(5~10시)_광역vsGTX역행.png", 15), "H10_범례_광역vsGTX역행.png")
hour_fig(range(5, 11), "H10_출근시간대(5~10시)_광역vsGTX역행_권역별축.png", 15, share=False)  # 부록용: 대곡 등 작은 권역 모양 확인

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print("광역 격자 합계가 챕터 1 승차와 일치:", match)
print(T2.round(1).T.to_string())
print(T3.round(3).to_string(index=False))
for st in STATIONS:
    t = T1[T1.권역 == st].set_index("hour")[["광역 변화", "GTX역행 2025", "GTX역행 변화"]]
    print(st); print(t.T.to_string())
