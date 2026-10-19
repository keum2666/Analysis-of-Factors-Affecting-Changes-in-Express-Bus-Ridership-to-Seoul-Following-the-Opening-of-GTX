"""⑩-13 GTX역행 마을·시내버스 EDA + 설명력 보강 변수 시험.

정의 (s10_8·s10_10·s10_12와 같음)
  GTX역행 마을·시내버스 승차 = 126격자 정류장 승차 → 이름에 운정중앙역·킨텍스역·대곡역이 들어간 정류장 하차 (역 정류장 승차 제외)
EDA
  E1 격자 분포 요약 (권역별 0 격자 수, 평균, 최대)          E2 노선별 GTX역행 승차 (2024·2025)
  E3 상위 격자 10개                                         E4 마을버스로 역까지 걸린 시간 vs 걸어서 걸리는 시간
  그림: 2025 GTX역행 분포 / 보행시간·연면적 산점도 / 출근 시간 Δ광역 vs 2025 GTX역행 산점도 / 노선별 막대 (2025, 2024는 괄호)
  (챕터 2 주모형 종속 = 2025 GTX역행 승차 → EDA·변수 시험도 2025 기준)
변수 시험 (모두 개통 전 값, 주모형 4개 + 후보 1개씩 → 좋은 것만 묶어서)
  log(1+차량 접근시간), 보행시간 ÷ 차량시간, 고령인구 비율, log(1+공시지가), 격자 정류장 수, 개통 전 마을·시내버스 노선 수

실행 (k 폴더에서): python 건수기준_재분석/code/s10_13_feeder_eda_vars.py <STTN_20241017.parquet>
"""

# UROP 통합: 실행 위치에 독립적인 경로 설정 (계산 방식은 원본 유지)
import sys as _fb_sys
from pathlib import Path as _FBPath
_fb_dir = _FBPath(__file__).resolve().parent
if str(_fb_dir) not in _fb_sys.path:
    _fb_sys.path.insert(0, str(_fb_dir))
from feeder_paths import transfer_path as _fb_path, station_path as _fb_station_path, korean_font as _fb_font

import glob
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

plt.rcParams["font.family"] = _fb_font()
plt.rcParams["axes.unicode_minus"] = False

STTN = _fb_station_path(sys.argv)
RAW = Path(_fb_path('건수기준_재분석/10_가설검증/raw'))
STOPS = Path(_fb_path('건수기준_재분석/10_가설검증/stops'))
TAB = Path(_fb_path('건수기준_재분석/10_가설검증/tables'))
FIG = Path(_fb_path('건수기준_재분석/10_가설검증/figures'))
PAT = "운정중앙역|킨텍스역|대곡역"
STATIONS = ["운정중앙", "킨텍스", "대곡"]
COLORS = {"운정중앙": "tab:blue", "킨텍스": "tab:orange", "대곡": "tab:green"}


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
cells.index = cells["셀 ID"].astype(int)
s2c = {y: {sid.lstrip("0"): cid for cid, s in zip(cells.index, cells[f"{y}년 정류장 ID"])
           for sid in re.findall(r"\d{6,}", s)} for y in ("2024", "2025")}

F = pd.concat([pd.read_csv(f, dtype=str, usecols=["opr_ymd", "rte_id", "ride_sttn_id", "goff_sttn_id", "ride_dt", "goff_dt"])
               for f in sorted(RAW.glob("경기_*.csv")) if f.stat().st_size > 10], ignore_index=True)
F["year"] = F.opr_ymd.str[:4]
F["on"], F["off"] = norm(F.ride_sttn_id), norm(F.goff_sttn_id)
F["cell"] = [s2c[y].get(o) for y, o in zip(F.year, F.on)]
F = F[F.cell.notna()].copy()
F["cell"] = F.cell.astype(int)
F["권역"] = F.cell.map(info.nearest_station)
F["gtx"] = F.off.isin(station_stops) & ~F.on.isin(station_stops)
F["탑승시간(분)"] = (pd.to_datetime(F.goff_dt, format="%Y%m%d%H%M%S", errors="coerce")
                 - pd.to_datetime(F.ride_dt, format="%Y%m%d%H%M%S", errors="coerce")).dt.total_seconds() / 60
G = F[F.gtx]

d = pd.read_csv(TAB / "H11a_격자별_공급구조_GTX역행.csv", encoding="utf-8-sig").set_index("셀 ID").reindex(info.index)
for y in ("2024", "2025"):
    d[f"GTX역행 {y}"] = G[G.year == y].groupby("cell").size().reindex(d.index).fillna(0)
d["보행시간(분)"] = info.walk_min
d["건축물 연면적(천㎡)"] = info["2024년 건축물 연면적(천㎡)"]

# ---------- EDA ----------
rows = []
for st in STATIONS + ["전체"]:
    x = d if st == "전체" else d[d.권역 == st]
    rows.append({"권역": st, "격자 수": len(x), "GTX역행 2024 합": int(x["GTX역행 2024"].sum()),
                 "GTX역행 2025 합": int(x["GTX역행 2025"].sum()), "Δ 합": int(x["Δ GTX역행"].sum()),
                 "0→0 격자": int(((x["GTX역행 2024"] == 0) & (x["GTX역행 2025"] == 0)).sum()),
                 "증가 격자": int((x["Δ GTX역행"] > 0).sum()), "감소 격자": int((x["Δ GTX역행"] < 0).sum()),
                 "Δ 평균": x["Δ GTX역행"].mean(), "Δ 중앙값": x["Δ GTX역행"].median(), "Δ 최대": x["Δ GTX역행"].max(),
                 "Δ광역 합": int(x["Δ광역"].sum())})
E1 = pd.DataFrame(rows)

# 노선번호: 2024 노선 원자료(ROUTE, STTN과 같은 폴더) → 없으면(2025 신설) H3b
rt = pd.read_parquet(Path(STTN).parent / "ROUTE_20241017.parquet", columns=["노선 ID", "노선 번호"])
rt_no = pd.Series(rt["노선 번호"].astype(str).str.strip().values, index=rt["노선 ID"].astype(str).str.strip())
rt_no = rt_no[~rt_no.index.duplicated()]
new = pd.read_csv(TAB / "H3b_신설연계노선.csv", dtype=str).set_index("노선 ID")
E2 = G.groupby(["rte_id", "year"]).size().unstack("year").fillna(0).astype(int)
ids = E2.index.to_series()
E2["노선번호"] = ids.map(rt_no).fillna(ids.map(new["노선번호"])).fillna("")
E2["신설(2025)"] = E2.index.isin(new.index)
top_reg = lambda y: G[G.year == y].groupby("rte_id").권역.agg(lambda s: s.value_counts().index[0])
E2["주 권역(2025)"] = top_reg("2025").reindex(E2.index).fillna(top_reg("2024"))  # 2025 기록이 없으면 2024 권역

# 기존 노선의 경로 변경 판정 (공급구조 판정과 같은 정류장 대응 규칙: ID 같음 → 이름 같음 → 좌표 100m 이내)
#   2024 노선-정류장 원자료(ROUTESTTN)의 그 노선 정류장 목록에 대응되지 않는 정류장을 2025년에 이용(승·하차)했으면 "경로 변경"
#   한쪽 방향만 본다: 2025에 이용이 없는 정류장은 그날 아무도 안 탔을 수도 있어 "빠졌다"고 단정할 수 없기 때문
rs = pd.read_parquet(Path(STTN).parent / "ROUTESTTN_20241017.parquet",
                     columns=["노선ID", "정류장 ID", "정류장 명칭", "정류장 X 좌표", "정류장 Y 좌표"])
rs["노선ID"] = rs["노선ID"].astype(str).str.strip()
rs = rs[rs["노선ID"].isin(E2.index)].copy()
rs["sid"] = norm(rs["정류장 ID"])
st_xy = pq.read_table(STTN, columns=["정류장 ID", "정류장 X 좌표", "정류장 Y 좌표"]).to_pandas()
st_xy.index = norm(st_xy["정류장 ID"])
c_xy = c.set_index(norm(c["정류장 ID"]))
lat = pd.concat([c_xy["위도"].astype(float), st_xy["정류장 X 좌표"].astype(float)])
lon = pd.concat([c_xy["경도"].astype(float), st_xy["정류장 Y 좌표"].astype(float)])
lat, lon = lat[~lat.index.duplicated()], lon[~lon.index.duplicated()]
F25 = pd.concat([pd.read_csv(f, dtype=str, usecols=["rte_id", "ride_sttn_id", "goff_sttn_id"])
                 for f in sorted(RAW.glob("경기_20251106_*.csv")) if f.stat().st_size > 10], ignore_index=True)
used25 = pd.concat([pd.DataFrame({"rte": F25.rte_id, "sid": norm(F25.ride_sttn_id)}),
                    pd.DataFrame({"rte": F25.rte_id, "sid": norm(F25.goff_sttn_id)})]).value_counts()


def unmatched_stops(rte):
    r = rs[rs["노선ID"] == rte]
    ids, nms = set(r.sid), set(r["정류장 명칭"].astype(str))
    la, lo = np.radians(r["정류장 X 좌표"].astype(float).values), np.radians(r["정류장 Y 좌표"].astype(float).values)
    out = []
    for s, k in used25[rte].items():
        # 9로 시작하는 7자리 ID = 기점·종점 정류장의 카드용 ID (4로 바꾸면 같은 정류장: 예 9122182 → 4122182 야당역)
        if len(s) == 7 and s[0] == "9" and ("4" + s[1:]) in (set(name.index) | set(lat.index)):
            s = "4" + s[1:]
        if s in ids or name.get(s) in nms:
            continue
        if s in lat.index and len(la):
            p, l = np.radians(lat[s]), np.radians(lon[s])
            h = np.sin((la - p) / 2) ** 2 + np.cos(p) * np.cos(la) * np.sin((lo - l) / 2) ** 2
            if (6371008.8 * 2 * np.arcsin(np.sqrt(h)) <= 100).any():
                continue
        out.append((name.get(s, s), int(k)))
    return sorted(out, key=lambda x: -x[1])


chg = {}
for rte in E2.index:
    if E2.at[rte, "신설(2025)"]:
        chg[rte] = ("신설", 0, "")
    elif rte not in set(rs["노선ID"]) or rte not in used25.index.get_level_values(0):
        chg[rte] = ("판정 불가", 0, "")
    else:
        u = unmatched_stops(rte)
        chg[rte] = ("경로 변경" if u else "경로 유지", len(u), ", ".join(f"{n}({k})" for n, k in u[:6]))
E2["노선 구분"] = [chg[r][0] for r in E2.index]
E2["2024 경로에 없는 2025 이용 정류장 수"] = [chg[r][1] for r in E2.index]
E2["그 정류장 (이름(2025 승·하차))"] = [chg[r][2] for r in E2.index]
E2 = E2.sort_values("2025", ascending=False).reset_index().rename(columns={"rte_id": "카드 노선 ID"})

E3 = d.sort_values("GTX역행 2025", ascending=False).head(10)[
    ["권역", "보행시간(분)", "건축물 연면적(천㎡)", "GTX역행 2024", "GTX역행 2025", "Δ GTX역행", "Δ광역", "Δ광역 출근"]].reset_index()

t = G[G.year == "2025"].groupby("cell")["탑승시간(분)"].median()
E4 = pd.DataFrame({"권역": d.권역, "보행시간(분)": d["보행시간(분)"], "마을버스 탑승시간 중앙값(분)": t,
                   "GTX역행 2025": d["GTX역행 2025"]}).dropna()
E4s = pd.DataFrame({st: {"GTX역행 승차가 있는 격자 수": len(x), "보행시간 중앙값(분)": x["보행시간(분)"].median(),
                          "마을버스 탑승시간(분, 승차가중 평균)": np.average(x["마을버스 탑승시간 중앙값(분)"],
                                                                     weights=x["GTX역행 2025"])}
                    for st, x in E4.groupby("권역")}).T.rename_axis("권역")

E1.round(2).to_csv(TAB / "H13a_EDA_권역별_분포.csv", index=False, encoding="utf-8-sig")
E2.to_csv(TAB / "H13b_EDA_노선별_GTX역행.csv", index=False, encoding="utf-8-sig")
E3.round(2).to_csv(TAB / "H13c_EDA_상위격자.csv", index=False, encoding="utf-8-sig")
E4s.round(1).reset_index().to_csv(TAB / "H13d_EDA_탑승시간vs보행시간.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
for st in STATIONS:
    x = d[d.권역 == st]
    ax[0, 0].hist(x["GTX역행 2025"], bins=np.arange(0, 100, 5), alpha=.6, label=st, color=COLORS[st])
    ax[0, 1].scatter(x["보행시간(분)"], x["GTX역행 2025"], s=18, label=st, color=COLORS[st])
    ax[1, 0].scatter(x["건축물 연면적(천㎡)"], x["GTX역행 2025"], s=18, label=st, color=COLORS[st])
    ax[1, 1].scatter(x["Δ광역 출근"], x["GTX역행 2025 출근"], s=18, label=st, color=COLORS[st])
ax[0, 0].set(title="2025 GTX역행 마을·시내버스 승차 분포 (126격자)", xlabel="2025 승차 (건)", ylabel="격자 수")
ax[0, 1].set(title="GTX역 보행시간 vs 2025 GTX역행", xlabel="보행시간(분)", ylabel="2025 GTX역행(건)", xscale="log")
ax[1, 0].set(title="건축물 연면적 vs 2025 GTX역행", xlabel="연면적(천㎡)", ylabel="2025 GTX역행(건)", xscale="symlog")
ax[1, 1].set(title="출근 시간: Δ광역 vs 2025 GTX역행", xlabel="Δ광역 출근(건)", ylabel="2025 GTX역행 출근(건)")
for a_ in ax.flat:
    a_.legend()
fig.tight_layout()
fig.savefig(FIG / "H13_EDA_GTX역행마을버스.png", dpi=150)

# 노선별 그림 3개 (2025 승차 막대, 막대 위 숫자 = 2025 (2024)): ① 경로 유지 ② 경로 변경 ③ 신설 노선끼리 (신설 12개 전부)
for kind, fn in [("경로 유지", "H13_EDA_노선별_GTX역행_①경로유지.png"), ("경로 변경", "H13_EDA_노선별_GTX역행_②경로변경.png")]:
    r_ = E2[(E2["노선 구분"] == kind) & (E2["2025"] > 0)].head(15)
    fig, a1 = plt.subplots(figsize=(max(6, 0.8 * len(r_) + 2), 4.8))
    xx = np.arange(len(r_))
    b2 = a1.bar(xx, r_["2025"], .6, color=[COLORS.get(g, "grey") for g in r_["주 권역(2025)"]])
    a1.bar_label(b2, labels=[f"{v}\n({u})" for v, u in zip(r_["2025"], r_["2024"])], fontsize=8)
    a1.set_xticks(xx, [f"{n or r}\n{g}" for r, n, g in zip(r_["카드 노선 ID"], r_["노선번호"], r_["주 권역(2025)"])], fontsize=9)
    a1.set(title=f"{kind} 노선별 2025 GTX역행 승차 ({len(r_)}개, 괄호 = 2024)", xlabel="노선번호 / 주로 내린 역 권역", ylabel="GTX역행 승차 건수")
    a1.set_ylim(0, r_["2025"].max() * 1.2)
    a1.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / fn, dpi=150)

nr = new[["노선번호", "GTX역"]].copy()
nr["2025"] = E2.set_index("카드 노선 ID")["2025"].reindex(nr.index).fillna(0).astype(int)
nr = nr.sort_values(["GTX역", "2025"], ascending=[False, False])
fig, a2 = plt.subplots(figsize=(11, 4.5))
xx = np.arange(len(nr))
bars = a2.bar(xx, nr["2025"], .6, color=[COLORS.get(s, "grey") for s in nr["GTX역"]])
a2.bar_label(bars, fontsize=9)
a2.set_xticks(xx, [f"{n}\n{s}" for n, s in zip(nr["노선번호"], nr["GTX역"])], fontsize=9)
a2.set(title=f"2025 신설 연계 노선별 GTX역행 승차 (2025, 신설 {len(nr)}개 전부)", xlabel="노선번호 / 연계 GTX역", ylabel="GTX역행 승차 건수")
a2.grid(axis="y", alpha=.3)
fig.tight_layout()
fig.savefig(FIG / "H13_EDA_노선별_GTX역행_③신설.png", dpi=150)

# 연계 노선 전체 그림 (PPT용): 2025 GTX역행 이용 건수, 내린 GTX역별 색 (한 노선이 두 역에 서면 쌓아서 표시), 범례는 따로 저장
G25 = G[G.year == "2025"].copy()
G25["내린 역"] = G25.off.map(name).str.extract(f"({'|'.join(STATIONS)})", expand=False)
RS = G25.groupby(["rte_id", "내린 역"]).size().unstack("내린 역").reindex(columns=STATIONS).fillna(0).astype(int)
RS = RS.loc[RS.sum(axis=1).sort_values(ascending=False).index]
lab = RS.index.to_series().map(E2.set_index("카드 노선 ID")["노선번호"]).replace("", np.nan).fillna(RS.index.to_series())
fig, a3 = plt.subplots(figsize=(13, 4.8))
xx, bot = np.arange(len(RS)), np.zeros(len(RS))
for st in STATIONS:
    a3.bar(xx, RS[st], .65, bottom=bot, color=COLORS[st], label=f"{st}역")
    bot += RS[st].values
for x_, v in zip(xx, bot):
    a3.text(x_, v + 3, f"{int(v)}", ha="center", fontsize=9)
a3.set_xticks(xx, lab, fontsize=9, rotation=45)
a3.set(title=f"GTX역 연계 마을·시내버스 노선별 이용 건수 (2025.11.06, {len(RS)}개 노선)", xlabel="노선번호", ylabel="GTX역행 승차 건수")
a3.set_ylim(0, bot.max() * 1.12)
a3.grid(axis="y", alpha=.3)
fig.tight_layout()
fig.savefig(FIG / "H13_연계노선별_이용건수_2025.png", dpi=150)
h_, l_ = a3.get_legend_handles_labels()
fig = plt.figure(figsize=(6, 0.6))
fig.legend(h_, l_, loc="center", ncol=3, frameon=False, fontsize=11)
fig.savefig(FIG / "H13_범례_GTX역.png", dpi=150, bbox_inches="tight", transparent=True)
RS.assign(합계=RS.sum(axis=1), 노선번호=lab).to_csv(TAB / "H13f_연계노선별_이용건수_2025.csv", encoding="utf-8-sig")

# ---------- 변수 시험 ----------
drive = pd.read_csv(glob.glob(_fb_path('data/processed/26_*'))[0]).set_index("셀 ID").reindex(d.index)
pop = pd.read_csv(glob.glob(_fb_path('data/processed/07_*'))[0]).set_index("셀 ID").reindex(d.index)
land = pd.read_csv(glob.glob(_fb_path('data/processed/12_*'))[0]).set_index("셀 ID").reindex(d.index)
d["20~50대 인구 2025(명)"] = pd.read_csv(TAB / "H14_격자별_20~50대인구_2025.csv", encoding="utf-8-sig").set_index("셀 ID")["20~50대 인구 2025(명)"]
d["1km 안 기존 철도역 수"] = pd.read_csv(_fb_path('건수기준_재분석/05_변수확장/tables/S5_후보변수_126격자.csv'), encoding="utf-8-sig").set_index("셀 ID")["n_rail_1km"]
d["운정 더미"] = (d.권역 == "운정중앙").astype(int)
X = d[["log(1+GTX역 보행시간)", "20~50대 인구 2025(명)", "log(1+건축물 연면적)", "주거지역 비율(%)",
       "1km 안 기존 철도역 수", "운정 더미"]].copy()  # s10_12 주모형과 같음
BASE = list(X.columns)
C = pd.DataFrame(index=d.index)
C["log(1+차량 접근시간)"] = np.log1p(drive.drive_min)
C["보행시간÷차량시간"] = info.walk_min / drive.drive_min
C["고령인구 비율(%)"] = 100 * pop["고령인구"] / (pop["고령인구"] + pop["20~50대 인구"])
C["log(1+공시지가)"] = np.log1p(land["2024년 공시지가(원/㎡)"])
C["격자 정류장 수"] = cells.reindex(d.index)["2024년 정류장 ID"].str.count(r"\d{6,}")
C["개통 전 마을·시내버스 노선 수"] = F[F.year == "2024"].groupby("cell").rte_id.nunique().reindex(d.index).fillna(0)
miss = C.isna().sum()

rows = []
m0 = sm.OLS(d["GTX역행 2025"], sm.add_constant(X)).fit(cov_type="HC3")
rows.append({"모형": "주모형 (6개)", "추가 변수": "", "계수": np.nan, "p값": np.nan, "수정 R²": m0.rsquared_adj, "n": int(m0.nobs)})
for v in C.columns:
    Z = pd.concat([X, C[[v]]], axis=1).dropna()
    m = sm.OLS(d.loc[Z.index, "GTX역행 2025"], sm.add_constant(Z)).fit(cov_type="HC3")
    rows.append({"모형": "주모형 + 1개", "추가 변수": v, "계수": m.params[v], "p값": m.pvalues[v],
                 "수정 R²": m.rsquared_adj, "n": int(m.nobs)})
V = pd.DataFrame(rows)
V.round(4).to_csv(TAB / "H13e_변수시험_하나씩.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(E1.round(1).to_string(index=False))
print(E2.head(20).to_string(index=False))
print(E3.round(1).to_string(index=False))
print(E4s.round(1))
print("결측:", miss.to_dict())
print(C.describe().round(2).T)
print(V.round(3).to_string(index=False))
