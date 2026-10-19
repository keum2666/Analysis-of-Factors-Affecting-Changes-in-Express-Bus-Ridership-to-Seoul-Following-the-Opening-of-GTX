"""⑩-15 순위상관 그림 (PPT용): Δ광역 vs 2025 GTX역행 마을·시내버스 승차 (126격자, 종일).

① 순위 산점도: x = Δ광역 순위 (1 = 가장 많이 줄어듦), y = GTX역행 2025 순위 (126 = 가장 많이 탐), 권역별 색
   같은 값은 평균 순위 (Spearman과 같음). 왼쪽 위로 몰리면 "많이 준 격자에서 많이 탔다" (ρ < 0)
② 광역 감소 순 5묶음별 격자당 평균 GTX역행 2025 승차 (막대)
범례는 그래프에 넣지 않고 따로 저장 (PPT에 따로 붙임)

실행 (k 폴더에서): python 건수기준_재분석/code/s10_15_spearman_fig.py
"""

# UROP 통합: 실행 위치에 독립적인 경로 설정 (계산 방식은 원본 유지)
import sys as _fb_sys
from pathlib import Path as _FBPath
_fb_dir = _FBPath(__file__).resolve().parent
if str(_fb_dir) not in _fb_sys.path:
    _fb_sys.path.insert(0, str(_fb_dir))
from feeder_paths import transfer_path as _fb_path, station_path as _fb_station_path, korean_font as _fb_font

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams["font.family"] = _fb_font()
plt.rcParams["axes.unicode_minus"] = False

TAB = Path(_fb_path('건수기준_재분석/10_가설검증/tables'))
FIG = Path(_fb_path('건수기준_재분석/10_가설검증/figures'))
STATIONS = ["운정중앙", "킨텍스", "대곡"]
COLORS = {"운정중앙": "tab:blue", "킨텍스": "tab:orange", "대곡": "tab:green"}
d = pd.read_csv(TAB / "H11a_격자별_공급구조_GTX역행.csv", encoding="utf-8-sig").set_index("셀 ID")
assert len(d) == 126
a, b = d["Δ광역"], d["GTX역행 2025"]
rho, p = stats.spearmanr(a, b)
d["광역 순위"], d["GTX역행 순위"] = a.rank(), b.rank()  # 광역: 1 = 가장 많이 줄어듦 / GTX역행: 126 = 가장 많이 탐

# ① 순위 산점도
fig, ax = plt.subplots(figsize=(7, 5.5))
for st in STATIONS:
    x = d[d.권역 == st]
    ax.scatter(x["광역 순위"], x["GTX역행 순위"], s=28, alpha=.8, color=COLORS[st], label=st)
ax.set(xlabel="광역버스 감소 순위  (왼쪽 = 많이 줄어듦)", ylabel="GTX역행 마을·시내버스 승차 순위  (위 = 많이 탐)",
       title=f"광역버스 감소 순위 vs GTX역행 승차 순위 (126격자, ρ = {rho:.2f}, p {'< 0.001' if p < .001 else f'= {p:.3f}'})")
ax.set_xlim(0, 127)
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(FIG / "H15_순위상관_산점도.png", dpi=150)
h_, l_ = ax.get_legend_handles_labels()
lf = plt.figure(figsize=(5, 0.5))
lf.legend(h_, l_, loc="center", ncol=3, frameon=False, fontsize=11)
lf.savefig(FIG / "H15_범례_권역.png", dpi=150, bbox_inches="tight", transparent=True)

# ② 광역 감소 순 5묶음
d["묶음"] = pd.qcut(d["광역 순위"], 5, labels=["1\n(가장 많이 줄어듦)", "2", "3", "4", "5\n(가장 적게 줄거나 늘어남)"])
Q = d.groupby("묶음", observed=True).agg(격자수=("GTX역행 2025", "size"), 광역변화합=("Δ광역", "sum"),
                                        GTX역행합=("GTX역행 2025", "sum"), 격자당GTX역행=("GTX역행 2025", "mean"))
fig, ax = plt.subplots(figsize=(8, 4.8))
bars = ax.bar(range(len(Q)), Q["격자당GTX역행"], .6, color="tab:blue")
ax.bar_label(bars, labels=[f"{v:.1f}" for v in Q["격자당GTX역행"]], fontsize=10)
ax.set_xticks(range(len(Q)), Q.index, fontsize=9)
ax.set(xlabel="광역버스 감소 크기 순으로 나눈 5묶음 (같은 값이 있어 묶음당 19~31격자)", ylabel="격자당 GTX역행 승차 (건, 2025)",
       title="광역버스가 많이 줄어든 격자일수록 GTX역행 승차가 많음")
ax.set_ylim(0, Q["격자당GTX역행"].max() * 1.2)
ax.grid(axis="y", alpha=.3)
fig.tight_layout()
fig.savefig(FIG / "H15_순위상관_5묶음.png", dpi=150)
Q.round(2).reset_index().to_csv(TAB / "H15_광역감소5묶음_GTX역행.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 200)
print(f"rho {rho:.3f} p {p:.4f}")
print(Q.round(2))
