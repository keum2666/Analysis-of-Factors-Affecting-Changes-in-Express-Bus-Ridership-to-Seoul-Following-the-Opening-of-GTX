"""ORS 차량시간과 현재 OLS 설명변수의 상관·VIF를 화면에 출력한다.

차량시간 126개 조회가 모두 성공한 뒤 실행한다. 파일은 생성하지 않는다.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


PROCESSED = next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / "data" / "processed" / "gtx_a"


def read_one(pattern):
    paths = list(PROCESSED.glob(pattern))
    if len(paths) != 1:
        raise FileNotFoundError(f"{pattern}: 파일 {len(paths)}개 발견")
    return pd.read_csv(paths[0], encoding="utf-8-sig")


def main():
    drive = read_one("26_ORS_차량접근시간_250m격자.csv")
    if len(drive) != 126 or not drive["조회상태"].eq("성공").all():
        raise ValueError("ORS 차량시간 126개가 모두 성공한 뒤 진단하세요.")

    base = read_one("18_새데이터_회귀분석_격자통합데이터.csv")
    extra = read_one("18-1_*.csv")
    land = read_one("19_*.csv")
    floor = read_one("24_*.csv")
    id_col = "셀 ID"
    residential = "주거지역 비율(%)"
    floor_col = "2024년 건축물 연면적(천㎡)"
    for frame in (base, extra, land, floor, drive):
        frame[id_col] = pd.to_numeric(frame[id_col], errors="raise")

    data = (
        base.merge(extra[[id_col, "bus_time_mean"]], on=id_col, validate="one_to_one")
        .merge(land[[id_col, residential]], on=id_col, validate="one_to_one")
        .merge(floor[[id_col, floor_col]], on=id_col, validate="one_to_one")
        .merge(drive[[id_col, "drive_min"]], on=id_col, validate="one_to_one")
    )
    if len(data) != 126:
        raise ValueError(f"결합 후 격자가 {len(data)}개입니다. 셀 ID를 확인하세요.")

    for column in ("walk_min", "drive_min", "pop_20_50", floor_col,
                   residential, "competing_share", "bus_time_mean"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[["walk_min", "drive_min", "pop_20_50", floor_col,
             residential, "competing_share", "bus_time_mean"]].isna().any().any():
        raise ValueError("공선성 진단 변수에 결측값이 있습니다.")

    data["log_walk"] = np.log1p(data["walk_min"])
    data["log_floor"] = np.log1p(data[floor_col])
    if not np.isfinite(data[["log_walk", "log_floor"]].to_numpy()).all():
        raise ValueError("로그변환 변수에 유한하지 않은 값이 있습니다.")

    original = ["log_walk", "pop_20_50", "log_floor", residential,
                "competing_share", "bus_time_mean"]
    with_drive = original + ["drive_min"]
    print("격자 수:", len(data))
    print("보행시간·차량시간 상관계수:",
          round(data["walk_min"].corr(data["drive_min"]), 4))
    print("로그 보행시간·차량시간 상관계수:",
          round(data["log_walk"].corr(data["drive_min"]), 4))
    print("\n차량시간과 각 설명변수의 피어슨 상관계수")
    print(data[with_drive].corr()["drive_min"].round(4).to_string())

    for name, features in (("현재 모형(IC 제외)", original),
                           ("차량시간 추가 모형", with_drive)):
        x = sm.add_constant(data[features], has_constant="add")
        vif = pd.DataFrame({
            "변수": features,
            "VIF": [variance_inflation_factor(x.to_numpy(), i + 1)
                    for i in range(len(features))],
        })
        print(f"\n{name} VIF")
        print(vif.round(3).to_string(index=False))
    print("\nVIF만으로 변수 채택 여부를 결정하지 말고, 계수 안정성과 해석 가능성도 확인하세요.")


if __name__ == "__main__":
    main()
