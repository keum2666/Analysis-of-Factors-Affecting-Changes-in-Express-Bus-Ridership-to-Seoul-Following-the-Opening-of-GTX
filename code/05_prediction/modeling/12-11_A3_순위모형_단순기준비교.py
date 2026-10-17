"""A3 순위모형과 개통 전 승차량만 사용한 순위를 동일한 LOSO에서 비교한다.

승차량 기준의 방향(+/-)도 외부 검증역을 제외한 학습역만으로 정한다.
점수와 비교표는 화면에만 출력하며 파일로 저장하지 않는다.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent


def load_rank_module():
    path = next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob("12-8_A3_쌍대비교_순위모형.py"))
    spec = importlib.util.spec_from_file_location("pairwise_rank_12_8", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_baseline_direction(train: pd.DataFrame) -> int:
    """학습역의 승차량 단독 AUC 평균이 0.5 이상이 되도록 방향을 고른다."""
    aucs = [
        roc_auc_score(part["share_increase"], part["log1p_board_2024"])
        for _, part in train.groupby("nearest_station", sort=False)
    ]
    return 1 if np.mean(aucs) >= 0.5 else -1


def metrics(rank, station: pd.DataFrame, score: np.ndarray) -> dict:
    top20 = next(
        row for row in rank.precision_lift_rows(station, score)
        if row["상위 구간"] == "상위 20%"
    )
    return {
        "ROC-AUC": roc_auc_score(station["share_increase"], score),
        "상위 20% Precision": top20["Precision@K"],
        "상위 20% Lift": top20["Lift"],
    }


def main() -> None:
    rank = load_rank_module()
    data = rank.load_base_module().load_data().dropna(
        subset=["nearest_station", "share_increase", *rank.FEATURES]
    ).copy()
    data["share_increase"] = data["share_increase"].astype(int)
    rows = []

    for held_out in data["nearest_station"].astype(str).unique():
        train = data[data["nearest_station"].astype(str) != held_out]
        test = data[data["nearest_station"].astype(str) == held_out]

        # A3 규제강도는 외부 검증역을 제외한 두 역 내부의 LOSO만으로 선택한다.
        c_value, _ = rank.choose_c_inner_loso(train)
        pair_x, pair_y = rank.make_pairs(train)
        model = rank.make_model(c_value).fit(pair_x, pair_y)
        a3_score = rank.cell_scores(model, test)

        direction = choose_baseline_direction(train)
        baseline_score = direction * test["log1p_board_2024"].to_numpy(float)

        for name, score in [
            ("개통 전 승차량 단독", baseline_score),
            ("A3 쌍대 순위모형", a3_score),
        ]:
            rows.append({
                "검증역": held_out,
                "모형": name,
                "격자 수": len(test),
                "승차량 기준 방향": "큰 순" if direction == 1 else "작은 순",
                **metrics(rank, test, score),
            })

    result = pd.DataFrame(rows)
    print("===== 같은 외부 LOSO에서 단순 기준과 A3 비교 =====")
    print(result.round(4).to_string(index=False))
    print("\n역별 ROC-AUC 단순평균:")
    print(result.groupby("모형")["ROC-AUC"].mean().round(4).to_string())
    print("\n승차량 기준 방향은 각 검증역을 숨긴 학습역에서만 정했습니다.")
    print("A3 자체는 이전 탐색으로 선택된 모형이므로 이 표를 독립적 최종 성능으로 해석하지 않습니다.")
    print("결과 파일은 저장하지 않습니다.")


if __name__ == "__main__":
    main()
