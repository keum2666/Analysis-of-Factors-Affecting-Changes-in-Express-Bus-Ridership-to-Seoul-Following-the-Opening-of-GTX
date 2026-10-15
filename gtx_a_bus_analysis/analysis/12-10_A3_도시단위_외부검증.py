"""파주-고양 도시 단위 외부검증과 규제강도 민감도 분석.

외부 도시의 결과로 C를 선택하지 않는다. C=1을 사전 고정 주 결과로 보고하고,
다른 C에서 결론이 유지되는지만 민감도 분석으로 확인한다.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
RANK_SCRIPT = HERE / "12-8_A3_쌍대비교_순위모형.py"
FIXED_C = 1.0
SENSITIVITY_C = [0.1, 0.3162, 1.0, 3.1623, 10.0, 31.6228]


def load_rank_module():
    spec = importlib.util.spec_from_file_location("pairwise_rank_12_8", RANK_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(rank, test: pd.DataFrame, scores: np.ndarray) -> dict:
    evaluated = test.copy()
    evaluated["ranking_score"] = scores
    station_aucs, pair_losses, pair_count = [], [], 0
    for _, station in evaluated.groupby("nearest_station", sort=False):
        y = station["share_increase"].to_numpy(int)
        station_scores = station["ranking_score"].to_numpy(float)
        station_aucs.append(roc_auc_score(y, station_scores))
        loss, count = rank.pairwise_log_loss(station, station_scores)
        pair_losses.append((loss, count))
        pair_count += count
    weighted_loss = sum(loss * count for loss, count in pair_losses) / pair_count
    return {
        "역별 AUC 단순평균": float(np.mean(station_aucs)),
        "최저 역별 AUC": float(np.min(station_aucs)),
        "최고 역별 AUC": float(np.max(station_aucs)),
        "Pairwise Log Loss": float(weighted_loss),
        "외부 비교쌍 수": pair_count,
    }


def main() -> None:
    rank = load_rank_module()
    base = rank.load_base_module()
    data = base.load_data().dropna(
        subset=["nearest_station", "share_increase", *rank.FEATURES]
    ).copy()
    data["share_increase"] = data["share_increase"].astype(int)
    data["도시"] = np.where(data["nearest_station"].astype(str) == "운정중앙", "파주", "고양")

    main_rows, sensitivity_rows, top_rows = [], [], []
    for test_city in ["파주", "고양"]:
        train = data[data["도시"] != test_city]
        test = data[data["도시"] == test_city]
        pair_x, pair_y = rank.make_pairs(train)

        for c_value in SENSITIVITY_C:
            model = rank.make_model(c_value).fit(pair_x, pair_y)
            scores = rank.cell_scores(model, test)
            metrics = evaluate(rank, test, scores)
            sensitivity_rows.append(
                {
                    "학습 도시": train["도시"].iloc[0],
                    "검증 도시": test_city,
                    "C": c_value,
                    **metrics,
                }
            )
            if c_value == FIXED_C:
                main_rows.append(
                    {
                        "학습 도시": train["도시"].iloc[0],
                        "학습 역": "+".join(train["nearest_station"].astype(str).unique()),
                        "검증 도시": test_city,
                        "검증 역": "+".join(test["nearest_station"].astype(str).unique()),
                        "학습 격자": len(train),
                        "검증 격자": len(test),
                        "고정 C": c_value,
                        **metrics,
                    }
                )
                scored = test.copy()
                scored["ranking_score"] = scores
                for _, station in scored.groupby("nearest_station", sort=False):
                    top_rows.extend(
                        rank.precision_lift_rows(station, station["ranking_score"].to_numpy())
                    )

    print("===== 도시 단위 외부검증: 사전 고정 C=1 =====")
    print(pd.DataFrame(main_rows).round(4).to_string(index=False))
    print("\n===== 검증역별 Precision@K와 Lift =====")
    print(pd.DataFrame(top_rows).round(4).to_string(index=False))
    print("\n===== C 민감도 분석(외부 결과로 C를 선택하지 않음) =====")
    print(pd.DataFrame(sensitivity_rows).round(4).to_string(index=False))
    print("\n해석 원칙:")
    print("- C=1의 도시 단위 결과를 주 결과로 사용합니다.")
    print("- 민감도 표는 C에 따라 결론이 뒤집히는지 확인하는 용도입니다.")
    print("- 파주 학습자료에는 역이 하나뿐이므로 내부 역 LOSO로 C를 선택할 수 없습니다.")
    print("- 결과 파일은 저장하지 않습니다.")


if __name__ == "__main__":
    main()
