"""GTX-A 수요 비중 확대 격자의 권역 내 순위를 직접 학습하는 모형.

같은 역 권역의 확대 격자(1)와 비확대 격자(0)를 한 쌍으로 만들고,
확대 격자의 점수가 더 높아지도록 L2 규제 쌍대 로지스틱 모형을 학습한다.
검증역은 학습과 C 선택에서 완전히 제외하는 중첩 Leave-One-Station-Out 방식이다.
결과는 화면에만 출력하며 별도 결과 파일은 저장하지 않는다.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob("12-2_Ridge_Logistic_예측성능_개선실험.py"))
FEATURES = ["pre_share_pct", "log1p_board_2024", "walk_min", "competing_share"]
C_VALUES = np.logspace(-3, 3, 13)
TOP_RATES = (0.10, 0.20, 0.30)


def load_base_module():
    spec = importlib.util.spec_from_file_location("ridge_experiment_12_2", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_pairs(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """권역별 모든 양성-음성 쌍과 역방향 쌍을 생성한다."""
    pair_x: list[np.ndarray] = []
    pair_y: list[int] = []
    for _, station in frame.groupby("nearest_station", sort=False):
        positive = station.loc[station["share_increase"] == 1, FEATURES].to_numpy(float)
        negative = station.loc[station["share_increase"] == 0, FEATURES].to_numpy(float)
        for pos in positive:
            differences = pos - negative
            pair_x.extend(differences)
            pair_y.extend([1] * len(differences))
            pair_x.extend(-differences)
            pair_y.extend([0] * len(differences))
    if not pair_x:
        raise ValueError("확대-비확대 쌍을 만들 수 없습니다.")
    return np.asarray(pair_x), np.asarray(pair_y)


def make_model(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "ranker",
                LogisticRegression(
                    penalty="l2",
                    C=c_value,
                    solver="liblinear",
                    fit_intercept=False,
                    max_iter=10_000,
                    random_state=42,
                ),
            ),
        ]
    )


def cell_scores(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """분류확률이 아니라 권역 내 정렬에 사용하는 연속 점수를 반환한다."""
    return model.decision_function(frame[FEATURES].to_numpy(float))


def mean_station_auc(model: Pipeline, frame: pd.DataFrame) -> float:
    aucs = []
    for _, station in frame.groupby("nearest_station", sort=False):
        y = station["share_increase"].to_numpy(int)
        if np.unique(y).size == 2:
            aucs.append(roc_auc_score(y, cell_scores(model, station)))
    return float(np.mean(aucs))


def pairwise_log_loss(frame: pd.DataFrame, scores: np.ndarray) -> tuple[float, int]:
    """실제 확대 격자가 비확대 격자보다 위에 올 쌍대 로지스틱 손실."""
    evaluated = frame[["share_increase"]].copy()
    evaluated["ranking_score"] = scores
    positive_scores = evaluated.loc[
        evaluated["share_increase"] == 1, "ranking_score"
    ].to_numpy(float)
    negative_scores = evaluated.loc[
        evaluated["share_increase"] == 0, "ranking_score"
    ].to_numpy(float)
    margins = (positive_scores[:, None] - negative_scores[None, :]).ravel()
    # log(1 + exp(-margin))의 수치적으로 안정적인 계산
    losses = np.logaddexp(0.0, -margins)
    return float(losses.mean()), int(losses.size)


def choose_c_inner_loso(train: pd.DataFrame) -> tuple[float, float]:
    """외부 학습권역 안에서 한 역씩 제외하며 평균 역별 AUC로 C를 선택한다."""
    stations = train["nearest_station"].astype(str).unique()
    rows = []
    for c_value in C_VALUES:
        fold_aucs = []
        for validation_station in stations:
            inner_train = train[train["nearest_station"].astype(str) != validation_station]
            inner_valid = train[train["nearest_station"].astype(str) == validation_station]
            pair_x, pair_y = make_pairs(inner_train)
            model = make_model(c_value).fit(pair_x, pair_y)
            fold_aucs.append(mean_station_auc(model, inner_valid))
        rows.append((c_value, float(np.mean(fold_aucs))))
    # 동률이면 규제가 더 강한 작은 C를 선택한다.
    best_c, best_auc = max(rows, key=lambda item: (item[1], -item[0]))
    return float(best_c), float(best_auc)


def precision_lift_rows(station: pd.DataFrame, scores: np.ndarray) -> list[dict]:
    evaluated = station[["nearest_station", "share_increase"]].copy()
    evaluated["ranking_score"] = scores
    evaluated = evaluated.sort_values("ranking_score", ascending=False)
    baseline = evaluated["share_increase"].mean()
    result = []
    for rate in TOP_RATES:
        k = max(1, int(np.ceil(len(evaluated) * rate)))
        precision = evaluated.head(k)["share_increase"].mean()
        result.append(
            {
                "검증역": str(evaluated["nearest_station"].iloc[0]),
                "상위 구간": f"상위 {int(rate * 100)}%",
                "K": k,
                "권역 실제 확대 비율": baseline,
                "Precision@K": precision,
                "Lift": precision / baseline,
            }
        )
    return result


def main() -> None:
    base = load_base_module()
    data = base.load_data().dropna(subset=["nearest_station", "share_increase", *FEATURES]).copy()
    data["share_increase"] = data["share_increase"].astype(int)

    oof_score = pd.Series(index=data.index, dtype=float)
    fold_rows: list[dict] = []
    top_rows: list[dict] = []
    all_pair_losses: list[np.ndarray] = []

    for test_station in data["nearest_station"].astype(str).unique():
        train = data[data["nearest_station"].astype(str) != test_station]
        test = data[data["nearest_station"].astype(str) == test_station]

        best_c, inner_auc = choose_c_inner_loso(train)
        pair_x, pair_y = make_pairs(train)
        model = make_model(best_c).fit(pair_x, pair_y)
        train_scores = cell_scores(model, train)
        train_pair_loss_parts = []
        train_pair_count = 0
        for _, train_station in train.groupby("nearest_station", sort=False):
            part_loss, part_count = pairwise_log_loss(
                train_station, train_scores[train.index.get_indexer(train_station.index)]
            )
            train_pair_loss_parts.append((part_loss, part_count))
            train_pair_count += part_count
        train_pair_loss = sum(
            loss * count for loss, count in train_pair_loss_parts
        ) / train_pair_count
        scores = cell_scores(model, test)
        oof_score.loc[test.index] = scores

        station_auc = roc_auc_score(test["share_increase"], scores)
        station_pair_loss, validation_pairs = pairwise_log_loss(test, scores)

        positive_scores = scores[test["share_increase"].to_numpy(int) == 1]
        negative_scores = scores[test["share_increase"].to_numpy(int) == 0]
        margins = (positive_scores[:, None] - negative_scores[None, :]).ravel()
        all_pair_losses.append(np.logaddexp(0.0, -margins))
        fold_rows.append(
            {
                "외부 검증역": test_station,
                "학습 격자 수": len(train),
                "학습 쌍 수(양방향)": len(pair_y),
                "검증 격자 수": len(test),
                "내부 선택 C": best_c,
                "내부 LOSO 평균 AUC": inner_auc,
                "외부 ROC-AUC": station_auc,
                "외부 비교쌍 수": validation_pairs,
                "학습 Pairwise Log Loss": train_pair_loss,
                "Pairwise Log Loss": station_pair_loss,
                "Loss 차이(외부-학습)": station_pair_loss - train_pair_loss,
            }
        )
        top_rows.extend(precision_lift_rows(test, scores))

    # 권역마다 점수 척도가 다를 수 있으므로 전체 순위를 섞지 않고 역별 AUC를 가중 평균한다.
    fold_table = pd.DataFrame(fold_rows)
    weighted_auc = np.average(fold_table["외부 ROC-AUC"], weights=fold_table["검증 격자 수"])
    macro_auc = fold_table["외부 ROC-AUC"].mean()
    overall_pair_loss = float(np.concatenate(all_pair_losses).mean())
    macro_pair_loss = fold_table["Pairwise Log Loss"].mean()

    print("===== 쌍대 비교 Ridge 순위모형: 외부 역 권역별 중첩 LOSO =====")
    print(fold_table.round(4).to_string(index=False))
    print("\n===== 순위 성능 요약 =====")
    print(f"역별 ROC-AUC 단순평균: {macro_auc:.4f}")
    print(f"역별 ROC-AUC 격자수 가중평균: {weighted_auc:.4f}")
    print(f"역별 Pairwise Log Loss 단순평균: {macro_pair_loss:.4f}")
    print(f"전체 외부 비교쌍 Pairwise Log Loss: {overall_pair_loss:.4f}")
    print("무작위 순위 기준 Pairwise Log Loss: 0.6931")
    print("\n===== 역별 Precision@K와 Lift =====")
    print(pd.DataFrame(top_rows).round(4).to_string(index=False))
    print("\n해석: ROC-AUC와 Lift가 클수록, Pairwise Log Loss는 작을수록 좋습니다.")
    print("이 점수는 확대 확률이 아니며 0.5 분류 기준도 사용하지 않습니다.")
    print("결과 파일은 저장하지 않습니다.")


if __name__ == "__main__":
    main()
