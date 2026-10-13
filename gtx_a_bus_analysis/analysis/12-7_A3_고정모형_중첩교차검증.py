"""A3 변수 구성을 고정한 중첩 Leave-One-Station-Out 교차검증."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "12-2_Ridge_Logistic_예측성능_개선실험.py"
FEATURES = ["pre_share_pct", "log1p_board_2024", "walk_min", "competing_share"]


def load_base_module():
    spec = importlib.util.spec_from_file_location("ridge_experiment_12_2", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    base = load_base_module()
    data = base.load_data().dropna(subset=["nearest_station", "share_increase"]).copy()
    y = data["share_increase"].astype(int)
    groups = data["nearest_station"].astype(str)

    oof_probability = pd.Series(index=data.index, dtype=float)
    oof_prediction = pd.Series(index=data.index, dtype="Int64")
    fold_rows: list[dict] = []

    outer = LeaveOneGroupOut()
    for train_index, test_index in outer.split(data[FEATURES], y, groups):
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        train_groups = groups.iloc[train_index]
        test_station = groups.iloc[test_index].iloc[0]

        # 외부 검증역은 C 선택 과정에서 완전히 제외한다.
        search = GridSearchCV(
            estimator=base.make_pipeline(FEATURES, "linear"),
            param_grid={"model__C": base.C_VALUES},
            scoring="roc_auc",
            cv=LeaveOneGroupOut().split(train[FEATURES], y_train, train_groups),
            refit=True,
            n_jobs=1,
        )
        search.fit(train[FEATURES], y_train)
        probability = search.predict_proba(test[FEATURES])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        oof_probability.iloc[test_index] = probability
        oof_prediction.iloc[test_index] = prediction

        tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
        fold_rows.append(
            {
                "외부 검증역": test_station,
                "학습 표본": len(train_index),
                "검증 표본": len(test_index),
                "내부 선택 C": search.best_params_["model__C"],
                "내부 CV ROC-AUC": search.best_score_,
                "Accuracy": accuracy_score(y_test, prediction),
                "Balanced Accuracy": balanced_accuracy_score(y_test, prediction),
                "ROC-AUC": roc_auc_score(y_test, probability),
                "Brier Score": brier_score_loss(y_test, probability),
                "Log Loss": log_loss(y_test, probability, labels=[0, 1]),
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
            }
        )

    prediction = oof_prediction.astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp)
    overall = pd.DataFrame(
        [
            {
                "표본 수": len(data),
                "Accuracy": accuracy_score(y, prediction),
                "Balanced Accuracy": balanced_accuracy_score(y, prediction),
                "ROC-AUC": roc_auc_score(y, oof_probability),
                "Brier Score": brier_score_loss(y, oof_probability),
                "Log Loss": log_loss(y, oof_probability, labels=[0, 1]),
                "Sensitivity": recall_score(y, prediction),
                "Specificity": specificity,
                "Precision": precision_score(y, prediction),
                "F1": f1_score(y, prediction),
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
            }
        ]
    )

    print("===== A3 고정모형: 외부 역 권역별 중첩 LOSO =====")
    print(pd.DataFrame(fold_rows).round(4).to_string(index=False))
    print("\n===== 전체 OOF 성능 =====")
    print(overall.round(4).to_string(index=False))
    print("\nA3 변수 구성은 고정했으며, 각 외부 검증역을 숨긴 상태에서 C만 내부 선택했습니다.")
    print("결과 파일은 저장하지 않습니다.")


if __name__ == "__main__":
    main()
