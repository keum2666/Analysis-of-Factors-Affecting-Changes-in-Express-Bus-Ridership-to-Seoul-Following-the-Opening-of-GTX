"""A1~A4 변수 선택과 Ridge C 선택을 포함한 중첩 Leave-One-Station-Out.

외부 검증역은 후보모형과 C 선택에 전혀 사용하지 않는다.
결과는 화면에만 출력하고 파일로 저장하지 않는다.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, brier_score_loss
)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut

HERE = Path(__file__).resolve().parent
BASE_SCRIPT = next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob("12-2_Ridge_Logistic_예측성능_개선실험.py"))
spec = importlib.util.spec_from_file_location("ridge_experiment_12_2", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

CANDIDATES = [
    {"name": "A1", "description": "수요비중+보행시간+경쟁목적지",
     "features": ["pre_share_pct", "walk_min", "competing_share"]},
    {"name": "A2", "description": "로그승차+보행시간+경쟁목적지",
     "features": ["log1p_board_2024", "walk_min", "competing_share"]},
    {"name": "A3", "description": "수요비중+로그승차+보행시간+경쟁목적지",
     "features": ["pre_share_pct", "log1p_board_2024", "walk_min", "competing_share"]},
    {"name": "A4", "description": "보행시간+경쟁목적지",
     "features": ["walk_min", "competing_share"]},
]


def nested_model_selection(df):
    y = df["share_increase"].astype(int)
    groups = df["nearest_station"]
    outer = LeaveOneGroupOut()
    oof_probability = pd.Series(index=df.index, dtype=float)
    oof_prediction = pd.Series(index=df.index, dtype=int)
    selection_rows = []
    inner_rows = []

    for train_idx, test_idx in outer.split(df, y, groups):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        train_groups = groups.iloc[train_idx]
        test_station = groups.iloc[test_idx].iloc[0]
        candidate_results = []

        # 외부 학습자료의 두 역만 이용하여 후보모형과 C를 선택한다.
        for candidate in CANDIDATES:
            features = candidate["features"]
            search = GridSearchCV(
                base.make_pipeline(features, "linear"),
                {"model__C": base.C_VALUES},
                scoring="roc_auc",
                cv=LeaveOneGroupOut().split(train[features], y_train, train_groups),
                refit=True,
                n_jobs=1,
            )
            search.fit(train[features], y_train)
            row = {
                "외부 검증역": test_station,
                "후보모형": candidate["name"],
                "후보 설명": candidate["description"],
                "내부 CV ROC-AUC": search.best_score_,
                "내부 선택 C": search.best_params_["model__C"],
                "search": search,
                "candidate": candidate,
            }
            candidate_results.append(row)
            inner_rows.append({k: v for k, v in row.items() if k not in ["search", "candidate"]})

        # 내부 CV AUC 최대 후보만 외부 검증역에 적용한다.
        selected = max(candidate_results, key=lambda x: x["내부 CV ROC-AUC"])
        features = selected["candidate"]["features"]
        fitted = selected["search"].best_estimator_
        probability = fitted.predict_proba(test[features])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        oof_probability.iloc[test_idx] = probability
        oof_prediction.iloc[test_idx] = prediction

        selection_rows.append({
            "외부 검증역": test_station,
            "선택 모형": selected["후보모형"],
            "선택 모형 설명": selected["후보 설명"],
            "내부 CV ROC-AUC": selected["내부 CV ROC-AUC"],
            "선택 C": selected["내부 선택 C"],
            "학습 표본 수": len(train_idx),
            "검증 표본 수": len(test_idx),
            "Accuracy": accuracy_score(y_test, prediction),
            "Balanced Accuracy": balanced_accuracy_score(y_test, prediction),
            "ROC-AUC": roc_auc_score(y_test, probability),
            "Brier Score": brier_score_loss(y_test, probability),
        })

    overall = pd.DataFrame([{
        "표본 수": len(df),
        "Accuracy": accuracy_score(y, oof_prediction),
        "Balanced Accuracy": balanced_accuracy_score(y, oof_prediction),
        "ROC-AUC": roc_auc_score(y, oof_probability),
        "Brier Score": brier_score_loss(y, oof_probability),
    }])
    return pd.DataFrame(inner_rows), pd.DataFrame(selection_rows), overall


def main():
    df = base.load_data()
    inner, selections, overall = nested_model_selection(df)

    print("===== 외부 fold별 내부 후보 비교 =====")
    print(inner.round(4).to_string(index=False))
    print("\n===== 외부 fold별 선택 모형과 실제 검증성능 =====")
    print(selections.round(4).to_string(index=False))
    print("\n===== 변수 선택 과정을 포함한 전체 OOF 성능 =====")
    print(overall.round(4).to_string(index=False))

    print("\n===== 해석 =====")
    counts = selections["선택 모형"].value_counts().to_dict()
    print("외부 fold별 선택 횟수:", counts)
    print(
        "이 성능은 각 외부 검증역을 숨긴 채 A1~A4와 C를 학습 권역 내부에서 "
        "선택한 결과다. 고정된 A3의 탐색적 ROC-AUC 0.6548과 달리, "
        "후보 선택 절차 자체의 일반화 성능을 평가한다. 다만 독립된 역이 3개뿐이므로 "
        "추정 불확실성은 여전히 크며, 추가 역 또는 별도 자료 검증이 필요하다."
    )
    print("\n결과 파일은 저장하지 않습니다. 실행 화면에서만 확인합니다.")


if __name__ == "__main__":
    main()
