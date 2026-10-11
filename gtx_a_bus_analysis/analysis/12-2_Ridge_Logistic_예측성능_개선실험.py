"""GTX-A 수요 비중 확대 Ridge Logistic 사전지정 모형 비교.

외부 검증은 Leave-One-Station-Out, C 선택은 외부 검증역을 제외한
학습자료 내부 Leave-One-Station-Out으로 수행한다.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, brier_score_loss
)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
C_VALUES = np.logspace(-3, 3, 13)
ID = "셀 ID"

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
PROCESSED = PROJECT / "data" / "processed"


def read_csv(name):
    return pd.read_csv(PROCESSED / name, encoding="utf-8-sig", low_memory=False)


def load_data():
    model = read_csv("18_새데이터_회귀분석_격자통합데이터.csv")
    extra = read_csv("18-1_새데이터_추가피처_250m격자.csv")
    land = read_csv("19_용도지역_면적비율_250m격자.csv")
    floor = read_csv("24_건축물연면적_2024_250m격자.csv")
    for frame in (model, extra, land, floor):
        frame[ID] = pd.to_numeric(frame[ID], errors="coerce").astype("Int64")
    df = (
        model
        .merge(extra[[ID, "bus_time_mean"]], on=ID, how="left", validate="one_to_one")
        .merge(land[[ID, "주거지역 비율(%)"]], on=ID, how="left", validate="one_to_one")
        .merge(floor[[ID, "2024년 건축물 연면적(천㎡)"]], on=ID, how="left", validate="one_to_one")
    )
    numeric = [
        "board_2024_all", "board_2025_all", "walk_min", "pop_20_50",
        "competing_share", "bus_time_mean", "ic_access", "주거지역 비율(%)",
        "2024년 건축물 연면적(천㎡)"
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 종속변수의 분모는 각 시점·역 권역의 전체 수요이다.
    df["share_pre"] = df["board_2024_all"] / df.groupby("nearest_station")["board_2024_all"].transform("sum")
    df["share_post"] = df["board_2025_all"] / df.groupby("nearest_station")["board_2025_all"].transform("sum")
    df["share_increase"] = (df["share_post"] > df["share_pre"]).astype(int)

    # 결정론적 변환. 학습된 통계량을 사용하지 않으며 이후 표준화는 fold 내부에서 수행한다.
    df["pre_share_pct"] = df["share_pre"] * 100
    df["log1p_board_2024"] = np.log1p(df["board_2024_all"].clip(lower=0))
    df["log1p_walk_min"] = np.log1p(df["walk_min"].clip(lower=0))
    df["log1p_pop_20_50"] = np.log1p(df["pop_20_50"].clip(lower=0))
    df["log1p_floor_area"] = np.log1p(df["2024년 건축물 연면적(천㎡)"].clip(lower=0))
    df["walk_x_competing"] = df["walk_min"] * df["competing_share"]
    df["bus_time_x_competing"] = df["bus_time_mean"] * df["competing_share"]
    return df


LABELS = {
    "pre_share_pct": "개통 전 역 권역 내 수요 비중(%)",
    "log1p_board_2024": "log(1+개통 전 승차인원)",
    "walk_min": "GTX역 보행시간(분)",
    "log1p_walk_min": "log(1+GTX역 보행시간)",
    "log1p_pop_20_50": "log(1+20~50대 인구)",
    "log1p_floor_area": "log(1+건축물 연면적)",
    "competing_share": "GTX 경쟁 목적지 비율(%)",
    "bus_time_mean": "평균 버스 통행시간(분)",
    "ic_access": "IC 접근성(km)",
    "주거지역 비율(%)": "주거지역 비율(%)",
    "walk_x_competing": "보행시간×경쟁 목적지 비율",
    "bus_time_x_competing": "버스 통행시간×경쟁 목적지 비율",
}

M0 = [
    "pre_share_pct", "walk_min", "competing_share", "bus_time_mean", "ic_access",
    "log1p_pop_20_50", "log1p_floor_area", "주거지역 비율(%)"
]
MODEL_SPECS = [
    {"experiment": "실험1", "name": "M0 현재 전체모형", "features": M0, "walk_mode": "linear"},
    {"experiment": "실험1", "name": "M1 핵심 3변수", "features": ["pre_share_pct", "walk_min", "competing_share"], "walk_mode": "linear"},
    {"experiment": "실험1", "name": "M2 M1+버스시간+IC", "features": ["pre_share_pct", "walk_min", "competing_share", "bus_time_mean", "ic_access"], "walk_mode": "linear"},
    {"experiment": "실험1", "name": "M3 M2+인구+연면적", "features": ["pre_share_pct", "walk_min", "competing_share", "bus_time_mean", "ic_access", "log1p_pop_20_50", "log1p_floor_area"], "walk_mode": "linear"},
    {"experiment": "실험1", "name": "M4 개통 전 비중 제외", "features": [x for x in M0 if x != "pre_share_pct"], "walk_mode": "linear"},
    {"experiment": "실험2", "name": "보행시간 log(1+x)", "features": ["log1p_walk_min" if x == "walk_min" else x for x in M0], "walk_mode": "log"},
    {"experiment": "실험2", "name": "보행시간 Spline", "features": M0, "walk_mode": "spline"},
    {"experiment": "실험3", "name": "보행시간×경쟁목적지", "features": M0 + ["walk_x_competing"], "walk_mode": "linear"},
    {"experiment": "실험3", "name": "버스시간×경쟁목적지", "features": M0 + ["bus_time_x_competing"], "walk_mode": "linear"},
    {"experiment": "실험4", "name": "개통 전 승차인원 log(1+x)", "features": ["log1p_board_2024" if x == "pre_share_pct" else x for x in M0], "walk_mode": "linear"},
]


def make_pipeline(features, walk_mode):
    model = LogisticRegression(
        penalty="l2", solver="liblinear", max_iter=5000, random_state=RANDOM_STATE
    )
    if walk_mode != "spline":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])

    # Spline의 knot는 각 학습 fold에서만 추정된다. 낮은 자유도를 위해 3 knots, 2차식을 사용한다.
    other = [x for x in features if x != "walk_min"]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    spline_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("spline", SplineTransformer(n_knots=3, degree=2, knots="quantile", include_bias=False)),
        ("scaler", StandardScaler()),
    ])
    preprocessing = ColumnTransformer([
        ("walk_spline", spline_pipe, ["walk_min"]),
        ("other", numeric_pipe, other),
    ], remainder="drop")
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def evaluate_spec(df, spec):
    features = spec["features"]
    sample = df.dropna(subset=["nearest_station", "share_increase"]).copy()
    X, y, groups = sample[features], sample["share_increase"].astype(int), sample["nearest_station"]
    probabilities = pd.Series(index=sample.index, dtype=float)
    predictions = pd.Series(index=sample.index, dtype=int)
    fold_rows = []

    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        train_groups = groups.iloc[train_idx]
        station = groups.iloc[test_idx].iloc[0]

        search = GridSearchCV(
            make_pipeline(features, spec["walk_mode"]),
            {"model__C": C_VALUES},
            scoring="roc_auc",
            cv=LeaveOneGroupOut().split(X_train, y_train, train_groups),
            refit=True,
            n_jobs=1,
        )
        search.fit(X_train, y_train)
        prob = search.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        probabilities.iloc[test_idx] = prob
        predictions.iloc[test_idx] = pred
        fold_rows.append({
            "실험명": spec["experiment"], "모형": spec["name"], "검증역": station,
            "학습 표본 수": len(train_idx), "검증 표본 수": len(test_idx),
            "선택 C": search.best_params_["model__C"],
            "Accuracy": accuracy_score(y_test, pred),
            "Balanced Accuracy": balanced_accuracy_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, prob),
            "Brier Score": brier_score_loss(y_test, prob),
        })

    feature_text = " + ".join(LABELS.get(x, x) for x in features)
    overall = {
        "실험명": spec["experiment"], "모형": spec["name"],
        "사용 독립변수": feature_text, "표본 수": len(sample),
        "Accuracy": accuracy_score(y, predictions),
        "Balanced Accuracy": balanced_accuracy_score(y, predictions),
        "ROC-AUC": roc_auc_score(y, probabilities),
        "Brier Score": brier_score_loss(y, probabilities),
    }
    fold_df = pd.DataFrame(fold_rows)
    for station in ["대곡", "운정중앙", "킨텍스"]:
        overall[f"{station} ROC-AUC"] = fold_df.loc[fold_df["검증역"] == station, "ROC-AUC"].iloc[0]
    overall["역별 AUC 최솟값"] = fold_df["ROC-AUC"].min()
    overall["역별 AUC 최댓값"] = fold_df["ROC-AUC"].max()
    overall["역별 AUC 범위"] = fold_df["ROC-AUC"].max() - fold_df["ROC-AUC"].min()

    predictions_df = sample[[ID, "nearest_station", "share_increase"]].copy()
    predictions_df["실험명"] = spec["experiment"]
    predictions_df["모형"] = spec["name"]
    predictions_df["OOF 비중확대확률"] = probabilities
    predictions_df["OOF 예측"] = predictions
    return overall, fold_df, predictions_df


def main():
    df = load_data()
    overall_rows, fold_tables, prediction_tables = [], [], []
    for number, spec in enumerate(MODEL_SPECS, start=1):
        print(f"[{number}/{len(MODEL_SPECS)}] {spec['name']}")
        overall, folds, predictions = evaluate_spec(df, spec)
        overall_rows.append(overall)
        fold_tables.append(folds)
        prediction_tables.append(predictions)

    overall_df = pd.DataFrame(overall_rows)
    folds_df = pd.concat(fold_tables, ignore_index=True)
    predictions_df = pd.concat(prediction_tables, ignore_index=True)
    columns = [
        "실험명", "모형", "사용 독립변수", "표본 수", "Accuracy", "Balanced Accuracy",
        "ROC-AUC", "Brier Score", "대곡 ROC-AUC", "운정중앙 ROC-AUC", "킨텍스 ROC-AUC",
        "역별 AUC 범위"
    ]
    print("\n===== 전체 비교 =====")
    print(overall_df[columns].round(4).to_string(index=False))
    print("\n결과 파일은 저장하지 않습니다. 실행 화면에서만 확인합니다.")


if __name__ == "__main__":
    main()
