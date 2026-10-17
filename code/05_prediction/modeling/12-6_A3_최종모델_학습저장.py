"""GTX-A 126개 격자로 A3 Ridge Logistic 최종 적용모형을 학습·저장한다.

모형 구조(A3)는 중첩 LOSO 검증 결과에 따라 사전에 고정한다. 규제강도 C만
전체 GTX-A 자료 내부의 Leave-One-Station-Out 교차검증으로 선택하고,
선택된 C로 126개 전체 표본을 재학습한다.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut


HERE = Path(__file__).resolve().parent
PROJECT = next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir())
BASE_SCRIPT = next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob("12-2_Ridge_Logistic_예측성능_개선실험.py"))
MODEL_DIR = PROJECT / "results" / "prediction" / "models"
MODEL_PATH = MODEL_DIR / "A3_ridge_logistic_final.joblib"
METADATA_PATH = MODEL_DIR / "A3_ridge_logistic_final_metadata.json"
CV_PATH = MODEL_DIR / "A3_ridge_logistic_C_selection.csv"

FEATURES = ["pre_share_pct", "log1p_board_2024", "walk_min", "competing_share"]
FEATURE_LABELS = {
    "pre_share_pct": "개통 전 역 권역 내 수요 비중(%)",
    "log1p_board_2024": "log(1+개통 전 승차인원)",
    "walk_min": "GTX역 보행시간(분)",
    "competing_share": "GTX 경쟁 목적지 비율(%)",
}


def load_base_module():
    spec = importlib.util.spec_from_file_location("ridge_experiment_12_2", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    base = load_base_module()
    data = base.load_data().dropna(subset=["nearest_station", "share_increase"]).copy()
    X = data[FEATURES]
    y = data["share_increase"].astype(int)
    groups = data["nearest_station"].astype(str)

    if len(data) != 126:
        raise ValueError(f"예상 표본 126개와 다릅니다: {len(data)}개")
    if groups.nunique() != 3:
        raise ValueError(f"예상 역 권역 3개와 다릅니다: {groups.nunique()}개")

    search = GridSearchCV(
        estimator=base.make_pipeline(FEATURES, "linear"),
        param_grid={"model__C": base.C_VALUES},
        scoring="roc_auc",
        cv=LeaveOneGroupOut().split(X, y, groups),
        refit=True,
        n_jobs=1,
        return_train_score=False,
    )
    search.fit(X, y)
    final_pipeline = search.best_estimator_
    selected_c = float(search.best_params_["model__C"])

    cv = pd.DataFrame(
        {
            "C": search.cv_results_["param_model__C"].astype(float),
            "평균_LOSO_ROC_AUC": search.cv_results_["mean_test_score"],
            "표준편차_LOSO_ROC_AUC": search.cv_results_["std_test_score"],
            "순위": search.cv_results_["rank_test_score"],
        }
    ).sort_values(["순위", "C"]).reset_index(drop=True)

    bundle = {
        "model_name": "A3 Ridge Logistic Regression",
        "pipeline": final_pipeline,
        "features": FEATURES,
        "feature_labels": FEATURE_LABELS,
        "selected_C": selected_c,
        "classification_threshold": 0.5,
        "target_definition": "역 권역 내 개통 후 수요 비중 > 개통 전 수요 비중이면 1, 아니면 0",
        "training_scope": "GTX-A 운정중앙·킨텍스·대곡 250m 격자 126개",
        "intended_use": "GTX-B·C 격자의 상대적 수요 비중 확대 예측점수 및 권역 내 순위 산출",
        "gtx_bc_feature_mapping": {
            "pre_share_pct": "GTX-B·C 적용 기준일의 역 권역 내 개통 전 수요 비중(%)",
            "log1p_board_2024": "GTX-B·C 적용 기준일의 개통 전 승차인원에 log(1+x) 적용",
            "walk_min": "GTX-B·C 예정역까지의 보행시간(분)",
            "competing_share": "GTX-B·C 공통 정의로 산출한 경쟁 목적지 비율(%)",
        },
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    cv.to_csv(CV_PATH, index=False, encoding="utf-8-sig")

    imputer = final_pipeline.named_steps["imputer"]
    scaler = final_pipeline.named_steps["scaler"]
    classifier = final_pipeline.named_steps["model"]
    metadata = {
        key: value for key, value in bundle.items() if key != "pipeline"
    }
    metadata.update(
        {
            "sample_count": int(len(data)),
            "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
            "station_counts": {str(k): int(v) for k, v in groups.value_counts().items()},
            "best_internal_loso_roc_auc": float(search.best_score_),
            "feature_ranges": {
                feature: {
                    "min": float(X[feature].min()),
                    "max": float(X[feature].max()),
                    "median": float(X[feature].median()),
                }
                for feature in FEATURES
            },
            "imputer_statistics": dict(zip(FEATURES, map(float, imputer.statistics_))),
            "scaler_mean": dict(zip(FEATURES, map(float, scaler.mean_))),
            "scaler_scale": dict(zip(FEATURES, map(float, scaler.scale_))),
            "standardized_coefficients": dict(zip(FEATURES, map(float, classifier.coef_[0]))),
            "intercept": float(classifier.intercept_[0]),
            "software": {
                "python": platform.python_version(),
                "numpy": importlib.metadata.version("numpy"),
                "pandas": importlib.metadata.version("pandas"),
                "scikit_learn": importlib.metadata.version("scikit-learn"),
                "joblib": importlib.metadata.version("joblib"),
            },
            "model_sha256": sha256(MODEL_PATH),
            "c_selection_file": CV_PATH.name,
            "methodological_note": (
                "A3 변수 구성은 중첩 LOSO 검증 후 고정했다. 저장 모형의 C는 126개 전체 자료의 "
                "내부 LOSO로 선택했으며, best_internal_loso_roc_auc는 독립 외부 성능이 아니다."
            ),
        }
    )
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    loaded = joblib.load(MODEL_PATH)
    probability = loaded["pipeline"].predict_proba(X)[:, 1]
    if not np.isfinite(probability).all():
        raise RuntimeError("저장 후 재로딩한 모형의 예측값에 비정상 값이 있습니다.")

    print("===== A3 최종 적용모형 저장 완료 =====")
    print(f"학습 표본: {len(data)}개 / 역 권역: {groups.nunique()}개")
    print(f"종속변수 분포: {y.value_counts().sort_index().to_dict()}")
    print(f"선택 C: {selected_c:g}")
    print(f"내부 LOSO 평균 ROC-AUC: {search.best_score_:.4f}")
    print(f"모형: {MODEL_PATH}")
    print(f"메타데이터: {METADATA_PATH}")
    print(f"C 선택표: {CV_PATH}")
    print("주의: 내부 LOSO 점수는 독립된 최종 일반화 성능이 아닙니다.")


if __name__ == "__main__":
    main()
