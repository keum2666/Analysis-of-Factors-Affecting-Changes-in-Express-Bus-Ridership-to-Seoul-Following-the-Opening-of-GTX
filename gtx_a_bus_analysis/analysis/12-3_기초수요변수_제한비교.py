"""A1~A4 Ridge Logistic 탐색적 비교.

결과는 화면에만 출력한다. 외부 검증은 Leave-One-Station-Out이며,
전처리와 C 선택은 각 외부 fold의 학습 권역 내부에서만 수행한다.
"""
from pathlib import Path
import importlib.util
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "12-2_Ridge_Logistic_예측성능_개선실험.py"
spec = importlib.util.spec_from_file_location("ridge_experiment_12_2", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

MODEL_SPECS = [
    {
        "experiment": "제한비교", "name": "A1 수요비중+보행시간+경쟁목적지",
        "features": ["pre_share_pct", "walk_min", "competing_share"],
        "walk_mode": "linear",
    },
    {
        "experiment": "제한비교", "name": "A2 로그승차+보행시간+경쟁목적지",
        "features": ["log1p_board_2024", "walk_min", "competing_share"],
        "walk_mode": "linear",
    },
    {
        "experiment": "제한비교", "name": "A3 수요비중+로그승차+보행시간+경쟁목적지",
        "features": ["pre_share_pct", "log1p_board_2024", "walk_min", "competing_share"],
        "walk_mode": "linear",
    },
    {
        "experiment": "제한비교", "name": "A4 보행시간+경쟁목적지",
        "features": ["walk_min", "competing_share"],
        "walk_mode": "linear",
    },
]


def main():
    df = base.load_data()
    overall_rows = []
    fold_tables = []

    print("목표변수 분포")
    print(pd.crosstab(df["nearest_station"], df["share_increase"], margins=True))

    for i, model_spec in enumerate(MODEL_SPECS, start=1):
        print(f"\n[{i}/{len(MODEL_SPECS)}] {model_spec['name']}")
        overall, folds, _ = base.evaluate_spec(df, model_spec)
        overall_rows.append(overall)
        fold_tables.append(folds)

    overall_df = pd.DataFrame(overall_rows)
    folds_df = pd.concat(fold_tables, ignore_index=True)
    columns = [
        "모형", "사용 독립변수", "표본 수", "Accuracy", "Balanced Accuracy",
        "ROC-AUC", "Brier Score", "대곡 ROC-AUC", "운정중앙 ROC-AUC",
        "킨텍스 ROC-AUC", "역별 AUC 범위"
    ]
    print("\n===== A1~A4 전체 비교 =====")
    print(overall_df[columns].round(4).to_string(index=False))

    print("\n===== 외부 검증 fold 상세 =====")
    detail_columns = [
        "모형", "검증역", "학습 표본 수", "검증 표본 수", "선택 C",
        "Balanced Accuracy", "ROC-AUC", "Brier Score"
    ]
    print(folds_df[detail_columns].round(4).to_string(index=False))

    print("\n===== 해석상 주의 =====")
    print(
        "이 비교는 기존 외부검증 결과를 확인한 뒤 구성한 탐색적 실험이다. "
        "따라서 같은 Leave-One-Station-Out 결과에서 가장 높은 모형을 선택하고 "
        "그 점수를 최종 일반화 성능으로 확정할 수 없다. 변수 선택까지 포함한 "
        "중첩 검증이나 추가 GTX역·별도 날짜 등의 독립 검증자료가 필요하다."
    )
    print("\n결과 파일은 저장하지 않습니다. 실행 화면에서만 확인합니다.")


if __name__ == "__main__":
    main()
