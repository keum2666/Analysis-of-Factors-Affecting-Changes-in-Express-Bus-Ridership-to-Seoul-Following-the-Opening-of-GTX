"""전체 GTX-A 자료로 쌍대 비교 Ridge 순위모형을 학습해 저장한다."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd


HERE = Path(__file__).resolve().parent
RANK_SCRIPT = HERE / "12-8_A3_쌍대비교_순위모형.py"
MODEL_DIR = HERE.parent / "models"


def load_rank_module():
    spec = importlib.util.spec_from_file_location("pairwise_rank_12_8", RANK_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    rank = load_rank_module()
    base = rank.load_base_module()
    data = base.load_data().dropna(
        subset=["nearest_station", "share_increase", *rank.FEATURES]
    ).copy()
    data["share_increase"] = data["share_increase"].astype(int)

    # 세 역 중 한 역씩 숨긴 검증만 이용해 최종 C를 고른다.
    selected_c, loso_auc = rank.choose_c_inner_loso(data)
    pair_x, pair_y = rank.make_pairs(data)
    model = rank.make_model(selected_c).fit(pair_x, pair_y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "A3_pairwise_ridge_ranker_final.joblib"
    metadata_path = MODEL_DIR / "A3_pairwise_ridge_ranker_final_metadata.json"

    bundle = {
        "model": model,
        "features": rank.FEATURES,
        "selected_c": selected_c,
        "score_meaning": "권역 내 수요 비중 확대 상대순위 점수(확률 아님)",
        "ranking_method": "각 GTX역 권역 안에서 score 내림차순",
        "training_target": "같은 권역의 확대 격자가 비확대 격자보다 높은 점수",
    }
    joblib.dump(bundle, model_path)

    metadata = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "training_cells": int(len(data)),
        "training_pairs_bidirectional": int(len(pair_y)),
        "stations": sorted(data["nearest_station"].astype(str).unique().tolist()),
        "features": rank.FEATURES,
        "selected_c": selected_c,
        "full_data_loso_mean_auc_for_c_selection": loso_auc,
        "warning": (
            "C 선택값은 전체 GTX-A 자료의 LOSO 결과이다. 독립 일반화 성능은 "
            "12-8 외부 중첩 LOSO 결과로 보고하며, score를 실제 확률로 해석하지 않는다."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    coefficient = pd.Series(
        model.named_steps["ranker"].coef_[0], index=rank.FEATURES, name="표준화 계수"
    ).sort_values(key=abs, ascending=False)

    print("===== 최종 쌍대 비교 Ridge 순위모형 =====")
    print(f"학습 격자 수: {len(data)}")
    print(f"학습 쌍 수(양방향): {len(pair_y)}")
    print(f"선택 C: {selected_c:.4f}")
    print(f"C 선택용 전체자료 LOSO 평균 AUC: {loso_auc:.4f}")
    print("\n표준화 계수(방향 참고용):")
    print(coefficient.round(4).to_string())
    print(f"\n모형 저장: {model_path}")
    print(f"설명 저장: {metadata_path}")
    print("GTX-C에서는 각 역 권역별 점수를 내림차순으로 정렬해 순위와 상위 10·20·30%를 산출합니다.")


if __name__ == "__main__":
    main()
