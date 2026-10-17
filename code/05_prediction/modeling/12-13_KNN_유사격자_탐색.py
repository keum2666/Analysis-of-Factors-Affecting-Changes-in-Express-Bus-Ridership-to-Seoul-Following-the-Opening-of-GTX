"""GTX-A 세 역 간 KNN 유사격자 탐색. 결과는 화면에만 출력한다.

목표를 확정하지 않은 탐색: 개통 후 승차인원, log(개통 후/전 승차+1),
역 권역 내 비중 변화(%p)를 각각 검토한다. 비중 확대 0/1 모형을 재사용하지 않는다.
외부 Leave-One-Station-Out의 검증역은 표준화와 K/가중치/로그 변환 선택에서 제외한다.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
STATIONS = ('대곡', '운정중앙', '킨텍스')
FEATURE_SETS = {
    '개통전승차만': ['log1p_board_2024'],
    'A3_네변수': ['log1p_board_2024', 'pre_share_pct', 'walk_min', 'competing_share'],
}
K_VALUES = (3, 5, 7)
WEIGHTS = ('uniform', 'distance')
ALPHAS = np.logspace(-3, 3, 13)


def load_data():
    path = next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob('12-2_Ridge_Logistic_예측성능_개선실험.py'))
    spec = importlib.util.spec_from_file_location('base_12_13', path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    data = base.load_data().copy()
    data['log_retention'] = np.log1p(data['board_2025_all']) - np.log1p(data['board_2024_all'])
    data['share_delta_pp'] = (data['share_post'] - data['share_pre']) * 100
    assert len(data) == 126 and data['셀 ID'].nunique() == 126
    return data


TARGETS = {
    '개통후승차인원': 'board_2025_all',
    '로그유지율': 'log_retention',
    '비중변화_pp': 'share_delta_pp',
}


def predict_knn(train, test, columns, y_col, setting):
    k, weight, log_target = setting
    if k > len(train):
        raise ValueError('K가 학습 격자보다 큽니다.')
    y = train[y_col].to_numpy(float)
    if log_target:
        y = np.log1p(y)
    model = make_pipeline(
        SimpleImputer(strategy='median'), StandardScaler(),
        KNeighborsRegressor(n_neighbors=k, weights=weight, metric='euclidean'),
    )
    model.fit(train[columns], y)
    prediction = model.predict(test[columns])
    if log_target:
        prediction = np.expm1(prediction)
    if y_col == 'board_2025_all':
        prediction = np.maximum(prediction, 0)
    return prediction


def predict_ridge(train, test, columns, y_col, setting):
    alpha, log_target = setting
    y = train[y_col].to_numpy(float)
    if log_target:
        y = np.log1p(y)
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=alpha))
    model.fit(train[columns], y)
    prediction = model.predict(test[columns])
    if log_target:
        prediction = np.expm1(prediction)
    if y_col == 'board_2025_all':
        prediction = np.maximum(prediction, 0)
    return prediction


def settings_for(kind, y_col):
    modes = (False, True) if y_col == 'board_2025_all' else (False,)
    if kind == 'KNN':
        return [(k, weight, log_mode) for k in K_VALUES for weight in WEIGHTS for log_mode in modes]
    return [(alpha, log_mode) for alpha in ALPHAS for log_mode in modes]


def select_inner(train, columns, y_col, kind):
    predictor = predict_knn if kind == 'KNN' else predict_ridge
    rows = []
    for setting in settings_for(kind, y_col):
        losses = []
        for held_out in train['nearest_station'].astype(str).unique():
            inner_train = train[train['nearest_station'].astype(str) != held_out]
            inner_test = train[train['nearest_station'].astype(str) == held_out]
            prediction = predictor(inner_train, inner_test, columns, y_col, setting)
            losses.append(mean_absolute_error(inner_test[y_col], prediction))
        rows.append((float(np.mean(losses)), setting))
    # 동률이면 작은 K와 균등 가중을 우선한다. 설정 순서는 재현 가능하다.
    return min(enumerate(rows), key=lambda z: (z[1][0], z[0]))[1]


def predict_baseline(train, test, y_col):
    if y_col == 'board_2025_all':
        ratio = train['board_2025_all'].sum() / train['board_2024_all'].sum()
        return ratio * test['board_2024_all'].to_numpy(float)
    if y_col == 'share_delta_pp':
        return np.zeros(len(test))
    return np.full(len(test), train[y_col].median())


def distance_diagnostics(train, test, columns):
    preprocessing = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
    x_train = preprocessing.fit_transform(train[columns])
    x_test = preprocessing.transform(test[columns])
    within = NearestNeighbors(n_neighbors=2).fit(x_train).kneighbors(x_train)[0][:, 1]
    outside = NearestNeighbors(n_neighbors=1).fit(x_train).kneighbors(x_test)[0][:, 0]
    return float(np.median(within)), float(np.median(outside))


def main():
    data = load_data()
    rows = []
    neighbor_distances = []
    for label, y_col in TARGETS.items():
        for station in STATIONS:
            train = data[data['nearest_station'].astype(str) != station]
            test = data[data['nearest_station'].astype(str) == station]
            truth = test[y_col].to_numpy(float)
            base = predict_baseline(train, test, y_col)
            rows.append({'대상': label, '검증역': station, '방법': '단순기준', 'K/규제': '-', 'n': len(test),
                         'MAE': mean_absolute_error(truth, base), '순위상관': np.nan})
            for feature_name, columns in FEATURE_SETS.items():
                for kind, predictor in [('KNN', predict_knn), ('Ridge', predict_ridge)]:
                    _, setting = select_inner(train, columns, y_col, kind)
                    predicted = predictor(train, test, columns, y_col, setting)
                    correlation = spearmanr(truth, predicted).statistic if np.ptp(predicted) > 0 else np.nan
                    rows.append({'대상': label, '검증역': station, '방법': f'{kind}_{feature_name}', 'n': len(test),
                                 'K/규제': str(setting), 'MAE': mean_absolute_error(truth, predicted),
                                 '순위상관': correlation})
                if label == '로그유지율':
                    within, outside = distance_diagnostics(train, test, columns)
                    neighbor_distances.append({'검증역': station, '입력변수': feature_name,
                                               '학습격자끼리_최근접거리중앙값': within,
                                               '검증역_최근접학습거리중앙값': outside})
    result = pd.DataFrame(rows)
    print('===== GTX-A 유사격자 KNN 탐색: 외부 LOSO, 내부 K·가중치 선택 =====')
    print('격자 126개·독립 역 3개. 성능은 목표별로만 비교하며 서로 다른 목표의 MAE를 비교하지 않는다.')
    for label in TARGETS:
        part = result[result['대상'] == label]
        print(f'\n===== {label} =====')
        summary = part.groupby('방법', sort=False)[['MAE', '순위상관']].mean()
        summary['전체126개_MAE'] = [
            np.average(part.loc[part['방법'] == name, 'MAE'],
                       weights=part.loc[part['방법'] == name, 'n'])
            for name in summary.index
        ]
        print(summary.round(4).to_string())
        print(part[['검증역', '방법', 'K/규제', 'MAE', '순위상관']].round(4).to_string(index=False))
    print('\n===== 유사격자 거리 진단 (각 학습 fold 표준화 거리) =====')
    print(pd.DataFrame(neighbor_distances).round(3).to_string(index=False))
    print('\n주의: 이 결과로 예측 목표나 KNN 사용 여부를 고르면 탐색적 선택이다.')
    print('후보가 비슷해도 결과가 비슷하지 않거나, 제외된 역의 최근접 거리가 크면 GTX-C 적용 근거가 약하다.')
    print('파일은 저장하지 않는다.')


if __name__ == '__main__':
    main()
