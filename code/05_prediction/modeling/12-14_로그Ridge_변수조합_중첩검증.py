"""로그 Ridge 2~9변수 502조합 탐색. 결과는 화면에만 출력한다.

외부 LOSO 안에서 조합/alpha를 내부 역별 MAE 평균으로 선택한다.
외부 결과로 정렬한 표는 탐색용이며 최종 선택의 독립 성능이 아니다.
전체 A자료 CV 선택 결과는 배포 후보이며 독립 검증 결과가 아니다.
"""
from pathlib import Path
import importlib.util
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('base', next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob('12-12_예측대상_가능성_탐색.py')))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
FEATURES = base.FULL
LABELS = ['개통전비중', '로그개통전승차', '보행시간', '경쟁목적지비율',
          '버스시간', 'IC거리', '로그인구', '로그연면적', '주거비율']
ALPHAS = base.ALPHAS
COMBOS = [c for k in range(2, 10) for c in combinations(range(9), k)]


def name(c):
    return '+'.join(LABELS[i] for i in c)


def prepare(train, valid):
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    x = scaler.fit_transform(imputer.fit_transform(train[FEATURES]))
    z = scaler.transform(imputer.transform(valid[FEATURES]))
    assert x.shape[1] == len(FEATURES), '전체 결측 변수 확인 필요'
    y = np.log1p(train.board_2025_all.to_numpy(float))
    return x, z, y


def predictions(prepared, combo):
    # Ridge의 절편 비규제 해를 고유값 분해로 계산: 모든 alpha를 동시에 평가.
    x, z, y = prepared
    x, z = x[:, combo], z[:, combo]
    center = x.mean(axis=0)
    x, z = x - center, z - center
    values, vectors = np.linalg.eigh(x.T @ x)
    rhs = vectors.T @ (x.T @ (y - y.mean()))
    weights = vectors @ (rhs[:, None] / (np.maximum(values, 0)[:, None] + ALPHAS))
    return np.maximum(np.expm1(z @ weights + y.mean()), 0)


def select(train):
    losses = np.zeros((len(COMBOS), len(ALPHAS)))
    groups = train.nearest_station.unique()
    for station in groups:
        fit, valid = train[train.nearest_station != station], train[train.nearest_station == station]
        prepared = prepare(fit, valid)
        truth = valid.board_2025_all.to_numpy(float)
        for i, combo in enumerate(COMBOS):
            losses[i] += np.abs(predictions(prepared, combo) - truth[:, None]).mean(axis=0) / len(groups)
    # 손실이 같으면 강한 규제, 조합 간 손실이 같으면 적은 변수 선택.
    alpha_ids = losses.shape[1] - 1 - losses[:, ::-1].argmin(axis=1)
    scores = losses[np.arange(len(COMBOS)), alpha_ids]
    winner = min(range(len(COMBOS)), key=lambda i: (scores[i], len(COMBOS[i])))
    return winner, alpha_ids, scores


def metrics(y, p):
    return {'MAE': mean_absolute_error(y, p), 'RMSE': np.sqrt(mean_squared_error(y, p)), 'R2': r2_score(y, p)}


def main():
    data = base.load_data().reset_index(drop=True)
    all_pred = np.zeros((len(data), len(COMBOS)))
    nested = np.zeros(len(data))
    baseline = np.zeros(len(data))
    rows = []
    for station in base.STATIONS:
        train, valid = data[data.nearest_station != station], data[data.nearest_station == station]
        winner, alpha_ids, scores = select(train)
        prepared = prepare(train, valid)
        for i, combo in enumerate(COMBOS):
            all_pred[valid.index, i] = predictions(prepared, combo)[:, alpha_ids[i]]
        nested[valid.index] = all_pred[valid.index, winner]
        baseline[valid.index] = valid.board_2024_all * train.board_2025_all.sum() / train.board_2024_all.sum()
        rows.append({'검증역': station, 'n': len(valid), '선택변수': name(COMBOS[winner]),
                     'alpha': ALPHAS[alpha_ids[winner]], '내부MAE': scores[winner],
                     **metrics(valid.board_2025_all, nested[valid.index])})
    print('\n=== 조합 선택까지 포함한 중첩 LOSO ===')
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print('전체 OOF:', metrics(data.board_2025_all, nested))
    print('학습역 총량비례 기준:', metrics(data.board_2025_all, baseline))
    # 계산식이 기존 sklearn Ridge 파이프라인과 일치하는지 검증.
    station = base.STATIONS[0]
    train, valid = data[data.nearest_station != station], data[data.nearest_station == station]
    check = base.predict(train, valid, 'board_2025_all', ('A3', 'log', base.A3), 0.01)
    actual = predictions(prepare(train, valid), (0, 1, 2, 3))[:, np.argmin(abs(ALPHAS - .01))]
    np.testing.assert_allclose(actual, check, rtol=1e-7, atol=1e-7)
    summary = []
    for i, combo in enumerate(COMBOS):
        p = all_pred[:, i]
        row = {'변수수': len(combo), '변수': name(combo), **metrics(data.board_2025_all, p)}
        for station in base.STATIONS:
            mask = data.nearest_station == station
            row[station+' MAE'] = mean_absolute_error(data.loc[mask, 'board_2025_all'], p[mask])
        row['역평균 MAE'] = np.mean([row[s+' MAE'] for s in base.STATIONS])
        summary.append(row)
    table = pd.DataFrame(summary)
    print('\n=== 외부 OOF MAE 상위 12조합: 탐색용, 최고값을 최종성능으로 주장 금지 ===')
    print(table.sort_values('MAE').head(12).round(4).to_string(index=False))
    print('\n=== 변수 개수별 탐색상 최저 MAE ===')
    print(table.loc[table.groupby('변수수').MAE.idxmin()].round(4).to_string(index=False))
    print('\n=== 기존 4변수 ===')
    print(table[table.변수 == name((0, 1, 2, 3))].round(4).to_string(index=False))
    winner, alpha_ids, scores = select(data)
    print('\n=== 전체 A자료의 역별 CV로 선택한 후속 적합 후보 ===')
    print(name(COMBOS[winner]), 'alpha=', ALPHAS[alpha_ids[winner]], '선택 CV MAE=', scores[winner])
    print('이 후보의 선택 CV 값은 독립 성능이 아님. C 적용 전 변수정의/분포 검토 필요.')
    print('반복 탐색 및 독립 역 3개 한계가 남음. 결과/모델 파일 저장 없음.')


if __name__ == '__main__':
    main()
