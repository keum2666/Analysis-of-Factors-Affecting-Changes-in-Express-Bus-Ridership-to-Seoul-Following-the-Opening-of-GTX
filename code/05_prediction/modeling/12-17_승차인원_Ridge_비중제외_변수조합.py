"""원 단위 승차인원 Ridge의 지역·교통 변수 64조합 탐색.

개통 전 승차인원과 log(1+보행시간)은 모든 조합에 포함하고,
개통 전 수요 비중은 제외한다. 조합·alpha는 외부 학습역의
내부 LOSO에서 선택한다. 외부 결과로 정렬한 순위는 탐색용이다.
결과/모델 파일을 저장하지 않는다.
"""
from pathlib import Path
import importlib.util
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ridge_experiment_base', next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob('12-12_예측대상_가능성_탐색.py')))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

REQUIRED = ['board_2024_all', 'log1p_walk_min']
EXTRA = [
    ('경쟁목적지비율', 'competing_share'),
    ('평균버스시간', 'bus_time_mean'),
    ('IC거리', 'ic_access'),
    ('로그인구', 'log1p_pop_20_50'),
    ('로그연면적', 'log1p_floor_area'),
    ('주거지역비율', '주거지역 비율(%)'),
]
SPECS = [
    ('기본2' + ''.join('+' + x[0] for x in subset), 'raw',
     REQUIRED + [x[1] for x in subset])
    for k in range(len(EXTRA) + 1)
    for subset in combinations(EXTRA, k)
]
assert len(SPECS) == 64
assert all('pre_share_pct' not in s[2] and s[2][:2] == REQUIRED for s in SPECS)


def stats(y, p):
    return {'MAE': mean_absolute_error(y, p),
            'RMSE': np.sqrt(mean_squared_error(y, p)),
            'R2': r2_score(y, p)}


def main():
    data = base.load_data().reset_index(drop=True)
    assert len(data) == 126 and data['셀 ID'].nunique() == 126
    oof = {s[0]: np.full(len(data), np.nan) for s in SPECS}
    selected_oof = np.full(len(data), np.nan)
    pre_only_oof = np.full(len(data), np.nan)
    ratio_oof = np.full(len(data), np.nan)
    folds = []
    for station in base.STATIONS:
        train = data[data.nearest_station != station]
        valid = data[data.nearest_station == station]
        candidates = base.inner_select(train, 'board_2025_all', SPECS)
        winner = min(candidates, key=lambda row: (row[2], len(row[0][2]), row[0][0]))
        for model, alpha, inner_mae in candidates:
            pred = base.predict(train, valid, 'board_2025_all', model, alpha)
            oof[model[0]][valid.index] = pred
            if model[0] == winner[0][0]:
                selected_oof[valid.index] = pred
                folds.append({'검증역': station, '선택 변수': ', '.join(model[2]),
                              'alpha': alpha, '내부역평균MAE': inner_mae,
                              **stats(valid.board_2025_all, pred)})
        pre_spec = ('승차인원단독', 'raw', ['board_2024_all'])
        pre_choice = base.inner_select(train, 'board_2025_all', [pre_spec])[0]
        pre_only_oof[valid.index] = base.predict(train, valid, 'board_2025_all',
                                                 pre_spec, pre_choice[1])
        ratio_oof[valid.index] = base.predict(train, valid, 'board_2025_all',
                                              ('학습역총량비례', 'ratio', []), 1.)
    y = data.board_2025_all.to_numpy(float)
    assert all(np.isfinite(p).all() for p in [*oof.values(), selected_oof,
                                             pre_only_oof, ratio_oof])
    table = []
    for model in SPECS:
        name, _, features = model
        pred = oof[name]
        row = {'모형': name, '변수수': len(features), **stats(y, pred)}
        for station in base.STATIONS:
            mask = data.nearest_station == station
            row[station + ' MAE'] = mean_absolute_error(y[mask], pred[mask])
        row['역평균MAE'] = np.mean([row[s + ' MAE'] for s in base.STATIONS])
        table.append(row)
    result = pd.DataFrame(table).sort_values('MAE')
    print('===== 고정 조건: 개통 전 승차인원 + log(1+보행시간), 비중 제외 =====')
    print('후보 지역·교통 변수:', [name for name, _ in EXTRA])
    print('64개 조합, alpha 후보 13개; 결측치/표준화는 학습 fold 내부에서만 처리.')
    print('\n===== 외부 OOF MAE 상위 12조합: 탐색 결과 =====')
    print(result.head(12).round(4).to_string(index=False))
    print('\n===== 변수 개수별 탐색상 최저 MAE =====')
    print(result.loc[result.groupby('변수수').MAE.idxmin()].sort_values('변수수').round(4).to_string(index=False))
    print('\n===== 조합과 alpha 선택을 포함한 중첩 LOSO =====')
    print(pd.DataFrame(folds).round(4).to_string(index=False))
    print('\n===== 전체 126개 OOF 비교 =====')
    print(pd.DataFrame([
        {'모형': '승차인원 단독 Ridge', **stats(y, pre_only_oof)},
        {'모형': '학습역 총량비례 기준', **stats(y, ratio_oof)},
        {'모형': '지역변수 조합 선택 절차', **stats(y, selected_oof)},
        *[{'모형': name, **stats(y, oof[name])} for name in
          ['기본2', '기본2+경쟁목적지비율+평균버스시간+로그연면적']],
    ]).round(4).to_string(index=False))
    print('\n주의: 상위 조합을 외부 OOF 성능으로 선택하면 낙관적 편향이 있다.')
    print('같은 3개 역·126개 격자를 반복 사용한 탐색이며 GTX-C의 독립 검증은 아니다.')
    print('결과/모델 파일은 저장하지 않는다.')


if __name__ == '__main__':
    main()
