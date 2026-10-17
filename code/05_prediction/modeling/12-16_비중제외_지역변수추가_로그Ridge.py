"""비중 제외, 보행시간 포함 로그 Ridge: 지역변수 추가 16조합.

기본 4변수 고정. 후보 선택/전처리/alpha는 외부 학습역 내부에서만 수행.
기존 데이터를 반복 탐색한 실험이므로 독립 확인 연구로 해석하지 않는다.
결과는 화면에만 출력하며 결과 파일과 학습 모델을 저장하지 않는다.
"""
from pathlib import Path
import importlib.util
from itertools import combinations
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('base', next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob('12-12_예측대상_가능성_탐색.py')))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
CORE = ['log1p_board_2024', 'walk_min', 'competing_share', 'bus_time_mean']
EXTRA = [('로그인구', 'log1p_pop_20_50'), ('로그연면적', 'log1p_floor_area'),
         ('주거비율', '주거지역 비율(%)'), ('IC거리', 'ic_access')]
SPECS = [('기본4' + ''.join('+' + x[0] for x in subset), 'log', CORE + [x[1] for x in subset])
         for k in range(5) for subset in combinations(EXTRA, k)]
assert all('pre_share_pct' not in s[2] for s in SPECS)


def scores(y, pred):
    return {'MAE': base.mean_absolute_error(y, pred),
            'RMSE': np.sqrt(base.mean_squared_error(y, pred)), 'R2': base.r2_score(y, pred)}


def main():
    data = base.load_data().reset_index(drop=True)
    oof = {s[0]: np.full(len(data), np.nan) for s in SPECS}
    chosen_oof = np.full(len(data), np.nan)
    baseline_oof = np.full(len(data), np.nan)
    rows = []
    for station in base.STATIONS:
        train = data[data.nearest_station != station]
        valid = data[data.nearest_station == station]
        selected = base.inner_select(train, 'board_2025_all', SPECS)
        winner = min(selected, key=lambda s: (s[2], len(s[0][2])))
        for candidate, alpha, loss in selected:
            pred = base.predict(train, valid, 'board_2025_all', candidate, alpha)
            oof[candidate[0]][valid.index] = pred
            if candidate[0] == winner[0][0]:
                chosen_oof[valid.index] = pred
                rows.append({'검증역': station, '선택모형': candidate[0], 'alpha': alpha,
                             '내부MAE': loss, **scores(valid.board_2025_all, pred)})
        baseline_oof[valid.index] = base.predict(train, valid, 'board_2025_all', ('총량비례', 'ratio', []), 1.)
    table = []
    for candidate, _, features in SPECS:
        pred = oof[candidate]
        assert np.isfinite(pred).all()
        row = {'모형': candidate, '변수수': len(features), **scores(data.board_2025_all, pred)}
        for station in base.STATIONS:
            mask = data.nearest_station == station
            row[station+' MAE'] = base.mean_absolute_error(data.loc[mask, 'board_2025_all'], pred[mask])
        row['역평균MAE'] = np.mean([row[s+' MAE'] for s in base.STATIONS])
        table.append(row)
    print('기본4:', CORE)
    print('\n=== 16조합 비교: 외부 성능 정렬은 탐색용 ===')
    print(pd.DataFrame(table).sort_values('MAE').round(4).to_string(index=False))
    print('\n=== 변수 조합 선택을 포함한 중첩 LOSO ===')
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print('선택절차 전체 OOF:', scores(data.board_2025_all, chosen_oof))
    print('총량비례 기준 전체 OOF:', scores(data.board_2025_all, baseline_oof))
    selected = base.inner_select(data, 'board_2025_all', SPECS)
    winner = min(selected, key=lambda s: (s[2], len(s[0][2])))
    print('\n=== 전체 A자료 역별 CV로 선택한 후속 후보 (최종 모델 저장 없음) ===')
    print('모형:', winner[0][0], '입력:', winner[0][2], 'alpha:', winner[1], '선택CV MAE:', winner[2])
    print('정답: 실제 2025년 승차인원을 log1p 변환. 예측: 역변환한 개통 후 승차인원(명).')
    print('전체자료 선택CV 성능은 독립 성능이 아니며 C 예측력도 입증하지 않음.')
    print('결과 파일 저장 없음.')


if __name__ == '__main__':
    main()
