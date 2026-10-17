"""보행시간을 포함한 5변수 로그 Ridge와 비중 제외 모형 비교.

외부 LOSO에서 학습역 내부 CV로 alpha 선택. 결과는 화면 출력만 한다.
후보는 이전 탐색 후 지정되었으므로 새로운 독립 검증으로 해석하지 않는다.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('base', next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob('12-12_예측대상_가능성_탐색.py')))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
FIVE = base.A3 + ['bus_time_mean']
SPECS = [('기존4변수', 'log', base.A3),
         ('보행포함5변수', 'log', FIVE),
         ('비중제외4변수', 'log', [x for x in FIVE if x != 'pre_share_pct'])]


def main():
    data = base.load_data().reset_index(drop=True)
    rows = []
    oof = {s[0]: np.full(len(data), np.nan) for s in SPECS}
    for station in base.STATIONS:
        train = data[data.nearest_station != station]
        valid = data[data.nearest_station == station]
        for candidate, alpha, inner_loss in base.inner_select(train, 'board_2025_all', SPECS):
            pred = base.predict(train, valid, 'board_2025_all', candidate, alpha)
            oof[candidate[0]][valid.index] = pred
            rows.append({'모형': candidate[0], '검증역': station, 'n': len(valid),
                         'alpha': alpha, '내부MAE': inner_loss,
                         **base.metrics(valid, pred, 'board_2025_all')})
    print('입력변수:', {s[0]: s[2] for s in SPECS})
    print('정답: log(1+실제 2025년 승차인원); 출력: expm1 역변환한 예상 승차인원(명)')
    print(pd.DataFrame(rows)[['모형', '검증역', 'n', 'alpha', 'MAE', 'RMSE', 'R2']].round(4).to_string(index=False))
    overall = []
    for model, pred in oof.items():
        assert np.isfinite(pred).all()
        y = data.board_2025_all.to_numpy(float)
        overall.append({'모형': model, 'MAE': base.mean_absolute_error(y, pred),
                        'RMSE': np.sqrt(base.mean_squared_error(y, pred)), 'R2': base.r2_score(y, pred)})
    print('\n전체 OOF:')
    print(pd.DataFrame(overall).round(4).to_string(index=False))
    print('반복 탐색 후 지정한 후보 비교이며 독립 역은 3개. C 성능 입증 아님.')
    print('결과 및 모델 파일 저장 없음.')


if __name__ == '__main__':
    main()
