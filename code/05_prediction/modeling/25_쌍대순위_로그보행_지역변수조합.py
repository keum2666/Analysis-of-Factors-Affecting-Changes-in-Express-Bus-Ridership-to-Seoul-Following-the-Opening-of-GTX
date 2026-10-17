"""로그 보행·로그 연면적·경쟁비율 고정, 지역변수 16조합 탐색.

126개와 기존 정답 유지. 승차인원/승차비중 입력 제외.
각 후보 C 선택 및 후보 선택 절차 모두 외부 검증역을 제외한다.
후보별 외부 최고점은 탐색값이며 확정 성능이 아니다. 결과 파일 저장 없음.
"""
from pathlib import Path
import importlib.util
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
FIXED = ['log1p_walk_min', 'log1p_floor_area', 'competing_share']
EXTRA = ['주거지역 비율(%)', 'log1p_pop_20_50', 'bus_time_mean', 'ic_access']
NAMES = dict(zip(EXTRA, ['주거비율', '로그인구', '버스시간', 'IC거리']))


def evaluate():
    spec = importlib.util.spec_from_file_location('rank25', next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob('12-8_A3_쌍대비교_순위모형.py')))
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    data = r.load_base_module().load_data()
    assert len(data) == 126 and data['셀 ID'].is_unique
    assert np.isfinite(data[FIXED+EXTRA].to_numpy(float)).all()
    candidates = [list(c) for k in range(5) for c in combinations(EXTRA, k)]
    rows = []
    for extra in candidates:
        r.FEATURES = FIXED+extra
        name = '+'.join(NAMES[f] for f in extra) or '기본3변수'
        for station in data.nearest_station.unique():
            train = data[data.nearest_station != station]
            test = data[data.nearest_station == station].copy()
            c, inner = r.choose_c_inner_loso(train)
            x, y = r.make_pairs(train)
            model = r.make_model(c).fit(x, y)
            test['점수'] = r.cell_scores(model, test)
            top = test.sort_values('점수', ascending=False).head(int(np.ceil(len(test)*.2)))
            loss, pairs = r.pairwise_log_loss(test, test['점수'].to_numpy())
            rows.append(dict(추가변수=name, 변수수=len(r.FEATURES), 검증역=station,
                             C=c, inner=inner, AUC=roc_auc_score(test.share_increase, test['점수']),
                             n=len(test), K=len(top), hits=int(top.share_increase.sum()),
                             Precision=top.share_increase.mean(),
                             Lift=top.share_increase.mean()/test.share_increase.mean(),
                             Loss=loss, pairs=pairs))
    folds = pd.DataFrame(rows)
    summaries = []
    for name, g in folds.groupby('추가변수', sort=False):
        row = {'추가변수':name, '변수수':g['변수수'].iloc[0], '평균AUC':g.AUC.mean(),
               '상위적중':f'{g.hits.sum()}/{g.K.sum()}', 'Precision':g.hits.sum()/g.K.sum(),
               'Loss':np.average(g.Loss, weights=g.pairs)}
        row.update({f'{s}AUC':p.AUC for s,p in g.set_index('검증역').iterrows()})
        summaries.append(row)
    # 각 외부 fold에서 후보와 C 모두 내부 AUC만으로 선택. 동률은 변수 수가 적은 후보 우선.
    chosen = (folds.sort_values(['inner','변수수','추가변수'], ascending=[False,True,True])
              .groupby('검증역', sort=False).head(1))
    return pd.DataFrame(summaries).sort_values('평균AUC', ascending=False), folds, chosen


if __name__ == '__main__':
    summary, folds, chosen = evaluate()
    print('고정: log(1+보행시간), log(1+연면적[천㎡]), 경쟁목적지비율. 표본126, 기존 정답 유지.')
    print(summary.round(4).to_string(index=False))
    print('\n내부 검증으로 변수조합까지 선택한 외부 검증 결과')
    print(chosen.round(4).to_string(index=False))
    print('선택절차 평균AUC:', round(chosen.AUC.mean(),4))
    print('선택절차 상위적중:', int(chosen.hits.sum()), '/', int(chosen.K.sum()))
    print('모든 후보와 선택절차는 동일한 3개 역을 반복 탐색한 결과로 독립검증이 아니다.')
    print('결과 CSV/모델/그림 파일 저장 없음.')
