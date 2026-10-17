"""기존 모형의 LOSO 예측을 동일 설정으로 재현해 추가 기여를 진단한다.

새 변수/규제 탐색 없이 기존 12-12의 학습역 내부 alpha 선택을 재현한다.
12-17 최상위 조합은 탐색용 비교이며 최종 선택/독립검증으로 해석하지 않는다.
구간별 표는 설명용이며 새로운 후보 선택에 사용하지 않는다. 결과 파일 저장 없음.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('base', next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob('12-12_예측대상_가능성_탐색.py')))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
MODELS = [
    ('승차단독', 'raw', ['board_2024_all']),
    ('기존5변수', 'raw', ['board_2024_all', 'log1p_walk_min', 'competing_share', 'bus_time_mean', 'log1p_floor_area']),
    ('탐색최상위5변수', 'raw', ['board_2024_all', 'log1p_walk_min', 'competing_share', 'ic_access', 'log1p_floor_area']),
]

def summary(g, name):
    e0, e1 = g['승차단독_절대오차'], g[name+'_절대오차']
    gain = e0-e1
    return {'n':len(g), '실제평균':g.board_2025_all.mean(),
            '단독MAE':e0.mean(), '지역MAE':e1.mean(), 'MAE개선':gain.mean(),
            '개선격자':int((gain>1e-8).sum()), '악화격자':int((gain < -1e-8).sum()),
            '개선중앙값':gain.median()}

def main():
    d = base.load_data().reset_index(drop=True)
    assert len(d)==126 and d['셀 ID'].nunique()==126
    for model in MODELS:
        p = np.full(len(d), np.nan)
        for station in base.STATIONS:
            train, valid = d[d.nearest_station!=station], d[d.nearest_station==station]
            chosen = base.inner_select(train, 'board_2025_all', [model])[0]
            p[valid.index] = base.predict(train, valid, 'board_2025_all', model, chosen[1])
        assert np.isfinite(p).all()
        d[model[0]+'_예측'] = p
        d[model[0]+'_절대오차'] = abs(p-d.board_2025_all)
    expected = [11.8223,13.4343,12.0477]
    for model, ref in zip(MODELS, expected):
        assert abs(d[model[0]+'_절대오차'].mean()-ref)<0.0001
    d['승차규모'] = pd.cut(d.board_2024_all, [-np.inf,40,100,300,np.inf], right=False,
                          labels=['40명 미만','40~100명 미만','100~300명 미만','300명 이상'])
    rows=[]
    for name, _, _ in MODELS[1:]:
        rows.append({'모형':name,'구분':'전체', **summary(d,name)})
        for field in ['nearest_station','승차규모']:
            for label,g in d.groupby(field,observed=True):
                rows.append({'모형':name,'구분':label, **summary(g,name)})
    print('양의 MAE개선 = 지역변수 포함 모형의 오차가 작음')
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    for name,_,_ in MODELS[1:]:
        print('\n===== 변수별 4분위 오차 진단:',name,'=====')
        rows=[]
        for col in ['walk_min','competing_share','bus_time_mean','ic_access','log1p_floor_area']:
            group = pd.qcut(d[col],4,duplicates='drop')
            for label, g in d.groupby(group,observed=True):
                row={'변수':col,'구간':str(label),**summary(g,name)}
                for station in base.STATIONS:
                    s = g[g.nearest_station==station]
                    row[station+'_n']=len(s)
                    row[station+'_개선']=(s['승차단독_절대오차']-s[name+'_절대오차']).mean()
                rows.append(row)
        print(pd.DataFrame(rows).round(3).to_string(index=False))
        gain=d['승차단독_절대오차']-d[name+'_절대오차']
        view=d[['셀 ID','nearest_station','board_2024_all','board_2025_all','승차단독_예측',name+'_예측','walk_min','competing_share']].copy()
        view['오차개선']=gain
        print('\n개선 상위 6개'); print(view.nlargest(6,'오차개선').round(3).to_string(index=False))
        print('\n악화 상위 6개'); print(view.nsmallest(6,'오차개선').round(3).to_string(index=False))
    print('\n기존 수치 재현 확인 완료. 결과 파일 저장 없음. 구간별 분석은 탐색적 설명용.')

if __name__=='__main__':
    main()
