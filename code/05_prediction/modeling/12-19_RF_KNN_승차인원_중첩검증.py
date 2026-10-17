"""고정된 변수구성으로 Ridge / 얕은 RF / KNN 비교. 결과 파일 저장 없음.

정답: 원 단위 2025년 승차인원. 개통 전 비중 제외.
외부 LOSO 3회, 내부 LOSO 2회에서 전처리와 설정 선택.
내부 선택 기준: 역별 MAE 단순평균. 외부 결과로 후보/설정을 재선택하지 않는다.
같은 자료를 반복 이용한 탐색이며 GTX-C에 대한 독립 검증은 아니다.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('existing_base', next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob('12-12_예측대상_가능성_탐색.py')))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
FEATURES = {
    '승차단독': ['board_2024_all'],
    '전체8변수': ['board_2024_all', 'log1p_walk_min', 'competing_share',
                 'bus_time_mean', 'ic_access', 'log1p_pop_20_50',
                 'log1p_floor_area', '주거지역 비율(%)'],
}
# 동률이면 목록의 앞 후보 선택: 강한 Ridge 규제/얕은 RF/많은 K 순서.
PARAMS = {
    'Ridge': [{'alpha':float(a)} for a in base.ALPHAS[::-1]],
    'RF': [{'max_depth':d, 'min_samples_leaf':l} for d in [2,4] for l in [8,3]],
    'KNN': [{'n_neighbors':k, 'weights':w} for k in [11,7,5,3] for w in ['uniform','distance']],
}

def predict(train, test, alg, features, params):
    if alg == 'Ridge':
        estimator = Ridge(**params)
    elif alg == 'RF':
        estimator = RandomForestRegressor(n_estimators=200, random_state=42,
                                         n_jobs=1, max_features=1.0, **params)
    else:
        estimator = KNeighborsRegressor(metric='euclidean', **params)
    steps = [SimpleImputer(strategy='median')]
    if alg != 'RF':
        steps.append(StandardScaler())
    pipe = make_pipeline(*steps, estimator)
    pipe.fit(train[features], train.board_2025_all)
    return np.maximum(pipe.predict(test[features]),0)

def metrics(y,p):
    return {'MAE':mean_absolute_error(y,p), 'RMSE':np.sqrt(mean_squared_error(y,p)),
            'R2':r2_score(y,p), '평균오차(예측-실제)':np.mean(p-y)}

def main():
    data=base.load_data().reset_index(drop=True)
    assert len(data)==126 and data['셀 ID'].nunique()==126
    oof={}; folds=[]
    print('정답=2025년 승차인원 원 단위. 표본126개. 비중 제외.',flush=True)
    print('8변수:',FEATURES['전체8변수'],flush=True)
    for alg in PARAMS:
        for feature_name, features in FEATURES.items():
            name=alg+'_'+feature_name
            out=np.full(len(data),np.nan)
            for station in base.STATIONS:
                train=data[data.nearest_station!=station]
                test=data[data.nearest_station==station]
                assert set(train['셀 ID']).isdisjoint(test['셀 ID'])
                losses=[]
                for params in PARAMS[alg]:
                    values=[]
                    for inner_station in train.nearest_station.unique():
                        fit=train[train.nearest_station!=inner_station]
                        valid=train[train.nearest_station==inner_station]
                        p=predict(fit,valid,alg,features,params)
                        values.append(mean_absolute_error(valid.board_2025_all,p))
                    losses.append(np.mean(values))
                index=int(np.argmin(losses)); chosen=PARAMS[alg][index]
                p=predict(train,test,alg,features,chosen)
                out[test.index]=p
                folds.append({'모형':name,'검증역':station,'n':len(test),
                              '설정':str(chosen),'내부역평균MAE':losses[index],
                              **metrics(test.board_2025_all.to_numpy(),p)})
            assert np.isfinite(out).all()
            oof[name]=out
            print(name+' 검증 완료',flush=True)
    y=data.board_2025_all.to_numpy()
    rows=[]
    for name,p in oof.items():
        row={'모형':name,**metrics(y,p)}
        for station in base.STATIONS:
            mask=(data.nearest_station==station).to_numpy()
            row[station+' MAE']=mean_absolute_error(y[mask],p[mask])
        rows.append(row)
    assert abs(mean_absolute_error(y,oof['Ridge_승차단독'])-11.8223)<0.0001
    print('\n===== 전체 OOF 성능 및 역별 MAE =====')
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print('\n===== 학습역 내부에서 선택한 설정 =====')
    print(pd.DataFrame(folds).round(4).to_string(index=False))
    rows=[]
    bins=pd.cut(data.board_2024_all,[-np.inf,40,100,300,np.inf],right=False,
                labels=['40미만','40~100미만','100~300미만','300이상'])
    for label in bins.cat.categories:
        mask=(bins==label).to_numpy()
        row={'2024년 승차규모':label,'n':int(mask.sum())}
        for name,p in oof.items():
            row[name]=mean_absolute_error(y[mask],p[mask])
        rows.append(row)
    print('\n===== 개통 전 승차규모별 MAE =====')
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print('\n===== 동일 알고리즘 내 지역변수 추가 기여 =====')
    for alg in PARAMS:
        gain=abs(y-oof[alg+'_승차단독'])-abs(y-oof[alg+'_전체8변수'])
        print(alg, 'MAE 개선(양수=개선):',round(gain.mean(),4),
              '개선/악화/동률:',int((gain>1e-8).sum()),int((gain< -1e-8).sum()),int((abs(gain)<=1e-8).sum()))
    print('\n기존 기준모형 재현 확인 완료. 결과/모델 파일 저장 없음.')
    print('외부 성능을 보고 최고 알고리즘을 선택하면 선택 편향이 남는다. GTX-C 독립검증이 아니다.')

if __name__=='__main__':
    main()
