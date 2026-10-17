"""상대적 수요 변화율 차이(%p) Ridge 탐색. 화면 출력 전용.

Y = 격자 변화율 - 소속 역 권역의 총량 변화율.
X = 로그 보행시간, 경쟁 목적지 비율, 주거지역 비율, 로그 연면적.
외부 LOSO, 학습역 내부 LOSO에서 alpha 선택/결측치 대체/표준화.
0명 기초 격자는 Y가 정의되지 않으므로 회귀에서 제외하되 권역 총량에는 포함.
기존 OLS/분류/순위모형을 변경하지 않으며 CSV/모델 파일을 저장하지 않는다.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

HERE=Path(__file__).resolve().parent
FEATURES=['log1p_walk_min','competing_share','주거지역 비율(%)','log1p_floor_area']
ALPHAS=np.logspace(-3,3,13)
STATIONS=['운정중앙','킨텍스','대곡']

def load_data():
    spec=importlib.util.spec_from_file_location('relative_base',next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob('12-2_Ridge_Logistic_예측성능_개선실험.py')))
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    all_data=base.load_data().reset_index(drop=True)
    assert len(all_data)==126 and all_data['셀 ID'].is_unique
    counts=all_data[['board_2024_all','board_2025_all']]
    assert counts.notna().all().all() and (counts>=0).all().all()
    for raw in ['walk_min','competing_share','주거지역 비율(%)','2024년 건축물 연면적(천㎡)']:
        if (all_data[raw].dropna()<0).any(): raise ValueError(f'음수 입력: {raw}')
    totals=all_data.groupby('nearest_station').agg(n=('셀 ID','size'),
                  pre=('board_2024_all','sum'),post=('board_2025_all','sum'),
                  zero_pre=('board_2024_all',lambda s:int((s==0).sum())))
    assert (totals.pre>0).all() and (totals.post>0).all()
    totals['R_s']=100*(totals.post/totals.pre-1)
    all_data['R_s']=all_data.nearest_station.map(totals.R_s)
    # +1 보정 없이 0 분모는 결측으로 처리한다.
    all_data['R_i']=100*(all_data.board_2025_all/all_data.board_2024_all.where(all_data.board_2024_all>0)-1)
    all_data['Y']=all_data.R_i-all_data.R_s
    data=all_data[all_data.board_2024_all>0].copy().reset_index(drop=True)
    assert np.isfinite(data.Y).all()
    assert ((data.Y>0).astype(int)==data.share_increase).all()
    # 권역 총량은 정답 정의에만 사용하고 입력·설정 선택에는 사용하지 않는다.
    assert 'board_2024_all' not in FEATURES and 'R_s' not in FEATURES
    return all_data,data,totals

def fit(train,alpha):
    pipe=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=alpha))
    pipe.fit(train[FEATURES],train.Y)
    return pipe

def choose_alpha(train):
    rows=[]
    for alpha in ALPHAS:
        losses=[]
        for station in train.nearest_station.unique():
            a=train[train.nearest_station!=station];b=train[train.nearest_station==station]
            assert set(a['셀 ID']).isdisjoint(b['셀 ID'])
            pred=fit(a,alpha).predict(b[FEATURES])
            losses.append(mean_absolute_error(b.Y,pred))
        rows.append((np.mean(losses),alpha))
    loss,alpha=min(rows,key=lambda row:(row[0],-row[1]))
    return alpha,loss

def metrics(y,p):
    return {'MAE(%p)':mean_absolute_error(y,p),'RMSE(%p)':np.sqrt(mean_squared_error(y,p)),
            'R2':r2_score(y,p) if len(y)>1 and np.ptp(y)>0 else np.nan}

def evaluate(data):
    parts=[];folds=[];coefficients=[]
    for station in STATIONS:
        train=data[data.nearest_station!=station];test=data[data.nearest_station==station].copy()
        assert set(train['셀 ID']).isdisjoint(test['셀 ID'])
        alpha,loss=choose_alpha(train);pipe=fit(train,alpha)
        test['Ridge']=pipe.predict(test[FEATURES]) # 음수 Y도 유효하므로 0으로 잘라내지 않는다.
        test['0 기준']=0.;test['학습평균 기준']=train.Y.mean();test['학습중앙값 기준']=train.Y.median()
        for model in ['Ridge','0 기준','학습평균 기준','학습중앙값 기준']:
            folds.append({'검증역':station,'모형':model,'n':len(test),
                          'alpha':alpha if model=='Ridge' else np.nan,
                          '내부역평균MAE':loss if model=='Ridge' else np.nan,
                          **metrics(test.Y.to_numpy(),test[model].to_numpy())})
        coefs=pipe.named_steps['ridge'].coef_
        coefficients.append({'검증역':station,'alpha':alpha,'절편':pipe.named_steps['ridge'].intercept_,
                             **dict(zip(FEATURES,coefs))})
        parts.append(test)
    oof=pd.concat(parts,ignore_index=True)
    assert len(oof)==len(data) and oof['셀 ID'].is_unique and np.isfinite(oof.Ridge).all()
    overall=pd.DataFrame([{'모형':m,**metrics(oof.Y.to_numpy(),oof[m].to_numpy())}
                          for m in ['Ridge','0 기준','학습평균 기준','학습중앙값 기준']])
    ranks=[];tops=[]
    for station,g in oof.groupby('nearest_station',sort=False):
        y=g.Y.to_numpy();p=g.Ridge.to_numpy();pos=y>0
        auc=roc_auc_score(pos,p) if len(np.unique(pos))==2 else np.nan
        rho=spearmanr(y,p).statistic if np.ptp(p)>0 else np.nan
        ranks.append({'검증역':station,'n':len(g),'Y>0 격자수':int(pos.sum()),'Y>0 비율':pos.mean(),
                      'ROC_AUC(Y>0)':auc,'Spearman(연속Y)':rho})
        order=g.sort_values(['Ridge','셀 ID'],ascending=[False,True])
        for rate in [.1,.2,.3]:
            k=int(np.ceil(len(g)*rate));head=order.head(k);precision=(head.Y>0).mean()
            tops.append({'검증역':station,'상위비율':rate,'K':k,'Y>0 격자수':int((head.Y>0).sum()),
                         'Precision':precision,'Lift':precision/pos.mean() if pos.any() else np.nan,
                         '실제Y중앙값':head.Y.median(),'실제Y평균':head.Y.mean()})
    oof['절대오차']=abs(oof.Ridge-oof.Y)
    oof['기초승차규모']=pd.cut(oof.board_2024_all,[0,5,10,20,np.inf],right=False,
                                labels=['1~4','5~9','10~19','20명 이상'])
    size=[]
    for label,g in oof.groupby('기초승차규모',observed=True):
        size.append({'기초승차규모':str(label),'n':len(g),'R_i최소':g.R_i.min(),'R_i최대':g.R_i.max(),
                     'Y중앙값':g.Y.median(),'Y최소':g.Y.min(),'Y최대':g.Y.max(),
                     '절대오차합비중':g['절대오차'].sum()/oof['절대오차'].sum(),
                     **metrics(g.Y.to_numpy(),g.Ridge.to_numpy())})
    tables=[overall,pd.DataFrame(folds),pd.DataFrame(ranks),pd.DataFrame(tops),pd.DataFrame(size),pd.DataFrame(coefficients)]
    return (oof,*tables)

def main():
    all_data,data,totals=load_data()
    oof,overall,folds,ranks,tops,size,coefs=evaluate(data)
    print('입력:',FEATURES,'| 개통 전 승차인원 입력 제외 | 보행/연면적 log1p')
    print('전체',len(all_data),'기초 0명',int((all_data.board_2024_all==0).sum()),'회귀',len(data))
    print('0명 격자는 회귀 제외, 권역 총량 포함. 0명 격자에 +1 보정 안 함.')
    for title,table in [('권역 총량 및 변화율',totals),('전체 OOF',overall),('역별 검증',folds),
                        ('순위 진단',ranks),('상위 순위',tops),('기초승차 규모별',size),('표준화 계수',coefs)]:
        print('\n=====',title,'=====');print(table.round(4).to_string())
    print('\n===== 극단 Y 격자 =====')
    print(oof.loc[oof.Y.abs().nlargest(12).index,['셀 ID','nearest_station','board_2024_all','board_2025_all','R_i','R_s','Y','Ridge','절대오차']].round(3).to_string(index=False))
    print('\nY>0는 기초승차>0인 격자에서 기존 비중 확대 라벨과 수학적으로 동일하다.')
    print('검증역 R_s는 정답 정의에만 사용하며 X/alpha 선택/예측에는 사용하지 않는다.')
    print('같은 자료의 반복 탐색이며 GTX-B·C 독립검증 또는 최종모형 선정이 아니다.')
    print('CSV/모델/그림 파일 저장 없음.')

if __name__=='__main__':main()
