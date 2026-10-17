"""절대 감소40명 이상 분류: 기존승차규모/지역교통/결합 비교.
126개/40명 이상 감소63 그 외63, 중첩 역별 LOSO, 내부 BA 선택, 임계값0.5 고정.
외부 결과를 보고 최종 모형을 고르면 낙관성이 남는다. 결과 파일 저장 없음.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,roc_auc_score,
    brier_score_loss,log_loss,precision_score,recall_score,f1_score,confusion_matrix)

HERE=Path(__file__).resolve().parent
BASE=['log1p_walk_min','competing_share','pre_share_pct','log1p_floor_area','주거지역 비율(%)','log1p_pop_20_50']

def metrics(y,p):
    pred=p>=.5
    return dict(Accuracy=accuracy_score(y,pred),BA=balanced_accuracy_score(y,pred),
      Precision=precision_score(y,pred,zero_division=0),Recall=recall_score(y,pred,zero_division=0),
      Specificity=recall_score(y,pred,pos_label=0,zero_division=0),F1=f1_score(y,pred,zero_division=0),
      AUC=roc_auc_score(y,p),Brier=brier_score_loss(y,p),LogLoss=log_loss(y,p,labels=[0,1]))

def evaluate():
    spec=importlib.util.spec_from_file_location('base29',next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob('12-2_Ridge_Logistic_예측성능_개선실험.py')))
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    d=b.load_data().reset_index(drop=True)
    assert len(d)==126 and d.share_increase.sum()==66
    d['share_increase'] = ((d.board_2024_all-d.board_2025_all)>=40).astype(int)
    assert d.share_increase.sum()==63
    region=['log1p_walk_min','competing_share','log1p_floor_area','주거지역 비율(%)','log1p_pop_20_50','bus_time_mean','ic_access']
    feature_specs=[('승차인원만',['log1p_board_2024']),('지역교통만',region),('승차인원+지역교통',['log1p_board_2024']+region)]
    specs=[(name,features,b.make_pipeline(features,'linear'),{'model__C':b.C_VALUES}) for name,features in feature_specs]
    rows=[];probabilities={};summaries=[];matrices={}
    for name,features,pipe,grid in specs:
      probs=np.full(len(d),np.nan)
      for station in d.nearest_station.unique():
        tr=d[d.nearest_station!=station];te=d[d.nearest_station==station]
        cv=list(LeaveOneGroupOut().split(tr,tr.share_increase,tr.nearest_station))
        search=GridSearchCV(pipe,grid,scoring='balanced_accuracy',cv=cv,n_jobs=1,refit=True,error_score='raise')
        search.fit(tr[features],tr.share_increase)
        pr=search.predict_proba(te[features])[:,1];probs[te.index]=pr
        rows.append(dict(모형=name,검증역=station,n=len(te),내부BA=search.best_score_,설정=str(search.best_params_),**metrics(te.share_increase,pr)))
      probabilities[name]=probs
      summaries.append(dict(모형=name,**metrics(d.share_increase,probs)))
      matrices[name]=confusion_matrix(d.share_increase,probs>=.5).tolist()
    folds=pd.DataFrame(rows)
    # 모형 종류도 내부 성능만으로 선택한 절차를 함께 평가.
    chosen=folds.sort_values('내부BA',ascending=False,kind='stable').groupby('검증역',sort=False).head(1)
    selected=np.full(len(d),np.nan)
    for _,r in chosen.iterrows():
      ix=d.index[d.nearest_station==r['검증역']]
      selected[ix]=probabilities[r['모형']][ix]
    summaries.append(dict(모형='내부에서 모형 종류까지 선택',**metrics(d.share_increase,selected)))
    matrices['내부에서 모형 종류까지 선택']=confusion_matrix(d.share_increase,selected>=.5).tolist()
    return pd.DataFrame(summaries),folds,chosen,matrices

if __name__=='__main__':
    summary,folds,chosen,matrices=evaluate()
    print('정답: 2024-2025 승차인원 >=40이면1, 나머지0. 임계값0.5. 내부 BA로 C 선택.')
    print('===== 전체 OOF =====');print(summary.round(4).to_string(index=False))
    print('===== 역별 =====');print(folds.round(4).to_string(index=False))
    print('===== 내부 선택 =====');print(chosen[['검증역','모형','내부BA','설정']].round(4).to_string(index=False))
    print('===== 오차행렬 행=실제0/1 열=예측0/1 =====')
    for k,v in matrices.items():print(k,v)
    print('모든 설정은 외부 검증역을 제외하고 선택. 같은 3개 역 반복 탐색이므로 독립검증 아님. 결과 파일 저장 없음.')
