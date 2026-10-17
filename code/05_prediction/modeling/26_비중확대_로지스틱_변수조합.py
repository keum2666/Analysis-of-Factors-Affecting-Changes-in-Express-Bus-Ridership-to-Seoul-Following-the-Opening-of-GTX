"""비중 확대/비확대 분류: 로그 보행 고정·개통 전 승차인원 제외 10후보, 중첩 역별 검증.
결과는 화면 출력만. 기존 분석 수정 및 결과 파일 저장 없음.
내부 Balanced Accuracy로 변수조합과 C 선택, 분류 기준 0.5 고정.
외부 최고 후보 성능은 탐색값이며 최종 성능으로 주장하지 않는다.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, brier_score_loss, log_loss, confusion_matrix

HERE=Path(__file__).resolve().parent

def evaluate(walk_modes=('log1p_walk_min',)):
    s=importlib.util.spec_from_file_location('base26',next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob('12-2_Ridge_Logistic_예측성능_개선실험.py')))
    b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
    d=b.load_data().reset_index(drop=True)
    assert len(d)==126 and d['셀 ID'].is_unique
    demands=[('수요없음',[]),('비중',['pre_share_pct'])]
    regions=[('없음',[]),('로그연면적',['log1p_floor_area']),('연면적+주거',['log1p_floor_area','주거지역 비율(%)']),('연면적+주거+인구',['log1p_floor_area','주거지역 비율(%)','log1p_pop_20_50']),('전체지역',['log1p_floor_area','주거지역 비율(%)','log1p_pop_20_50','bus_time_mean','ic_access'])]
    rows=[];predictions={};features_by_name={}
    for walk in walk_modes:
      for dn,df in demands:
       for rn,rf in regions:
        name=('로그보행' if walk.startswith('log') else '보행')+' / '+dn+' / '+rn
        features=[walk,'competing_share']+df+rf
        features_by_name[name]=features
        probs=np.full(len(d),np.nan)
        for station in d.nearest_station.unique():
          tr=d[d.nearest_station!=station];te=d[d.nearest_station==station]
          cv=list(LeaveOneGroupOut().split(tr, tr.share_increase,tr.nearest_station))
          search=GridSearchCV(b.make_pipeline(features,'linear'),{'model__C':b.C_VALUES},scoring='balanced_accuracy',cv=cv,n_jobs=1)
          search.fit(tr[features],tr.share_increase)
          prob=search.predict_proba(te[features])[:,1];pred=(prob>=.5).astype(int)
          probs[te.index]=prob
          rows.append(dict(후보명=name,검증역=station,n=len(te),변수수=len(features),inner=search.best_score_,C=search.best_params_['model__C'],Accuracy=accuracy_score(te.share_increase,pred),BA=balanced_accuracy_score(te.share_increase,pred),AUC=roc_auc_score(te.share_increase,prob),Brier=brier_score_loss(te.share_increase,prob)))
        predictions[name]=probs
    folds=pd.DataFrame(rows)
    def metrics(y,p):
      return dict(Accuracy=accuracy_score(y,p>=.5),BA=balanced_accuracy_score(y,p>=.5),AUC=roc_auc_score(y,p),Brier=brier_score_loss(y,p),LogLoss=log_loss(y,p))
    summary=[]
    for name,p in predictions.items():
      g=folds[folds['후보명']==name]
      row=dict(모형=name,**metrics(d.share_increase,p),역평균AUC=g.AUC.mean())
      row.update({str(s)+'AUC':v for s,v in zip(g['검증역'],g.AUC)})
      summary.append(row)
    # 외부 정답을 사용하지 않고 학습역 내부 성능으로 후보 선택.
    chosen=folds.sort_values(['inner','변수수','후보명'],ascending=[False,True,True]).groupby('검증역',sort=False).head(1)
    nested=np.full(len(d),np.nan)
    for _,row in chosen.iterrows():
      ix=d.index[d.nearest_station==row['검증역']]
      nested[ix]=predictions[row['후보명']][ix]
    assert np.isfinite(nested).all()
    return pd.DataFrame(summary).sort_values(['BA','AUC'],ascending=False),chosen,metrics(d.share_increase,nested),confusion_matrix(d.share_increase,nested>=.5),features_by_name

if __name__=='__main__':
    summary,chosen,nested,cm,features=evaluate()
    print('log(1+보행시간) 고정. 126개, 실제 확대66/비확대60. L2 로지스틱 분류, 임계값0.5. 내부 선택 기준 Balanced Accuracy.')
    print(summary.round(4).to_string(index=False))
    print('\n내부에서 변수+C 선택한 외부 fold');print(chosen.round(4).to_string(index=False))
    print('선택절차 OOF:',nested);print('오차행렬 행=실제0/1 열=예측0/1:',cm.tolist())
    print('탐색 최고 조합의 입력:',features[summary.iloc[0]['모형']])
    print('후보별 외부 최고점은 탐색값. 중첩 결과도 반복 사용한 3개 역이므로 독립검증이 아님. 결과 파일 저장 없음.')
