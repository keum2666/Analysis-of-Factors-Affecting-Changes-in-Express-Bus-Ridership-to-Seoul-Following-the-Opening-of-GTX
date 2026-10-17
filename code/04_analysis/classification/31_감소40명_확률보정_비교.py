"""40명 이상 감소 분류: 내부 BA/LogLoss 튜닝과 역별 sigmoid 보정.
지역교통7변수 고정. 외부 LOSO. 보정 비교는 C=1 사전고정해
보정 정답을 C 선택에 재사용하지 않는다. ensemble=False: 내부 역별
OOF 점수로 sigmoid 학습 후 외부학습 전체로 기본모형 재학습.
결과는 화면 출력만, 외부 예측으로 보정/임계값 선택 금지.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,roc_auc_score,
 brier_score_loss,log_loss,confusion_matrix,precision_score,recall_score)
HERE=Path(__file__).resolve().parent
FEATURES=['log1p_walk_min','competing_share','log1p_floor_area','주거지역 비율(%)','log1p_pop_20_50','bus_time_mean','ic_access']

def metrics(y,p):
 return dict(Accuracy=accuracy_score(y,p>=.5),BA=balanced_accuracy_score(y,p>=.5),AUC=roc_auc_score(y,p),Brier=brier_score_loss(y,p),LogLoss=log_loss(y,p),Precision=precision_score(y,p>=.5,zero_division=0),Recall=recall_score(y,p>=.5,zero_division=0))

def evaluate():
 s=importlib.util.spec_from_file_location('base31',next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob('12-2_Ridge_Logistic_예측성능_개선실험.py')));b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
 d=b.load_data().reset_index(drop=True);y=((d.board_2024_all-d.board_2025_all)>=40).astype(int)
 assert len(d)==126 and y.sum()==63
 names=['BA로 C선택','LogLoss로 C선택','C1 보정전','C1 Sigmoid보정']
 ps={name:np.full(len(d),np.nan) for name in names};folds=[]
 for station in d.nearest_station.unique():
  ti=np.flatnonzero(d.nearest_station!=station);vi=np.flatnonzero(d.nearest_station==station)
  X=d.iloc[ti][FEATURES];yt=y.iloc[ti];Xv=d.iloc[vi][FEATURES]
  cv=list(LeaveOneGroupOut().split(X,yt,d.iloc[ti].nearest_station))
  for name,score in zip(names[:2],['balanced_accuracy','neg_log_loss']):
   search=GridSearchCV(b.make_pipeline(FEATURES,'linear'),{'model__C':b.C_VALUES},scoring=score,cv=cv,n_jobs=1,error_score='raise').fit(X,yt)
   prob=search.predict_proba(Xv)[:,1];ps[name][vi]=prob
   folds.append(dict(모형=name,검증역=station,C=search.best_params_['model__C'],**metrics(y.iloc[vi],prob)))
  raw=b.make_pipeline(FEATURES,'linear').set_params(model__C=1.0).fit(X,yt)
  calibrated=CalibratedClassifierCV(estimator=b.make_pipeline(FEATURES,'linear').set_params(model__C=1.0),method='sigmoid',cv=cv,ensemble=False,n_jobs=1).fit(X,yt)
  for name,model in [(names[2],raw),(names[3],calibrated)]:
   prob=model.predict_proba(Xv)[:,1];ps[name][vi]=prob
   folds.append(dict(모형=name,검증역=station,C=1.,**metrics(y.iloc[vi],prob)))
 rows=[];cms={};bins=[]
 for name,pr in ps.items():
  assert np.isfinite(pr).all()
  rows.append(dict(모형=name,**metrics(y,pr)))
  cms[name]=confusion_matrix(y,pr>=.5).tolist()
  frame=pd.DataFrame({'p':pr,'y':y});frame['bin']=pd.cut(frame.p,[0,.2,.4,.6,.8,1],include_lowest=True)
  for interval,g in frame.groupby('bin',observed=True):
   bins.append(dict(모형=name,구간=str(interval),n=len(g),평균예측=g.p.mean(),실제비율=g.y.mean()))
 return pd.DataFrame(rows),pd.DataFrame(folds),cms,pd.DataFrame(bins)

if __name__=='__main__':
 summary,folds,cms,bins=evaluate()
 print('===== 전체 OOF =====');print(summary.round(4).to_string(index=False))
 print('===== 역별 =====');print(folds.round(4).to_string(index=False))
 print('===== 오차행렬 행실제0/1 열예측0/1 =====')
 for name,cm in cms.items():print(name,cm)
 print('===== 보정 점검 구간 =====');print(bins.round(4).to_string(index=False))
 print('C1 보정전/후가 보정 자체의 비교. BA/LogLoss 튜닝과의 비교는 규제도 다름. 외부자료로 최고 모형 확정 금지. 결과 파일 저장 없음.')
