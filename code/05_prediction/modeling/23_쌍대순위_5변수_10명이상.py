"""로그연면적 포함5변수 순위모형: 기초10명 이상102개 고정 실험.

권역 총량/개통전 비중/정답 라벨은 원래126개 기준 유지.
새102개로 학습역 내부 C 선택 후 외부 LOSO. 기존5변수와 같은102개에서도 비교.
결과로 임계값을 변경하지 않으며 CSV/모델 파일을 저장하지 않는다.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('floor_base_23',next((next(p for p in Path(__file__).resolve().parents if (p / 'config.py').is_file() and (p / 'code').is_dir()) / 'code').rglob('22_쌍대순위_로그연면적추가_비교.py')))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
STATIONS=['운정중앙','킨텍스','대곡']

def score_tables(frame,name,ranker):
    rows=[];tops=[];parts=[]
    for station,g in frame.groupby('nearest_station',sort=False):
        ordered=g.sort_values('점수',ascending=False).copy()
        ordered['순위']=np.arange(1,len(g)+1)
        ordered['상위20']=ordered['순위']<=int(np.ceil(len(g)*.2))
        s=ordered[ordered['상위20']];p=(s.Y>0).mean()
        loss,count=ranker.pairwise_log_loss(g,g['점수'].to_numpy())
        rows.append({'모형':name,'검증역':station,'n':len(g),'AUC':roc_auc_score(g.share_increase,g['점수']),
                     'PairwiseLoss':loss,'비교쌍수':count,'권역상대유지비율':(g.Y>0).mean(),
                     '상위수':len(s),'적중수':int((s.Y>0).sum()),'Precision20':p,
                     'Lift20':p/(g.Y>0).mean(),'상위실제Y중앙값':s.Y.median()})
        for rate in [.1,.2,.3]:
            k=int(np.ceil(len(g)*rate));s=ordered.head(k);p=(s.Y>0).mean()
            tops.append({'모형':name,'검증역':station,'상위비율':rate,'K':k,'적중수':int((s.Y>0).sum()),
                         'Precision':p,'Lift':p/(g.Y>0).mean(),'실제Y중앙값':s.Y.median()})
        ordered['모형']=name;parts.append(ordered)
    return pd.DataFrame(rows),pd.DataFrame(tops),pd.concat(parts,ignore_index=True)

def evaluate():
    r=base.load_ranker(True)
    original=r.load_base_module().load_data()
    assert len(original)==126 and original['셀 ID'].is_unique
    totals=original.groupby('nearest_station')[['board_2024_all','board_2025_all']].sum()
    original['R_s']=original.nearest_station.map(100*(totals.board_2025_all/totals.board_2024_all-1))
    original['R_i']=100*(original.board_2025_all/original.board_2024_all-1)
    original['Y']=original.R_i-original.R_s
    data=original[original.board_2024_all>=10].copy()
    assert len(data)==102 and data[r.FEATURES].notna().all().all()
    assert np.isfinite(data[r.FEATURES].to_numpy()).all()
    assert ((data.Y>0).astype(int)==data.share_increase).all()
    assert data.Y.equals(original.loc[data.index,'Y'])
    selected=[];settings=[];coverage=[];coefs=[]
    for station in STATIONS:
        train=data[data.nearest_station!=station]
        test=data[data.nearest_station==station].copy()
        c,inner=r.choose_c_inner_loso(train)
        x,y=r.make_pairs(train);model=r.make_model(c).fit(x,y)
        test['점수']=r.cell_scores(model,test);selected.append(test)
        settings.append({'검증역':station,'학습n':len(train),'검증n':len(test),'학습양방향쌍수':len(y),'선택C':c,'내부역평균AUC':inner})
        prior=original[original.nearest_station==station]
        coverage.append({'권역':station,'원래n':len(prior),'10명이상n':len(test),'제외n':len(prior)-len(test),
                         '유지된개통전승차비율':test.board_2024_all.sum()/prior.board_2024_all.sum(),
                         '고정권역변화율':test.R_s.iloc[0]})
        coefs.append({'검증역':station,**dict(zip(r.FEATURES,model.named_steps['ranker'].coef_[0]))})
    new=pd.concat(selected,ignore_index=True)
    rows,tops,oof=score_tables(new,'10명이상재학습',r)
    # 공통102개에서 기존 모형을 평가. 정답/상위격자 수를 같게 맞춘다.
    old_oof,_,old_overall,_=base.evaluate()
    old=old_oof[(old_oof['모형']=='로그연면적추가5변수')&(old_oof.board_2024_all>=10)].copy()
    old=old.drop(columns=['모형','순위','상위20'])
    old['R_s']=old.nearest_station.map(100*(totals.board_2025_all/totals.board_2024_all-1))
    old['R_i']=100*(old.board_2025_all/old.board_2024_all-1);old['Y']=old.R_i-old.R_s
    assert set(new['셀 ID'])==set(old['셀 ID'])
    assert np.allclose(new.set_index('셀 ID').sort_index().Y,old.set_index('셀 ID').sort_index().Y)
    oldrows,oldtops,oldoof=score_tables(old,'전체학습_같은102개평가',r)
    rows=pd.concat([oldrows,rows],ignore_index=True);tops=pd.concat([oldtops,tops],ignore_index=True)
    overall=[]
    for name,g in rows.groupby('모형',sort=False):
        top=(oldoof if name.startswith('전체') else oof).loc[lambda x:x['상위20']]
        overall.append({'모형':name,'평가n':g.n.sum(),'역평균AUC':g.AUC.mean(),
                        '격자수가중AUC':np.average(g.AUC,weights=g.n),
                        '전체쌍Loss':np.average(g.PairwiseLoss,weights=g['비교쌍수']),
                        '상위수':len(top),'적중수':int((top.Y>0).sum()),
                        '상위Precision':(top.Y>0).mean(),'상위실제Y중앙값':top.Y.median()})
    return oof,rows,pd.DataFrame(overall),tops,pd.DataFrame(settings),pd.DataFrame(coverage),pd.DataFrame(coefs),old_overall

def main():
    oof,rows,overall,tops,settings,coverage,coefs,old=evaluate()
    for title,t in [('대상 범위',coverage),('학습 설정',settings),('같은102개 전체비교',overall),('역별',rows),('상위비율별',tops)]:
        print('\n=====',title,'=====');print(t.round(4).to_string(index=False))
    print('\n===== 새모형 상위21개 =====')
    print(oof[oof['상위20']].sort_values(['nearest_station','순위'])[['nearest_station','셀 ID','순위','board_2024_all','board_2025_all','R_s','R_i','Y']].round(3).to_string(index=False))
    print('\n원래126개 결과와102개 결과의 차이를 그대로 성능 개선이라고 해석하지 않는다.')
    print('기준10명은 사용자 지정. 기존 전체결과를 보존하며 이번 결과로 기준/C 후보를 추가 조정하지 않는다.')
    print('C의 기초10명 미만은 판단 유보로 표시해야 하며 0 예측으로 대체하지 않는다.')
    print('CSV/모델/그림 파일 저장 없음.')

if __name__=='__main__':main()
